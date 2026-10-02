//! Per-iteration row subsampling: bagging and GOSS.
//!
//! upstream: src/boosting/bagging.hpp (`BaggingSampleStrategy`),
//! src/boosting/goss.hpp (`GOSSStrategy`), include/LightGBM/sample_strategy.h.
//!
//! Rows are drawn from one generator per block of [`BAGGING_RAND_BLOCK`]
//! rows (`Random(bagging_seed + block)`), so bagging results do not depend
//! on the thread count. GOSS selects its top rows per partition chunk, and
//! the chunks come from upstream `Threading::BlockInfoForceSize` over
//! `num_threads`, so GOSS results depend on the thread count exactly as
//! upstream's do. Upstream's "subset" mode copies the in-bag rows into a
//! smaller dataset; it visits rows in the same order as training on the
//! in-bag indices, which is what this implementation does.

use crate::array_args::arg_max_at_k;
use crate::config::Config;
use crate::dataset::Dataset;
use crate::error::{LgbmError, Result};
use crate::objective::Objective;
use crate::random::Random;

/// upstream: `SampleStrategy::bagging_rand_block_`.
pub const BAGGING_RAND_BLOCK: usize = 1024;

#[derive(Debug, Clone, Copy)]
enum Kind {
    Bagging { fraction: f64, freq: i32, balanced: Option<(f64, f64)> },
    Goss { top_rate: f64, other_rate: f64, learning_rate: f64 },
}

#[derive(Debug, Clone)]
pub struct SampleStrategy {
    kind: Kind,
    num_data: usize,
    num_tree_per_iteration: usize,
    num_threads: usize,
    rands: Vec<Random>,
    /// In-bag rows (ascending) followed by out-of-bag rows.
    indices: Vec<u32>,
    bag_cnt: usize,
    need_re_bagging: bool,
    label: Vec<f32>,
    by_query: Option<ByQuery>,
    /// upstream `is_use_subset_`: the tree learner trains on a copy of the
    /// in-bag rows, so its row indices are positions in the bag.
    use_subset: bool,
}

/// upstream `bagging_by_query`: whole queries are drawn (one draw per query
/// from `rands[query / BAGGING_RAND_BLOCK]`, ignoring the balanced fractions).
#[derive(Debug, Clone)]
struct ByQuery {
    boundaries: Vec<i32>,
    /// upstream `bag_query_indices_[..num_sampled_queries_]` (ascending).
    sampled: Vec<u32>,
    /// upstream `is_use_subset_`. Without it upstream rewrites only the
    /// in-bag prefix of `bag_data_indices_`, so the "out-of-bag" rows whose
    /// scores get the tree prediction are whatever an earlier bag (or the
    /// zero-initialized buffer) left there; this reproduces that buffer.
    use_subset: bool,
}

/// upstream: `Threading::BlockInfoForceSize` -> (number of blocks, block size).
fn block_info_force_size(num_threads: usize, cnt: usize, min_cnt_per_block: usize) -> (usize, usize) {
    let nblock = num_threads.min(cnt.div_ceil(min_cnt_per_block));
    if nblock > 1 {
        let size = cnt.div_ceil(nblock);
        (nblock, size.div_ceil(min_cnt_per_block) * min_cnt_per_block)
    } else {
        (nblock, cnt)
    }
}

impl SampleStrategy {
    /// upstream: `CreateSampleStrategy` + `ResetSampleConfig(config, true)`.
    /// Returns `None` when no subsampling is configured.
    pub fn new(
        cfg: &Config,
        data: &Dataset,
        objective: Option<&Objective>,
        num_tree_per_iteration: usize,
        num_threads: usize,
    ) -> Result<Option<Self>> {
        let n = data.num_data();
        let kind = if cfg.data_sample_strategy == "goss" {
            if cfg.top_rate + cfg.other_rate > 1.0 {
                return Err(LgbmError::InvalidParameter(
                    "Check failed: (config_->top_rate + config_->other_rate) <= (1.0f)".into(),
                ));
            }
            if !(cfg.top_rate > 0.0 && cfg.other_rate > 0.0) {
                return Err(LgbmError::InvalidParameter(
                    "Check failed: config_->top_rate > 0.0f && config_->other_rate > 0.0f".into(),
                ));
            }
            if cfg.bagging_freq > 0 && cfg.bagging_fraction != 1.0 {
                return Err(LgbmError::InvalidParameter("Cannot use bagging in GOSS".into()));
            }
            Kind::Goss { top_rate: cfg.top_rate, other_rate: cfg.other_rate, learning_rate: cfg.learning_rate }
        } else {
            let num_pos = objective.map_or(0, |o| o.num_positive_data());
            let balanced = (cfg.pos_bagging_fraction < 1.0 || cfg.neg_bagging_fraction < 1.0) && num_pos > 0;
            if !((cfg.bagging_fraction < 1.0 || balanced) && cfg.bagging_freq > 0) {
                return Ok(None);
            }
            Kind::Bagging {
                fraction: cfg.bagging_fraction,
                freq: cfg.bagging_freq,
                balanced: balanced.then_some((cfg.pos_bagging_fraction, cfg.neg_bagging_fraction)),
            }
        };
        let bag_cnt = match kind {
            Kind::Bagging { balanced: Some((pos, neg)), .. } => {
                let num_pos = objective.map_or(0, |o| o.num_positive_data());
                (num_pos as f64 * pos) as usize + ((n - num_pos) as f64 * neg) as usize
            }
            Kind::Bagging { fraction, .. } => (fraction * n as f64) as usize,
            Kind::Goss { .. } => n,
        };
        let label = match kind {
            Kind::Bagging { balanced: Some(_), .. } => data.label().to_vec(),
            _ => Vec::new(),
        };
        let by_query = match kind {
            Kind::Bagging { freq, .. } if cfg.bagging_by_query => {
                // upstream has zero queries without query data: every bag is
                // empty and no tree can split
                let boundaries = data.metadata.query_boundaries.clone().unwrap_or_else(|| vec![0]);
                // upstream ResetSampleConfig (non-CUDA): subset when
                // average_bag_rate <= 0.5 and num_feature_groups < 100
                let average_bag_rate = (bag_cnt as f64 / n as f64) / freq as f64;
                Some(ByQuery {
                    boundaries,
                    sampled: Vec::new(),
                    use_subset: average_bag_rate <= 0.5 && data.num_feature_groups() < 100,
                })
            }
            _ => None,
        };
        let use_subset = match kind {
            // upstream GOSSStrategy::ResetSampleConfig
            Kind::Goss { top_rate, other_rate, .. } => top_rate + other_rate <= 0.5,
            // upstream BaggingSampleStrategy::ResetSampleConfig (non-CUDA)
            Kind::Bagging { freq, .. } => {
                (bag_cnt as f64 / n as f64) / freq as f64 <= 0.5 && data.num_feature_groups() < 100
            }
        };
        Ok(Some(Self {
            use_subset,
            kind,
            num_data: n,
            num_tree_per_iteration,
            num_threads: num_threads.max(1),
            rands: (0..n.div_ceil(BAGGING_RAND_BLOCK))
                .map(|i| Random::new(cfg.bagging_seed.wrapping_add(i as i32)))
                .collect(),
            indices: vec![0; n],
            bag_cnt,
            need_re_bagging: matches!(kind, Kind::Bagging { .. }),
            label,
            by_query,
        }))
    }

    /// upstream `bag_query_indices_` when `bagging_by_query` is on.
    pub fn sampled_queries(&self) -> Option<&[u32]> {
        self.by_query.as_ref().map(|q| q.sampled.as_slice())
    }

    /// `bagging_by_query` in upstream's subset mode, where `GBDT::TrainOneIter`
    /// compacts the in-bag gradients to the front of the gradient buffer in
    /// place. Only rows of unsampled queries keep those values into the next
    /// iteration (read by the position-bias update), so only this mode needs
    /// the compaction reproduced.
    pub fn by_query_subset(&self) -> bool {
        self.by_query.as_ref().is_some_and(|q| q.use_subset) && self.bag_cnt < self.num_data
    }

    /// upstream `BaggingSampleStrategy::Bagging`, `bagging_by_query` branch.
    fn bag_queries(&mut self, fraction: f64) {
        let q = self.by_query.as_mut().expect("bagging_by_query");
        let num_queries = q.boundaries.len() - 1;
        q.sampled.clear();
        for i in 0..num_queries {
            if (self.rands[i / BAGGING_RAND_BLOCK].next_float() as f64) < fraction {
                q.sampled.push(i as u32);
            }
        }
        let mut cnt = 0usize;
        for &qi in &q.sampled {
            let (start, end) = (q.boundaries[qi as usize] as usize, q.boundaries[qi as usize + 1] as usize);
            for row in start..end {
                self.indices[cnt] = row as u32;
                cnt += 1;
            }
        }
        if q.use_subset {
            // upstream predicts every row of the full dataset in subset mode
            let mut next = 0usize;
            let mut tail = cnt;
            for &qi in &q.sampled {
                let start = q.boundaries[qi as usize] as usize;
                for row in next..start {
                    self.indices[tail] = row as u32;
                    tail += 1;
                }
                next = q.boundaries[qi as usize + 1] as usize;
            }
            for row in next..self.num_data {
                self.indices[tail] = row as u32;
                tail += 1;
            }
        }
        self.bag_cnt = cnt;
    }

    /// upstream: `SampleStrategy::Bagging`. Returns `true` when a new bag was
    /// drawn (the tree learner must then be given [`in_bag`](Self::in_bag)).
    /// GOSS rescales the gradients and Hessians of sampled small-gradient rows.
    pub fn bagging(&mut self, iter: usize, grad: &mut [f32], hess: &mut [f32]) -> bool {
        match self.kind {
            Kind::Bagging { fraction, freq, balanced } => {
                let n = self.num_data;
                if !((self.bag_cnt < n && iter as i32 % freq == 0) || self.need_re_bagging) {
                    return false;
                }
                self.need_re_bagging = false;
                if self.by_query.is_some() {
                    self.bag_queries(fraction);
                    return true;
                }
                // Chunks of upstream's ParallelPartitionRunner are multiples of
                // BAGGING_RAND_BLOCK, so one sequential pass draws the same values.
                let mut left = 0usize;
                let mut right = n;
                for i in 0..n {
                    let r = self.rands[i / BAGGING_RAND_BLOCK].next_float() as f64;
                    let frac = match balanced {
                        Some((pos, neg)) => {
                            if self.label[i] > 0.0 {
                                pos
                            } else {
                                neg
                            }
                        }
                        None => fraction,
                    };
                    if r < frac {
                        self.indices[left] = i as u32;
                        left += 1;
                    } else {
                        right -= 1;
                        self.indices[right] = i as u32;
                    }
                }
                self.bag_cnt = left;
                true
            }
            Kind::Goss { top_rate, other_rate, learning_rate } => {
                let n = self.num_data;
                self.bag_cnt = n;
                // upstream: `iter < static_cast<int>(1.0f / config_->learning_rate)`
                if (iter as i64) < (1.0f32 as f64 / learning_rate) as i64 {
                    return false;
                }
                let (nblock, block) = block_info_force_size(self.num_threads, n, BAGGING_RAND_BLOCK);
                let mut left_all: Vec<u32> = Vec::with_capacity(n);
                let mut right_all: Vec<u32> = Vec::new();
                for b in 0..nblock {
                    let start = b * block;
                    if start >= n {
                        break;
                    }
                    let cnt = block.min(n - start);
                    let (l, r) = self.goss_helper(start, cnt, top_rate, other_rate, grad, hess);
                    left_all.extend_from_slice(&l);
                    right_all.extend_from_slice(&r);
                }
                self.bag_cnt = left_all.len();
                self.indices[..left_all.len()].copy_from_slice(&left_all);
                self.indices[left_all.len()..].copy_from_slice(&right_all);
                true
            }
        }
    }

    /// upstream: `GOSSStrategy::Helper` for rows `start..start + cnt`.
    fn goss_helper(
        &mut self,
        start: usize,
        cnt: usize,
        top_rate: f64,
        other_rate: f64,
        grad: &mut [f32],
        hess: &mut [f32],
    ) -> (Vec<u32>, Vec<u32>) {
        let n = self.num_data;
        let ntpi = self.num_tree_per_iteration;
        let abs_gh = |grad: &[f32], hess: &[f32], row: usize| -> f32 {
            let mut s = 0.0f32;
            for k in 0..ntpi {
                let idx = k * n + row;
                s += (grad[idx] * hess[idx]).abs();
            }
            s
        };
        let mut tmp: Vec<f32> = (0..cnt).map(|i| abs_gh(grad, hess, start + i)).collect();
        let top_k = ((cnt as f64 * top_rate) as i32).max(1);
        let other_k = (cnt as f64 * other_rate) as i32;
        arg_max_at_k(&mut tmp, 0, cnt as i32, top_k - 1);
        let threshold = tmp[(top_k - 1) as usize];
        let multiply = (cnt as i32 - top_k) as f32 / other_k as f32;
        let mut left: Vec<u32> = Vec::new();
        let mut right: Vec<u32> = Vec::new();
        let mut big_weight_cnt: i32 = 0;
        for i in 0..cnt {
            let cur = start + i;
            let g = abs_gh(grad, hess, cur);
            if g >= threshold {
                left.push(cur as u32);
                big_weight_cnt += 1;
            } else {
                let sampled = left.len() as i32 - big_weight_cnt;
                let rest_need = other_k - sampled;
                let rest_all = (cnt - i) as i32 - (top_k - big_weight_cnt);
                let prob = rest_need as f64 / rest_all as f64;
                if (self.rands[cur / BAGGING_RAND_BLOCK].next_float() as f64) < prob {
                    left.push(cur as u32);
                    for k in 0..ntpi {
                        let idx = k * n + cur;
                        grad[idx] *= multiply;
                        hess[idx] *= multiply;
                    }
                } else {
                    right.push(cur as u32);
                }
            }
        }
        (left, right)
    }

    pub fn bag_cnt(&self) -> usize {
        self.bag_cnt
    }

    /// upstream `SampleStrategy::is_use_subset`.
    pub fn is_use_subset(&self) -> bool {
        self.use_subset
    }

    pub fn in_bag(&self) -> &[u32] {
        &self.indices[..self.bag_cnt]
    }

    pub fn out_of_bag(&self) -> &[u32] {
        &self.indices[self.bag_cnt..]
    }
}

#[cfg(test)]
mod tests {
    use super::block_info_force_size;

    #[test]
    fn block_info_matches_upstream() {
        assert_eq!(block_info_force_size(1, 5000, 1024), (1, 5000));
        assert_eq!(block_info_force_size(4, 5000, 1024), (4, 2048));
        assert_eq!(block_info_force_size(8, 3000, 1024), (3, 1024));
        assert_eq!(block_info_force_size(4, 500, 1024), (1, 500));
    }
}
