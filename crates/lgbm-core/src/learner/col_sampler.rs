//! Per-tree and per-node feature subsampling.
//!
//! upstream: src/treelearner/col_sampler.hpp (`ColSampler`). One generator
//! seeded with `feature_fraction_seed` serves both the per-tree and per-node
//! draws, in upstream's call order.

use std::collections::HashSet;

use crate::config::Config;
use crate::dataset::Dataset;
use crate::random::Random;

#[derive(Debug, Clone)]
pub struct ColSampler {
    fraction_bynode: f64,
    need_reset_bytree: bool,
    used_cnt_bytree: i32,
    random: Random,
    /// By inner feature index.
    is_feature_used: Vec<bool>,
    /// Positions into `valid_feature_indices` picked for the current tree.
    used_feature_indices: Vec<i32>,
    /// Real indices of the trainable features, ascending.
    valid_feature_indices: Vec<usize>,
    /// Inner index of each entry of `valid_feature_indices`.
    valid_inner: Vec<usize>,
    /// Sets of real feature indices.
    interaction_constraints: Vec<HashSet<i32>>,
}

/// upstream: `ColSampler::GetCnt`.
fn get_cnt(total_cnt: usize, fraction: f64) -> i32 {
    let min = 1.min(total_cnt as i32);
    let used = (total_cnt as f64 * fraction + 0.5f32 as f64) as i32;
    used.max(min)
}

impl ColSampler {
    /// upstream: constructor followed by `SetTrainingData` (which draws the
    /// first per-tree sample).
    pub fn new(data: &Dataset, cfg: &Config) -> Self {
        let nf = data.num_features();
        let mut valid: Vec<(usize, usize)> = (0..nf).map(|f| (data.real_feature_index(f), f)).collect();
        valid.sort_unstable();
        let need_reset_bytree = cfg.feature_fraction < 1.0;
        let used_cnt_bytree = if need_reset_bytree { get_cnt(valid.len(), cfg.feature_fraction) } else { valid.len() as i32 };
        let mut s = Self {
            fraction_bynode: cfg.feature_fraction_bynode,
            need_reset_bytree,
            used_cnt_bytree,
            random: Random::new(cfg.feature_fraction_seed),
            is_feature_used: vec![true; nf],
            used_feature_indices: Vec::new(),
            valid_feature_indices: valid.iter().map(|v| v.0).collect(),
            valid_inner: valid.iter().map(|v| v.1).collect(),
            interaction_constraints: cfg
                .interaction_constraints_vector
                .iter()
                .map(|c| c.iter().copied().collect())
                .collect(),
        };
        s.reset_by_tree();
        s
    }

    /// upstream: `ColSampler::ResetByTree`.
    pub fn reset_by_tree(&mut self) {
        if self.need_reset_bytree {
            self.is_feature_used.iter_mut().for_each(|u| *u = false);
            self.used_feature_indices = self.random.sample(self.valid_feature_indices.len() as i32, self.used_cnt_bytree);
            for &i in &self.used_feature_indices {
                self.is_feature_used[self.valid_inner[i as usize]] = true;
            }
        }
    }

    pub fn is_feature_used_bytree(&self) -> &[bool] {
        &self.is_feature_used
    }

    /// Real features a branch with these split features may still use, or
    /// `None` without interaction constraints.
    fn allowed_features(&self, branch_features: &[i32]) -> Option<HashSet<i32>> {
        if self.interaction_constraints.is_empty() {
            return None;
        }
        let mut allowed: HashSet<i32> = branch_features.iter().copied().collect();
        for c in &self.interaction_constraints {
            if branch_features.iter().all(|f| c.contains(f)) {
                allowed.extend(c.iter().copied());
            }
        }
        Some(allowed)
    }

    /// upstream: `ColSampler::GetByNode` (by inner feature index);
    /// `branch_features` are the real split features on the leaf's path.
    pub fn get_by_node(&mut self, branch_features: &[i32]) -> Vec<bool> {
        let nf = self.is_feature_used.len();
        let allowed = self.allowed_features(branch_features);
        if self.fraction_bynode >= 1.0 {
            let Some(allowed) = allowed else { return vec![true; nf] };
            let mut ret = vec![false; nf];
            for (pos, &real) in self.valid_feature_indices.iter().enumerate() {
                if allowed.contains(&(real as i32)) {
                    ret[self.valid_inner[pos]] = true;
                }
            }
            return ret;
        }
        let mut ret = vec![false; nf];
        if self.need_reset_bytree {
            let mut cnt = get_cnt(self.used_feature_indices.len(), self.fraction_bynode);
            let candidates: Vec<i32> = match &allowed {
                None => self.used_feature_indices.clone(),
                Some(a) => {
                    let f: Vec<i32> = self
                        .used_feature_indices
                        .iter()
                        .copied()
                        .filter(|&i| a.contains(&(self.valid_feature_indices[i as usize] as i32)))
                        .collect();
                    cnt = cnt.min(f.len() as i32);
                    f
                }
            };
            let sampled = self.random.sample(candidates.len() as i32, cnt);
            for s in sampled {
                ret[self.valid_inner[candidates[s as usize] as usize]] = true;
            }
        } else {
            let mut cnt = get_cnt(self.valid_feature_indices.len(), self.fraction_bynode);
            let candidates: Vec<usize> = match &allowed {
                None => (0..self.valid_feature_indices.len()).collect(),
                Some(a) => {
                    let f: Vec<usize> = (0..self.valid_feature_indices.len())
                        .filter(|&p| a.contains(&(self.valid_feature_indices[p] as i32)))
                        .collect();
                    cnt = cnt.min(f.len() as i32);
                    f
                }
            };
            let sampled = self.random.sample(candidates.len() as i32, cnt);
            for s in sampled {
                ret[self.valid_inner[candidates[s as usize]]] = true;
            }
        }
        ret
    }
}

#[cfg(test)]
mod tests {
    use super::get_cnt;

    #[test]
    fn get_cnt_matches_upstream_rounding() {
        assert_eq!(get_cnt(10, 0.55), 6);
        assert_eq!(get_cnt(10, 0.01), 1);
        assert_eq!(get_cnt(0, 0.5), 0);
        assert_eq!(get_cnt(7, 1.0), 7);
    }
}
