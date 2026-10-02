//! Cost-effective gradient boosting: split, coupled (once per forest) and
//! lazy (once per row) feature penalties subtracted from split gains.
//!
//! upstream: src/treelearner/cost_effective_gradient_boosting.hpp

use crate::config::Config;
use crate::consts::K_MIN_SCORE;
use crate::error::{LgbmError, Result};

use super::split::SplitInfo;

pub(crate) struct Cegb {
    tradeoff: f64,
    penalty_split: f64,
    /// By real feature; empty when not set.
    coupled: Vec<f64>,
    /// By real feature; empty when not set.
    lazy: Vec<f64>,
    num_features: usize,
    /// Upstream's inner index of each inner feature, which numbers the
    /// blocks of `feature_used_in_data` (visible when the bag size varies).
    upstream_inner: Vec<usize>,
    /// upstream `splits_per_leaf_`: the last split found for each (leaf,
    /// inner feature), before the penalties.
    splits_per_leaf: Vec<SplitInfo>,
    /// upstream `is_feature_used_in_split_` (by inner feature, kept across trees).
    is_feature_used_in_split: Vec<bool>,
    /// upstream `feature_used_in_data_`: bit `stride * upstream_inner + row`,
    /// kept across trees.
    feature_used_in_data: Vec<u64>,
}

/// The tree learner's view of row indices: upstream's learner trains on a
/// copy of the bag when bagging uses a subset, so its row indices are
/// positions in the bag and its row count is the bag size.
pub(crate) struct RowView<'a> {
    pub num_data: usize,
    /// Position of each global row in the bag (subset mode only).
    pub local: Option<&'a [u32]>,
}

impl RowView<'_> {
    #[inline]
    fn index(&self, row: u32) -> usize {
        match self.local {
            Some(l) => l[row as usize] as usize,
            None => row as usize,
        }
    }
}

impl Cegb {
    /// upstream `CostEfficientGradientBoosting::IsEnable`.
    pub fn is_enable(cfg: &Config) -> bool {
        !(cfg.cegb_tradeoff >= 1.0
            && cfg.cegb_penalty_split <= 0.0
            && cfg.cegb_penalty_feature_coupled.is_empty()
            && cfg.cegb_penalty_feature_lazy.is_empty())
    }

    /// The size checks of upstream `Init`.
    pub fn check(cfg: &Config, num_total_features: usize) -> Result<()> {
        if !Self::is_enable(cfg) {
            return Ok(());
        }
        let coupled = &cfg.cegb_penalty_feature_coupled;
        if !coupled.is_empty() && coupled.len() != num_total_features {
            return Err(LgbmError::InvalidParameter(
                "cegb_penalty_feature_coupled should be the same size as feature number.".into(),
            ));
        }
        let lazy = &cfg.cegb_penalty_feature_lazy;
        if !lazy.is_empty() && lazy.len() != num_total_features {
            return Err(LgbmError::InvalidParameter(
                "cegb_penalty_feature_lazy should be the same size as feature number.".into(),
            ));
        }
        Ok(())
    }

    /// upstream `Init` (first call); sizes must have passed [`Cegb::check`].
    pub fn new(cfg: &Config, num_leaves: usize, upstream_inner: Vec<usize>, num_data: usize) -> Self {
        let num_features = upstream_inner.len();
        let lazy = cfg.cegb_penalty_feature_lazy.clone();
        let bits = if lazy.is_empty() { 0 } else { (num_features * num_data).div_ceil(64) + 1 };
        Self {
            tradeoff: cfg.cegb_tradeoff,
            penalty_split: cfg.cegb_penalty_split,
            coupled: cfg.cegb_penalty_feature_coupled.clone(),
            lazy,
            num_features,
            upstream_inner,
            splits_per_leaf: vec![SplitInfo::default(); num_leaves * num_features],
            is_feature_used_in_split: vec![false; num_features],
            feature_used_in_data: vec![0; bits],
        }
    }

    /// upstream `BeforeTrain`.
    pub fn before_train(&mut self) {
        for s in self.splits_per_leaf.iter_mut() {
            s.reset();
        }
    }

    fn bit(&self, pos: usize) -> bool {
        self.feature_used_in_data.get(pos / 64).is_some_and(|w| (w >> (pos % 64)) & 1 == 1)
    }

    fn set_bit(&mut self, pos: usize) {
        if pos / 64 >= self.feature_used_in_data.len() {
            self.feature_used_in_data.resize(pos / 64 + 1, 0);
        }
        self.feature_used_in_data[pos / 64] |= 1 << (pos % 64);
    }

    /// upstream `DeltaGain`: the penalty for `split` (found for inner
    /// feature `inner` on `leaf`, whose rows are `rows`), which is recorded.
    pub fn delta_gain(
        &mut self,
        inner: usize,
        real: usize,
        leaf: usize,
        split: &SplitInfo,
        rows: &[u32],
        view: &RowView<'_>,
    ) -> f64 {
        let mut delta = self.tradeoff * self.penalty_split * rows.len() as f64;
        if !self.coupled.is_empty() && !self.is_feature_used_in_split[inner] {
            delta += self.tradeoff * self.coupled[real];
        }
        if !self.lazy.is_empty() {
            delta += self.tradeoff * self.ondemand_costs(inner, real, rows, view);
        }
        self.splits_per_leaf[leaf * self.num_features + inner] = split.clone();
        delta
    }

    /// upstream `CalculateOndemandCosts` (a sequential sum, as upstream).
    fn ondemand_costs(&self, inner: usize, real: usize, rows: &[u32], view: &RowView<'_>) -> f64 {
        let penalty = self.lazy[real];
        let base = view.num_data * self.upstream_inner[inner];
        let mut total = 0.0f64;
        for &r in rows {
            if self.bit(base + view.index(r)) {
                continue;
            }
            total += penalty;
        }
        total
    }

    /// upstream `UpdateLeafBestSplits`, before `best_leaf` (with rows
    /// `rows`) is split on `best`; `best_split_per_leaf` holds the tree's
    /// current leaves.
    pub fn update_leaf_best_splits(
        &mut self,
        best_leaf: usize,
        best: &SplitInfo,
        inner: usize,
        best_split_per_leaf: &mut [SplitInfo],
        rows: &[u32],
        view: &RowView<'_>,
    ) {
        if !self.coupled.is_empty() && !self.is_feature_used_in_split[inner] {
            self.is_feature_used_in_split[inner] = true;
            let add = self.tradeoff * self.coupled[best.feature as usize];
            for (i, cur) in best_split_per_leaf.iter_mut().enumerate() {
                if i == best_leaf {
                    continue;
                }
                let split = &mut self.splits_per_leaf[i * self.num_features + inner];
                split.gain += add;
                // leaves that cannot split are left alone
                if cur.gain > K_MIN_SCORE && split.better_than(cur) {
                    *cur = split.clone();
                }
            }
        }
        if !self.lazy.is_empty() {
            let base = view.num_data * self.upstream_inner[inner];
            for &r in rows {
                self.set_bit(base + view.index(r));
            }
        }
    }
}
