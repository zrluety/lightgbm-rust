//! Leaf-wise histogram tree learner (CPU, serial algorithm, feature-parallel).
//!
//! upstream: src/treelearner/serial_tree_learner.cpp, data_partition.hpp,
//! leaf_splits.hpp, and `Dataset::ConstructHistograms` / `FixHistogram`.
//!
//! Determinism: leaf sums are sequential. Histograms are built either
//! col-wise (`force_col_wise`: each feature accumulated in data-index order,
//! parallel over features, so results do not depend on the thread count) or
//! row-wise (default: upstream's row-block scheme, see `multi_val_bin`, so
//! results match upstream's row-wise mode for the same thread count; with
//! one thread both modes give identical sums).

mod cegb;
pub mod col_sampler;
pub mod constraints;
mod forced;
pub mod partition;
pub mod split;

use std::sync::Arc;

use serde_json::Value;

use crate::binning::{BinType, MissingType};
use crate::config::Config;
use crate::consts::{K_EPSILON, K_MIN_SCORE};
use crate::dataset::Dataset;
use crate::error::{LgbmError, Result};
use crate::histogram::{self, HistLayout};
use crate::multi_val_bin::{HistSlots, MultiValBin, merge_blocks};
use crate::random::Random;
use crate::threading::{SharedMut, ThreadTeam, resolve_num_threads};
use crate::tree::{SplitArgs, Tree, construct_bitset, find_in_bitset};
pub(crate) use cegb::Cegb;
use cegb::RowView;
pub(crate) use forced::load_forced_splits;
use col_sampler::ColSampler;
use constraints::{LeafConstraints, Method, monotone_split_gain_penalty};
use partition::DataPartition;
use split::{FeatureMeta, SplitInfo, SplitParams, find_best_threshold, root_output};

/// Histograms of all features of one leaf in one flat buffer laid out by
/// [`HistSlots`] (`[g, h]` pairs per bin).
///
/// As upstream's `HistogramPool` (when it holds every leaf), each leaf index
/// owns a buffer that keeps its contents across trees: a build writes only
/// the features upstream writes, so a forced split can read the same left-over
/// values upstream reads.
struct LeafHist {
    data: Vec<f64>,
    splittable: Vec<bool>,
    /// Features built (or subtracted) by the last build of this slot.
    fresh: Vec<bool>,
}

#[derive(Debug, Clone, Copy)]
struct LeafSplits {
    leaf: i32,
    num_data: i32,
    sum_gradients: f64,
    sum_hessians: f64,
    weight: f64,
}

impl LeafSplits {
    fn none() -> Self {
        Self { leaf: -1, num_data: 0, sum_gradients: 0.0, sum_hessians: 0.0, weight: 0.0 }
    }
}

/// Diagnostics captured while training one tree (used by differential tests).
#[derive(Debug, Clone, Default)]
pub struct TreeTrace {
    /// For every split, in order: (leaf, real feature, threshold bin, gain).
    pub splits: Vec<(usize, i32, u32, f64)>,
}

pub struct SerialTreeLearner {
    data: Arc<Dataset>,
    params: SplitParams,
    num_leaves: usize,
    max_depth: i32,
    metas: Vec<FeatureMeta>,
    partition: DataPartition,
    hist_pool: Vec<Option<LeafHist>>,
    /// Whether each slot was built in the current tree.
    hist_built: Vec<bool>,
    /// Row-wise build target when only the by-tree features are copied out
    /// (upstream's sub-column multi-val bin).
    subcol_buf: Vec<f64>,
    /// Upstream inner feature order (its feature-group order).
    group_order: Vec<usize>,
    block_bufs: Vec<Vec<f64>>,
    slots: HistSlots,
    multi_val: Option<MultiValBin>,
    team: ThreadTeam,
    best_split_per_leaf: Vec<SplitInfo>,
    smaller: LeafSplits,
    larger: LeafSplits,
    ordered_g: Vec<f32>,
    ordered_h: Vec<f32>,
    /// Interleaved `[grad, hess]` per row for the row-wise kernel.
    gh: Vec<[f32; 2]>,
    col_sampler: ColSampler,
    /// Per-feature extra-trees generators (upstream `FeatureMetainfo::rand`).
    extra_rands: Option<Vec<Random>>,
    /// Present iff `monotone_constraints` is set (upstream `USE_MC`).
    constraints: Option<LeafConstraints>,
    monotone_penalty: f64,
    /// Whether `interaction_constraints` is set.
    track_branch_features: bool,
    /// Real split features on each leaf's path (empty unless tracked).
    branch_features: Vec<Vec<i32>>,
    /// upstream `cegb_` (present iff `CostEfficientGradientBoosting::IsEnable`).
    cegb: Option<Cegb>,
    /// Upstream's learner row count: the bag size when bagging uses a subset.
    view_num_data: usize,
    /// Position of each row in the bag when bagging uses a subset (kept only
    /// with CEGB, whose lazy penalties index rows that way).
    bag_local: Option<Vec<u32>>,
    /// upstream `forced_split_json_` (`None` for a null document).
    forced_split: Option<Arc<Value>>,
    /// The leaves whose pool slots hold the smaller and larger histograms of
    /// the last histogram build (upstream's `smaller_/larger_leaf_histogram_array_`,
    /// which keep pointing there when `BeforeFindBestSplit` skips a build).
    hist_slots: (i32, i32),
    pub trace: Option<TreeTrace>,
}

impl SerialTreeLearner {
    pub fn new(data: Arc<Dataset>, cfg: &Config) -> Self {
        let metas = (0..data.num_features())
            .map(|f| {
                let m = data.feature_bin_mapper(f);
                FeatureMeta {
                    num_bin: m.num_bin,
                    missing_type: m.missing_type,
                    offset: if m.most_freq_bin == 0 { 1 } else { 0 },
                    default_bin: m.default_bin,
                    most_freq_bin: m.most_freq_bin,
                    bin_type: m.bin_type,
                    monotone_type: cfg.monotone_constraints.get(data.real_feature_index(f)).copied().unwrap_or(0),
                    penalty: cfg.feature_contri.get(data.real_feature_index(f)).copied().unwrap_or(1.0),
                }
            })
            .collect();
        let num_leaves = cfg.num_leaves.max(2) as usize;
        let constraints = (!cfg.monotone_constraints.is_empty()).then(|| {
            LeafConstraints::new(
                Method::parse(&cfg.monotone_constraints_method),
                cfg.monotone_constraints.clone(),
                num_leaves,
                data.num_features(),
            )
        });
        let num_data = data.num_data();
        let slots = HistSlots::new(&data);
        let multi_val =
            (!cfg.force_col_wise && data.num_features() > 0).then(|| MultiValBin::new(&data, &slots));
        let col_sampler = ColSampler::new(&data, cfg);
        let extra_rands = cfg
            .extra_trees
            .then(|| {
                (0..data.num_features())
                    .map(|f| Random::new(cfg.extra_seed.wrapping_add(data.upstream_inner_index(f) as i32)))
                    .collect()
            });
        let cegb = Cegb::is_enable(cfg).then(|| {
            let upstream_inner = (0..data.num_features()).map(|f| data.upstream_inner_index(f)).collect();
            Cegb::new(cfg, num_leaves, upstream_inner, num_data)
        });
        Self {
            cegb,
            view_num_data: num_data,
            bag_local: None,
            forced_split: None,
            hist_slots: (-1, -1),
            col_sampler,
            extra_rands,
            constraints,
            monotone_penalty: cfg.monotone_penalty,
            track_branch_features: !cfg.interaction_constraints_vector.is_empty(),
            branch_features: vec![Vec::new(); num_leaves],
            slots,
            multi_val,
            team: ThreadTeam::new(resolve_num_threads(cfg.num_threads)),
            hist_built: vec![false; num_leaves],
            subcol_buf: Vec::new(),
            group_order: {
                let mut order: Vec<usize> = (0..data.num_features()).collect();
                order.sort_by_key(|&f| data.upstream_inner_index(f));
                order
            },
            block_bufs: Vec::new(),
            data,
            params: SplitParams {
                lambda_l1: cfg.lambda_l1,
                lambda_l2: cfg.lambda_l2,
                max_delta_step: cfg.max_delta_step,
                path_smooth: cfg.path_smooth,
                min_data_in_leaf: cfg.min_data_in_leaf,
                min_sum_hessian_in_leaf: cfg.min_sum_hessian_in_leaf,
                min_gain_to_split: cfg.min_gain_to_split,
                max_cat_to_onehot: cfg.max_cat_to_onehot,
                max_cat_threshold: cfg.max_cat_threshold,
                cat_l2: cfg.cat_l2,
                cat_smooth: cfg.cat_smooth,
                min_data_per_group: cfg.min_data_per_group,
            },
            num_leaves,
            max_depth: cfg.max_depth,
            metas,
            partition: DataPartition::new(num_data, num_leaves),
            hist_pool: (0..num_leaves).map(|_| None).collect(),
            best_split_per_leaf: vec![SplitInfo::default(); num_leaves],
            smaller: LeafSplits::none(),
            larger: LeafSplits::none(),
            ordered_g: Vec::new(),
            ordered_h: Vec::new(),
            gh: Vec::new(),
            trace: None,
        }
    }

    pub fn partition(&self) -> &DataPartition {
        &self.partition
    }

    /// New leaf outputs for `old` from the rows assigned to each leaf by
    /// `leaf_pred`: `decay * old + (1 - decay) * shrinkage * output`.
    ///
    /// upstream: `SerialTreeLearner::FitByExistingTree` after
    /// `DataPartition::ResetByLeafPred` (rows of a leaf in row order). With
    /// path smoothing upstream passes the leaf's parent node index as the
    /// parent output, and so does this.
    pub fn fit_by_existing_tree(&self, old: &Tree, leaf_pred: &[i32], grad: &[f32], hess: &[f32], decay: f64) -> Tree {
        let mut rows_of: Vec<Vec<u32>> = vec![Vec::new(); old.num_leaves];
        for (i, &l) in leaf_pred.iter().enumerate() {
            rows_of[l as usize].push(i as u32);
        }
        let mut tree = old.clone();
        for (leaf, rows) in rows_of.iter().enumerate() {
            let mut sum_grad = 0.0f64;
            let mut sum_hess = K_EPSILON;
            for &r in rows {
                sum_grad += grad[r as usize] as f64;
                sum_hess += hess[r as usize] as f64;
            }
            let smooth = self.params.path_smooth > K_EPSILON && leaf > 0;
            let parent = if smooth { old.leaf_parent[leaf] as f64 } else { 0.0 };
            let output = split::refit_leaf_output(sum_grad, sum_hess, &self.params, smooth, rows.len() as i32, parent);
            let new_output = output * old.shrinkage;
            tree.set_leaf_output(leaf, decay * old.leaf_value[leaf] + (1.0 - decay) * new_output);
        }
        tree
    }

    /// Add the last tree's leaf outputs to the training scores.
    pub fn add_leaf_outputs(&self, leaf_values: &[f64], score: &mut [f64]) {
        self.partition.add_leaf_outputs(&self.team, leaf_values, score);
    }

    /// upstream: `SerialTreeLearner::SetBaggingData`. `used` are the in-bag
    /// rows in ascending order; `None` trains on all rows. Rows are always
    /// partitioned by their global index; `subset` (upstream's subset mode)
    /// only changes how CEGB numbers them.
    pub fn set_bagging_data(&mut self, used: Option<&[u32]>, subset: bool) {
        self.partition.set_used_data_indices(used);
        match used {
            Some(u) if subset => {
                self.view_num_data = u.len();
                if self.cegb.is_some() {
                    let mut local = self.bag_local.take().unwrap_or_default();
                    local.resize(self.data.num_data(), 0);
                    for (pos, &row) in u.iter().enumerate() {
                        local[row as usize] = pos as u32;
                    }
                    self.bag_local = Some(local);
                }
            }
            _ => {
                self.view_num_data = self.data.num_data();
                self.bag_local = None;
            }
        }
    }

    /// Train one tree on `grad`/`hess` (length `num_data`).
    pub fn train(&mut self, grad: &[f32], hess: &[f32]) -> Result<Tree> {
        let n = self.data.num_data();
        // upstream: BeforeTrain
        self.hist_built.fill(false);
        self.col_sampler.reset_by_tree();
        self.partition.init();
        for b in self.branch_features.iter_mut() {
            b.clear();
        }
        if let Some(c) = self.constraints.as_mut() {
            c.reset();
        }
        if let Some(c) = self.cegb.as_mut() {
            c.before_train();
        }
        let root_cnt = self.partition.leaf_count(0);
        if self.multi_val.is_some() {
            self.gh.resize(n, [0.0; 2]);
            const CHUNK: usize = 1 << 15;
            let out = SharedMut::new(&mut self.gh);
            self.team.for_each(n.div_ceil(CHUNK), n, |c| {
                let s = c * CHUNK;
                let e = (s + CHUNK).min(n);
                // SAFETY: chunks are disjoint.
                let dst = unsafe { out.slice(s, e - s) };
                for (d, (&g, &h)) in dst.iter_mut().zip(grad[s..e].iter().zip(&hess[s..e])) {
                    *d = [g, h];
                }
            });
        }
        for s in self.best_split_per_leaf.iter_mut() {
            s.reset();
        }
        let mut sg = 0.0f64;
        let mut sh = 0.0f64;
        if root_cnt == n {
            for i in 0..n {
                sg += grad[i] as f64;
                sh += hess[i] as f64;
            }
        } else {
            for &i in self.partition.indices_on_leaf(0) {
                sg += grad[i as usize] as f64;
                sh += hess[i as usize] as f64;
            }
        }
        // upstream's root `LeafSplits::Init` keeps the weight of the last
        // smaller leaf (0 before the first split), which a forced root split reads
        self.smaller = LeafSplits {
            leaf: 0,
            num_data: root_cnt as i32,
            sum_gradients: sg,
            sum_hessians: sh,
            weight: self.smaller.weight,
        };
        self.larger = LeafSplits::none();
        if let Some(t) = self.trace.as_mut() {
            t.splits.clear();
        }

        let mut tree = Tree::new(self.num_leaves);
        tree.set_leaf_output(0, root_output(sg, sh, &self.params, root_cnt as i32));

        let mut left_leaf: i32 = 0;
        let mut right_leaf: i32 = -1;
        let init_splits = self.force_splits(&mut tree, grad, hess, &mut left_leaf, &mut right_leaf)?;
        for _ in init_splits..self.num_leaves - 1 {
            if self.before_find_best_split(&tree, left_leaf, right_leaf) {
                self.find_best_splits(&tree, grad, hess, left_leaf, right_leaf, None)?;
            }
            let best_leaf = self.best_leaf();
            if !(self.best_split_per_leaf[best_leaf].gain > 0.0) {
                break;
            }
            (left_leaf, right_leaf) = self.split(&mut tree, best_leaf)?;
        }
        Ok(tree)
    }

    /// upstream `ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_)`.
    fn best_leaf(&self) -> usize {
        let mut best_leaf = 0;
        for i in 1..self.best_split_per_leaf.len() {
            if self.best_split_per_leaf[i].better_than(&self.best_split_per_leaf[best_leaf]) {
                best_leaf = i;
            }
        }
        best_leaf
    }

    fn count(&self, leaf: i32) -> i32 {
        if leaf >= 0 { self.partition.leaf_count(leaf as usize) as i32 } else { 0 }
    }

    fn before_find_best_split(&mut self, tree: &Tree, left: i32, right: i32) -> bool {
        if self.max_depth > 0 && tree.leaf_depth[left as usize] >= self.max_depth {
            self.best_split_per_leaf[left as usize].gain = K_MIN_SCORE;
            if right >= 0 {
                self.best_split_per_leaf[right as usize].gain = K_MIN_SCORE;
            }
            return false;
        }
        let nl = self.count(left);
        let nr = self.count(right);
        let min2 = self.params.min_data_in_leaf * 2;
        if nr < min2 && nl < min2 {
            self.best_split_per_leaf[left as usize].gain = K_MIN_SCORE;
            if right >= 0 {
                self.best_split_per_leaf[right as usize].gain = K_MIN_SCORE;
            }
            return false;
        }
        true
    }

    /// The memory of slot `leaf` (upstream zero-initializes a new buffer).
    fn take_slot(&mut self, leaf: usize) -> LeafHist {
        let nf = self.data.num_features();
        self.hist_pool[leaf].take().unwrap_or_else(|| LeafHist {
            data: vec![0.0; self.slots.buf_len()],
            splittable: vec![true; nf],
            fresh: vec![false; nf],
        })
    }

    /// upstream `FindBestSplits`; `force` adds the forced-split features to
    /// the by-tree features whose histograms are built and searched.
    fn find_best_splits(
        &mut self,
        tree: &Tree,
        grad: &[f32],
        hess: &[f32],
        left: i32,
        right: i32,
        force: Option<&[bool]>,
    ) -> Result<()> {
        let nf = self.data.num_features();
        // upstream HistogramPool::Move: the parent's memory (left slot) goes
        // to the larger leaf, the right slot's old memory to the smaller one.
        let (parent, s_hist) = if right >= 0 {
            (Some(self.take_slot(left as usize)), self.take_slot(right as usize))
        } else {
            (None, self.take_slot(0))
        };
        let bytree = self.col_sampler.is_feature_used_bytree().to_vec();
        let searched: Vec<bool> = match force {
            Some(fs) => bytree.iter().zip(fs).map(|(&b, &f)| b || f).collect(),
            None => bytree.clone(),
        };
        let is_used: Vec<bool> = match &parent {
            Some(p) => p.splittable.iter().zip(&searched).map(|(&s, &b)| s && b).collect(),
            None => searched.clone(),
        };
        // upstream keeps `is_splittable` of features it does not search
        let mut s_spl_init = s_hist.splittable;
        for f in 0..nf {
            if searched[f] && !is_used[f] {
                s_spl_init[f] = false;
            }
        }
        let l_spl_init = parent.as_ref().map(|p| p.splittable.clone());
        let row_wise = self.multi_val.is_some();
        // upstream MultiValBinWrapper::CopyMultiValBinSubset: a row-wise build
        // writes every feature unless the by-tree features hold under 60% of
        // the dense rate, when only those are written.
        let subcol = row_wise && {
            let (mut used, mut total) = (0.0f64, 0.0f64);
            for &f in &self.group_order {
                let dense_rate = 1.0 - self.data.bin_mapper_by_real(self.data.real_feature_index(f)).sparse_rate;
                if bytree[f] {
                    used += dense_rate;
                }
                total += dense_rate;
            }
            used < total * 0.6
        };
        // upstream FindBestSplitsFromHistograms: smaller leaf first, then larger.
        let s_node = self.col_sampler.get_by_node(&self.branch_features[self.smaller.leaf as usize]);
        let l_node = if self.larger.leaf >= 0 {
            self.col_sampler.get_by_node(&self.branch_features[self.larger.leaf as usize])
        } else {
            Vec::new()
        };

        let sm_leaf = self.smaller.leaf as usize;
        let use_indices = right >= 0 || self.partition.leaf_count(sm_leaf) != self.data.num_data();
        let mut s_buf = s_hist.data;
        let indices = self.partition.indices_on_leaf(sm_leaf);
        let idx = if use_indices { Some(indices) } else { None };
        // Row-wise: accumulate row blocks now; blocks are merged per feature
        // below. Col-wise: each feature's histogram is built in its own task.
        let nblock = match &self.multi_val {
            Some(mv) if subcol => {
                self.subcol_buf.resize(self.slots.buf_len(), 0.0);
                let nblock = mv.construct_blocks(&self.team, idx, &self.gh, &mut self.subcol_buf, &mut self.block_bufs);
                for f in (0..nf).filter(|&f| bytree[f]) {
                    let v = self.slots.views[f];
                    let dst = &mut self.subcol_buf[v.start..v.start + v.len];
                    merge_blocks(dst, &self.block_bufs, v.start, nblock);
                    s_buf[v.start..v.start + v.len].copy_from_slice(dst);
                }
                1
            }
            Some(mv) => mv.construct_blocks(&self.team, idx, &self.gh, &mut s_buf, &mut self.block_bufs),
            None => {
                // gather gradients of the smaller leaf in partition order
                if use_indices {
                    self.ordered_g.clear();
                    self.ordered_h.clear();
                    self.ordered_g.extend(indices.iter().map(|&i| grad[i as usize]));
                    self.ordered_h.extend(indices.iter().map(|&i| hess[i as usize]));
                }
                0
            }
        };
        let (og, oh): (&[f32], &[f32]) =
            if use_indices { (&self.ordered_g, &self.ordered_h) } else { (grad, hess) };
        let block_bufs = &self.block_bufs;

        let data: &Dataset = &self.data;
        let metas = &self.metas;
        let params = self.params;
        let smaller = self.smaller;
        let larger = self.larger;
        let smaller_parent_output = parent_output(tree, &smaller, &params);
        let larger_parent_output =
            if larger.leaf >= 0 { parent_output(tree, &larger, &params) } else { 0.0 };
        // upstream ComputeBestSplitForFeature: RecomputeConstraintsIfNeeded
        // only touches the (leaf, feature) entry, so it can run up front.
        if let Some(c) = self.constraints.as_mut() {
            for f in 0..nf {
                if is_used[f] && metas[f].bin_type == BinType::Numerical {
                    c.recompute_if_needed(tree, f, sm_leaf, metas[f].num_bin as u32);
                    if larger.leaf >= 0 {
                        c.recompute_if_needed(tree, f, larger.leaf as usize, metas[f].num_bin as u32);
                    }
                }
            }
        }
        let constraints = self.constraints.as_ref();
        let penalty = |leaf: i32| {
            if leaf >= 0 { monotone_split_gain_penalty(tree.leaf_depth[leaf as usize], self.monotone_penalty) } else { 0.0 }
        };
        let (s_penalty, l_penalty) = (penalty(smaller.leaf), penalty(larger.leaf));

        struct FeatResult {
            smaller_split: SplitInfo,
            smaller_splittable: bool,
            larger_split: SplitInfo,
            larger_splittable: bool,
        }

        let mut l_buf = parent.map(|p| p.data);
        let s_shared = SharedMut::new(&mut s_buf);
        let l_shared = l_buf.as_mut().map(|b| SharedMut::new(b));
        let views = &self.slots.views;
        let rands = self.extra_rands.as_mut().map(|r| SharedMut::new(r));

        let work = if row_wise {
            nblock.max(1) * self.slots.buf_len()
        } else {
            nf * self.smaller.num_data as usize + self.slots.buf_len()
        };
        let results: Vec<FeatResult> = self.team.map(nf, work, |f| {
                let meta = &metas[f];
                let layout = HistLayout {
                    num_bin: meta.num_bin,
                    offset: meta.offset,
                    most_freq_bin: meta.most_freq_bin,
                };
                if !is_used[f] {
                    if row_wise && !subcol {
                        let v = views[f];
                        // SAFETY: feature views are disjoint and each task owns one feature.
                        merge_blocks(unsafe { s_shared.slice(v.start, v.len) }, block_bufs, v.start, nblock);
                    }
                    return FeatResult {
                        smaller_split: SplitInfo::default(),
                        smaller_splittable: false,
                        larger_split: SplitInfo::default(),
                        larger_splittable: false,
                    };
                }
                let v = views[f];
                // SAFETY: feature views are disjoint and each task owns one feature.
                let hist = unsafe { s_shared.slice(v.start, v.len) };
                let lview = l_shared.as_ref().map(|l| unsafe { l.slice(v.start, v.len) });
                if row_wise {
                    merge_blocks(hist, block_bufs, v.start, nblock);
                } else {
                    hist.fill(0.0);
                    histogram::construct(data.feature_bins(f), idx, og, oh, &layout, hist);
                }
                histogram::fix(&layout, smaller.sum_gradients, smaller.sum_hessians, hist);
                let real = data.real_feature_index(f) as i32;
                // SAFETY: task `f` is the only one touching generator `f`.
                let mut rand = rands.as_ref().map(|r| unsafe { r.get(f) });
                let mut s_split = SplitInfo::default();
                let s_ok = find_best_threshold(
                    hist,
                    meta,
                    &params,
                    smaller.sum_gradients,
                    smaller.sum_hessians,
                    smaller.num_data,
                    smaller_parent_output,
                    constraints.map(|c| c.feature_constraint(smaller.leaf as usize, f)),
                    rand.as_deref_mut(),
                    &mut s_split,
                );
                s_split.feature = real;
                let mut l_split = SplitInfo::default();
                let mut l_ok = false;
                if larger.leaf >= 0 {
                    let lh = lview.expect("parent histogram present");
                    histogram::subtract(lh, hist);
                    l_ok = find_best_threshold(
                        lh,
                        meta,
                        &params,
                        larger.sum_gradients,
                        larger.sum_hessians,
                        larger.num_data,
                        larger_parent_output,
                        constraints.map(|c| c.feature_constraint(larger.leaf as usize, f)),
                        rand.as_deref_mut(),
                        &mut l_split,
                    );
                    l_split.feature = real;
                }
                FeatResult {
                    smaller_split: s_split,
                    smaller_splittable: s_ok,
                    larger_split: l_split,
                    larger_splittable: l_ok,
                }
            });
        drop(s_shared);
        drop(l_shared);
        drop(rands);

        // upstream ComputeBestSplitForFeature: the by-node mask is applied
        // after the threshold search so `is_splittable` stays accurate.
        let mut s_best = SplitInfo::default();
        let mut l_best = SplitInfo::default();
        let mut s_spl = s_spl_init;
        let mut l_spl = l_spl_init.unwrap_or_default();
        let view = RowView { num_data: self.view_num_data, local: self.bag_local.as_deref() };
        let s_rows = self.partition.indices_on_leaf(sm_leaf);
        let l_rows = if larger.leaf >= 0 { self.partition.indices_on_leaf(larger.leaf as usize) } else { &[] };
        let mut cegb = self.cegb.as_mut();
        for (f, mut r) in results.into_iter().enumerate() {
            if is_used[f] {
                finish_split(cegb.as_deref_mut(), &view, s_rows, f, &smaller, s_penalty, &mut r.smaller_split);
                if s_node[f] && r.smaller_split.better_than(&s_best) {
                    s_best = r.smaller_split;
                }
                s_spl[f] = r.smaller_splittable;
                if larger.leaf >= 0 {
                    finish_split(cegb.as_deref_mut(), &view, l_rows, f, &larger, l_penalty, &mut r.larger_split);
                    if l_node[f] && r.larger_split.better_than(&l_best) {
                        l_best = r.larger_split;
                    }
                    l_spl[f] = r.larger_splittable;
                }
            }
        }
        self.best_split_per_leaf[sm_leaf] = s_best;
        self.hist_pool[sm_leaf] = Some(LeafHist { data: s_buf, splittable: s_spl, fresh: is_used.clone() });
        self.hist_built[sm_leaf] = true;
        if larger.leaf >= 0 {
            let lg = larger.leaf as usize;
            self.best_split_per_leaf[lg] = l_best;
            self.hist_pool[lg] =
                Some(LeafHist { data: l_buf.expect("parent histogram"), splittable: l_spl, fresh: is_used });
            self.hist_built[lg] = true;
        }
        self.hist_slots = (smaller.leaf, larger.leaf);
        Ok(())
    }

    fn split(&mut self, tree: &mut Tree, best_leaf: usize) -> Result<(i32, i32)> {
        let mut info = self.best_split_per_leaf[best_leaf].clone();
        let inner = self
            .data
            .inner_feature_index(info.feature as usize)
            .expect("split feature is used");
        if let Some(c) = self.cegb.as_mut() {
            let view = RowView { num_data: self.view_num_data, local: self.bag_local.as_deref() };
            c.update_leaf_best_splits(
                best_leaf,
                &info,
                inner,
                &mut self.best_split_per_leaf[..tree.num_leaves],
                self.partition.indices_on_leaf(best_leaf),
                &view,
            );
        }
        let mapper = self.data.feature_bin_mapper(inner);
        let next_leaf = tree.num_leaves;
        let meta = self.metas[inner];
        let num_bin = meta.num_bin.max(1) as u32;
        let numerical = meta.bin_type == BinType::Numerical;
        if let Some(c) = self.constraints.as_mut() {
            c.before_split(tree, best_leaf, next_leaf, info.monotone_type);
        }
        // upstream: SerialTreeLearner::SplitInner (Common::ConstructBitset of
        // the bins, and of their categories via RealThreshold)
        let (lut, cat_bitset_inner, cat_bitset) = if numerical {
            let lut: Vec<bool> = (0..num_bin).map(|b| goes_left(b, &meta, info.threshold, info.default_left)).collect();
            (lut, Vec::new(), Vec::new())
        } else {
            let inner_bits = construct_bitset(info.cat_threshold.iter().map(|&b| b as i32));
            let bits = construct_bitset(info.cat_threshold.iter().map(|&b| mapper.bin_to_value(b) as i32));
            let lut = (0..num_bin).map(|b| goes_left_categorical(b, &meta, &inner_bits)).collect();
            (lut, inner_bits, bits)
        };
        self.partition.split(&self.team, best_leaf, self.data.feature_bins(inner), &lut, next_leaf);
        info.left_count = self.partition.leaf_count(best_leaf) as i32;
        info.right_count = self.partition.leaf_count(next_leaf) as i32;
        let gain = (info.gain + self.params.min_gain_to_split) as f32;
        let args = SplitArgs {
            leaf: best_leaf,
            feature_inner: inner as i32,
            feature_real: info.feature,
            threshold_bin: info.threshold,
            threshold: if numerical { mapper.bin_to_value(info.threshold) } else { 0.0 },
            left_value: info.left_output,
            right_value: info.right_output,
            left_count: info.left_count,
            right_count: info.right_count,
            left_weight: info.left_sum_hessian,
            right_weight: info.right_sum_hessian,
            gain,
            missing_type: mapper.missing_type,
            default_left: info.default_left,
        };
        let right = if numerical {
            tree.split(&args)
        } else {
            tree.split_categorical(&args, &cat_bitset_inner, &cat_bitset)
        };
        if self.track_branch_features {
            // upstream: Tree::Split with track_branch_features_
            let mut b = self.branch_features[best_leaf].clone();
            b.push(info.feature);
            self.branch_features[right] = b;
            self.branch_features[best_leaf].push(info.feature);
        }
        if let Some(t) = self.trace.as_mut() {
            t.splits.push((best_leaf, info.feature, info.threshold, info.gain));
        }
        let left_s = LeafSplits {
            leaf: best_leaf as i32,
            num_data: info.left_count,
            sum_gradients: info.left_sum_gradient,
            sum_hessians: info.left_sum_hessian,
            weight: info.left_output,
        };
        let right_s = LeafSplits {
            leaf: right as i32,
            num_data: info.right_count,
            sum_gradients: info.right_sum_gradient,
            sum_hessians: info.right_sum_hessian,
            weight: info.right_output,
        };
        // only a forced split can leave a child empty
        let check_gt = |count: i32, side: &str| {
            if count > 0 {
                Ok(())
            } else {
                Err(LgbmError::InvalidParameter(format!("Check failed: (best_split_info.{side}_count) > (0)")))
            }
        };
        if info.left_count < info.right_count {
            check_gt(info.left_count, "left")?;
            self.smaller = left_s;
            self.larger = right_s;
        } else {
            check_gt(info.right_count, "right")?;
            self.smaller = right_s;
            self.larger = left_s;
        }
        if let Some(c) = self.constraints.as_mut() {
            let leaves = c.update(
                tree,
                numerical,
                best_leaf,
                right,
                info.monotone_type,
                inner as i32,
                &info,
                &self.best_split_per_leaf,
            );
            for leaf in leaves {
                self.recompute_best_split_for_leaf(tree, leaf);
            }
        }
        Ok((best_leaf as i32, right as i32))
    }

    /// upstream: `SerialTreeLearner::RecomputeBestSplitForLeaf` (after a
    /// monotone split tightened the constraints of `leaf`).
    fn recompute_best_split_for_leaf(&mut self, tree: &Tree, leaf: usize) {
        if !self.hist_built[leaf] {
            return;
        }
        let Some(mut lh) = self.hist_pool[leaf].take() else { return };
        let split = &self.best_split_per_leaf[leaf];
        let sum_gradients = split.left_sum_gradient + split.right_sum_gradient;
        let sum_hessians = split.left_sum_hessian + split.right_sum_hessian;
        let num_data = split.left_count + split.right_count;
        let params = self.params;
        let parent_output =
            if params.path_smooth > K_EPSILON { root_output(sum_gradients, sum_hessians, &params, num_data) } else { 0.0 };
        let node_used = self.col_sampler.get_by_node(&self.branch_features[leaf]);
        let penalty = monotone_split_gain_penalty(tree.leaf_depth[leaf], self.monotone_penalty);
        let mut best = SplitInfo::default();
        for f in 0..self.data.num_features() {
            if !self.col_sampler.is_feature_used_bytree()[f] || !lh.splittable[f] {
                continue;
            }
            let meta = self.metas[f];
            let constraints = self.constraints.as_mut().expect("monotone constraints");
            if meta.bin_type == BinType::Numerical {
                constraints.recompute_if_needed(tree, f, leaf, meta.num_bin as u32);
            }
            let v = self.slots.views[f];
            let mut new_split = SplitInfo::default();
            lh.splittable[f] = find_best_threshold(
                &lh.data[v.start..v.start + v.len],
                &meta,
                &params,
                sum_gradients,
                sum_hessians,
                num_data,
                parent_output,
                Some(constraints.feature_constraint(leaf, f)),
                self.extra_rands.as_mut().map(|r| &mut r[f]),
                &mut new_split,
            );
            new_split.feature = self.data.real_feature_index(f) as i32;
            let ls = LeafSplits { leaf: leaf as i32, num_data, ..LeafSplits::none() };
            let view = RowView { num_data: self.view_num_data, local: self.bag_local.as_deref() };
            let rows = self.partition.indices_on_leaf(leaf);
            finish_split(self.cegb.as_mut(), &view, rows, f, &ls, penalty, &mut new_split);
            if new_split.better_than(&best) && node_used[f] {
                best = new_split;
            }
        }
        self.hist_pool[leaf] = Some(lh);
        self.best_split_per_leaf[leaf] = best;
    }
}

/// The end of upstream `ComputeBestSplitForFeature`: the CEGB penalty (which
/// records the split as found), then the monotone depth penalty.
fn finish_split(
    cegb: Option<&mut Cegb>,
    view: &RowView<'_>,
    rows: &[u32],
    inner: usize,
    ls: &LeafSplits,
    monotone_penalty: f64,
    split: &mut SplitInfo,
) {
    if let Some(c) = cegb {
        debug_assert_eq!(rows.len(), ls.num_data as usize);
        let delta = c.delta_gain(inner, split.feature as usize, ls.leaf as usize, split, rows, view);
        split.gain -= delta;
    }
    if split.monotone_type != 0 {
        split.gain *= monotone_penalty;
    }
}

/// upstream: `SerialTreeLearner::GetParentOutput`.
fn parent_output(tree: &Tree, ls: &LeafSplits, p: &SplitParams) -> f64 {
    if tree.num_leaves == 1 {
        root_output(ls.sum_gradients, ls.sum_hessians, p, ls.num_data)
    } else {
        ls.weight
    }
}

/// Route one bin value the way upstream `DenseBin::Split` does.
#[inline]
pub(crate) fn goes_left(b: u32, meta: &FeatureMeta, threshold: u32, default_left: bool) -> bool {
    let is_missing = (meta.missing_type == MissingType::Zero && b == meta.default_bin)
        || (meta.missing_type == MissingType::NaN && b == (meta.num_bin - 1) as u32);
    if is_missing { default_left } else { b <= threshold }
}

/// Route one bin the way upstream `DenseBin::SplitCategorical` does for a
/// single-feature group: the most frequent bin is stored as 0 and follows
/// the bitset test of `most_freq_bin` (only when it is not bin 0).
#[inline]
pub(crate) fn goes_left_categorical(b: u32, meta: &FeatureMeta, bitset: &[u32]) -> bool {
    if b == meta.most_freq_bin {
        meta.most_freq_bin > 0 && find_in_bitset(bitset, meta.most_freq_bin as i32)
    } else {
        find_in_bitset(bitset, b as i32)
    }
}

#[cfg(test)]
mod tests;
