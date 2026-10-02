//! Monotone constraints on leaf outputs.
//!
//! upstream: src/treelearner/monotone_constraints.hpp (`BasicLeafConstraints`,
//! `IntermediateLeafConstraints`, `AdvancedLeafConstraints` and the feature
//! constraints read by the threshold search). Feature indices are inner
//! indices unless noted; `monotone` is indexed by real feature.

use crate::consts::{K_EPSILON, K_MIN_SCORE};
use crate::learner::split::SplitInfo;
use crate::tree::Tree;

/// `std::max(a, b)`.
#[inline]
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b { b } else { a }
}

/// `std::min(a, b)`.
#[inline]
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BasicConstraint {
    pub min: f64,
    pub max: f64,
}

impl Default for BasicConstraint {
    fn default() -> Self {
        Self { min: -f64::MAX, max: f64::MAX }
    }
}

impl BasicConstraint {
    /// upstream `CalculateSplittedLeafOutput<USE_MC = true, ...>` clamping.
    #[inline]
    pub fn clamp(&self, v: f64) -> f64 {
        if v < self.min {
            self.min
        } else if v > self.max {
            self.max
        } else {
            v
        }
    }
}

/// Piecewise-constant bound over bins: `constraints[i]` holds on
/// `[thresholds[i], thresholds[i + 1])`.
///
/// upstream: `FeatureMinOrMaxConstraints`.
#[derive(Debug, Clone, Default)]
pub struct MinOrMax {
    pub constraints: Vec<f64>,
    pub thresholds: Vec<u32>,
}

impl MinOrMax {
    fn new(extremum: f64) -> Self {
        Self { constraints: vec![extremum], thresholds: vec![0] }
    }

    fn reset(&mut self, extremum: f64) {
        self.constraints.clear();
        self.constraints.push(extremum);
        self.thresholds.clear();
        self.thresholds.push(0);
    }

    fn update_min(&mut self, min: f64) {
        for c in &mut self.constraints {
            if min > *c {
                *c = min;
            }
        }
    }

    fn update_max(&mut self, max: f64) {
        for c in &mut self.constraints {
            if max < *c {
                *c = max;
            }
        }
    }
}

/// upstream: `AdvancedFeatureConstraints`.
#[derive(Debug, Clone)]
pub struct AdvancedFeature {
    min: MinOrMax,
    max: MinOrMax,
    min_to_recompute: bool,
    max_to_recompute: bool,
}

impl AdvancedFeature {
    fn new() -> Self {
        Self { min: MinOrMax::new(-f64::MAX), max: MinOrMax::new(f64::MAX), min_to_recompute: false, max_to_recompute: false }
    }

    /// Like upstream, the recompute flags survive a reset.
    fn reset(&mut self) {
        self.min.reset(-f64::MAX);
        self.max.reset(f64::MAX);
    }

    fn update_min(&mut self, v: f64, trigger_recompute: bool) {
        if trigger_recompute {
            self.min_to_recompute = true;
        }
        self.min.update_min(v);
    }

    fn update_max(&mut self, v: f64, trigger_recompute: bool) {
        if trigger_recompute {
            self.max_to_recompute = true;
        }
        self.max.update_max(v);
    }
}

/// The constraint of one (leaf, feature) pair as seen by the threshold search
/// (upstream `FeatureConstraint`).
#[derive(Debug, Clone, Copy)]
pub enum FeatureConstraint<'a> {
    Basic(BasicConstraint),
    Advanced(&'a AdvancedFeature),
}

impl FeatureConstraint<'_> {
    pub fn different_depending_on_threshold(&self) -> bool {
        match self {
            FeatureConstraint::Basic(_) => false,
            FeatureConstraint::Advanced(a) => a.min.thresholds.len() > 1 || a.max.thresholds.len() > 1,
        }
    }

    /// upstream `InitCumulativeConstraints(REVERSE)`.
    pub fn init_cumulative(&self, reverse: bool) -> ScanConstraint {
        match self {
            FeatureConstraint::Basic(b) => ScanConstraint::Basic(*b),
            FeatureConstraint::Advanced(a) => ScanConstraint::Advanced(Box::new(Cumulative::new(&a.min, &a.max, reverse))),
        }
    }
}

/// State of a feature constraint during one threshold scan.
#[derive(Debug, Clone)]
pub enum ScanConstraint {
    Basic(BasicConstraint),
    Advanced(Box<Cumulative>),
}

impl ScanConstraint {
    #[inline]
    pub fn update(&mut self, threshold: i32) {
        if let ScanConstraint::Advanced(c) = self {
            c.update(threshold);
        }
    }

    #[inline]
    pub fn left(&self) -> BasicConstraint {
        match self {
            ScanConstraint::Basic(b) => *b,
            ScanConstraint::Advanced(c) => {
                BasicConstraint { min: c.min_l2r[c.i_min_l2r], max: c.max_l2r[c.i_max_l2r] }
            }
        }
    }

    #[inline]
    pub fn right(&self) -> BasicConstraint {
        match self {
            ScanConstraint::Basic(b) => *b,
            ScanConstraint::Advanced(c) => {
                BasicConstraint { min: c.min_r2l[c.i_min_r2l], max: c.max_r2l[c.i_max_r2l] }
            }
        }
    }
}

/// upstream: `CumulativeFeatureConstraint`.
#[derive(Debug, Clone)]
pub struct Cumulative {
    thr_min: Vec<u32>,
    thr_max: Vec<u32>,
    min_l2r: Vec<f64>,
    min_r2l: Vec<f64>,
    max_l2r: Vec<f64>,
    max_r2l: Vec<f64>,
    i_min_l2r: usize,
    i_min_r2l: usize,
    i_max_l2r: usize,
    i_max_r2l: usize,
}

fn cumulative_extremum(f: fn(f64, f64) -> f64, left_to_right: bool, v: &mut [f64]) {
    let n = v.len();
    if n <= 1 {
        return;
    }
    if left_to_right {
        for i in 0..n - 1 {
            v[i + 1] = f(v[i + 1], v[i]);
        }
    } else {
        for i in (1..n).rev() {
            v[i - 1] = f(v[i - 1], v[i]);
        }
    }
}

impl Cumulative {
    fn new(min: &MinOrMax, max: &MinOrMax, reverse: bool) -> Self {
        let mut c = Self {
            thr_min: min.thresholds.clone(),
            thr_max: max.thresholds.clone(),
            min_l2r: min.constraints.clone(),
            min_r2l: min.constraints.clone(),
            max_l2r: max.constraints.clone(),
            max_r2l: max.constraints.clone(),
            i_min_l2r: 0,
            i_min_r2l: 0,
            i_max_l2r: 0,
            i_max_r2l: 0,
        };
        cumulative_extremum(cpp_max, true, &mut c.min_l2r);
        cumulative_extremum(cpp_max, false, &mut c.min_r2l);
        cumulative_extremum(cpp_min, true, &mut c.max_l2r);
        cumulative_extremum(cpp_min, false, &mut c.max_r2l);
        if reverse {
            c.i_min_l2r = c.thr_min.len() - 1;
            c.i_min_r2l = c.thr_min.len() - 1;
            c.i_max_l2r = c.thr_max.len() - 1;
            c.i_max_r2l = c.thr_max.len() - 1;
        }
        c
    }

    fn update(&mut self, threshold: i32) {
        while self.thr_min[self.i_min_l2r] as i32 > threshold - 1 {
            self.i_min_l2r -= 1;
        }
        while self.thr_min[self.i_min_r2l] as i32 > threshold {
            self.i_min_r2l -= 1;
        }
        while self.thr_max[self.i_max_l2r] as i32 > threshold - 1 {
            self.i_max_l2r -= 1;
        }
        while self.thr_max[self.i_max_r2l] as i32 > threshold {
            self.i_max_r2l -= 1;
        }
    }
}

/// upstream `monotone_constraints_method`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Basic,
    Intermediate,
    Advanced,
}

impl Method {
    /// upstream `LeafConstraintsBase::Create`: anything else is "basic".
    pub fn parse(s: &str) -> Self {
        match s {
            "intermediate" => Method::Intermediate,
            "advanced" => Method::Advanced,
            _ => Method::Basic,
        }
    }
}

/// upstream `BasicConstraintEntry` / `AdvancedConstraintEntry`.
#[derive(Debug, Clone)]
enum Entry {
    Basic(BasicConstraint),
    Advanced(Vec<AdvancedFeature>),
}

impl Entry {
    fn reset(&mut self) {
        match self {
            Entry::Basic(b) => *b = BasicConstraint::default(),
            Entry::Advanced(v) => v.iter_mut().for_each(AdvancedFeature::reset),
        }
    }

    fn update_min(&mut self, v: f64) {
        match self {
            Entry::Basic(b) => b.min = cpp_max(v, b.min),
            Entry::Advanced(fs) => fs.iter_mut().for_each(|f| f.update_min(v, false)),
        }
    }

    fn update_max(&mut self, v: f64) {
        match self {
            Entry::Basic(b) => b.max = cpp_min(v, b.max),
            Entry::Advanced(fs) => fs.iter_mut().for_each(|f| f.update_max(v, false)),
        }
    }

    /// Advanced entries always report a change: an unconstrained feature
    /// still has to be recomputed from scratch.
    fn update_min_changed(&mut self, v: f64) -> bool {
        match self {
            Entry::Basic(b) => {
                if v > b.min {
                    b.min = v;
                    true
                } else {
                    false
                }
            }
            Entry::Advanced(fs) => {
                fs.iter_mut().for_each(|f| f.update_min(v, true));
                true
            }
        }
    }

    fn update_max_changed(&mut self, v: f64) -> bool {
        match self {
            Entry::Basic(b) => {
                if v < b.max {
                    b.max = v;
                    true
                } else {
                    false
                }
            }
            Entry::Advanced(fs) => {
                fs.iter_mut().for_each(|f| f.update_max(v, true));
                true
            }
        }
    }
}

/// Splits on the path from the original leaf to the root, recorded while
/// going up (upstream's three parallel vectors).
#[derive(Default)]
struct Path {
    features: Vec<i32>,
    thresholds: Vec<u32>,
    was_right: Vec<bool>,
}

impl Path {
    fn push(&mut self, feature: i32, threshold: u32, was_right: bool) {
        self.was_right.push(was_right);
        self.thresholds.push(threshold);
        self.features.push(feature);
    }

    /// upstream `OppositeChildShouldBeUpdated`.
    fn opposite_child_should_be_updated(&self, numerical: bool, inner_feature: i32, is_in_right: bool) -> bool {
        if !numerical {
            return false;
        }
        !self.features.iter().zip(&self.was_right).any(|(&f, &r)| f == inner_feature && r == is_in_right)
    }

    /// upstream `ShouldKeepGoingLeftRight`.
    fn keep_going_left_right(&self, tree: &Tree, node: usize) -> (bool, bool) {
        let inner = tree.split_feature_inner[node];
        let threshold = tree.threshold_in_bin[node];
        let mut right = true;
        let mut left = true;
        if !tree.is_categorical(node) {
            for i in 0..self.features.len() {
                if self.features[i] == inner {
                    if threshold >= self.thresholds[i] && !self.was_right[i] {
                        right = false;
                        if !left {
                            break;
                        }
                    }
                    if threshold <= self.thresholds[i] && self.was_right[i] {
                        left = false;
                        if !right {
                            break;
                        }
                    }
                }
            }
        }
        (left, right)
    }
}

/// upstream `LeafConstraintsBase` and its three implementations.
#[derive(Debug, Clone)]
pub struct LeafConstraints {
    method: Method,
    /// By real feature index.
    monotone: Vec<i8>,
    entries: Vec<Entry>,
    leaf_is_in_monotone_subtree: Vec<bool>,
    node_parent: Vec<i32>,
    leaves_to_update: Vec<usize>,
}

impl LeafConstraints {
    pub fn new(method: Method, monotone: Vec<i8>, num_leaves: usize, num_features: usize) -> Self {
        let entry = if method == Method::Advanced {
            Entry::Advanced(vec![AdvancedFeature::new(); num_features])
        } else {
            Entry::Basic(BasicConstraint::default())
        };
        Self {
            method,
            monotone,
            entries: vec![entry; num_leaves],
            leaf_is_in_monotone_subtree: vec![false; num_leaves],
            node_parent: vec![-1; num_leaves.saturating_sub(1)],
            leaves_to_update: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.entries.iter_mut().for_each(Entry::reset);
        if self.method != Method::Basic {
            self.leaf_is_in_monotone_subtree.iter_mut().for_each(|v| *v = false);
            self.node_parent.iter_mut().for_each(|v| *v = -1);
            self.leaves_to_update.clear();
        }
    }

    pub fn feature_constraint(&self, leaf: usize, feature: usize) -> FeatureConstraint<'_> {
        match &self.entries[leaf] {
            Entry::Basic(b) => FeatureConstraint::Basic(*b),
            Entry::Advanced(fs) => FeatureConstraint::Advanced(&fs[feature]),
        }
    }

    /// Called before `tree` splits `leaf` into `leaf` and `new_leaf`.
    pub fn before_split(&mut self, tree: &Tree, leaf: usize, new_leaf: usize, monotone_type: i8) {
        if self.method == Method::Basic {
            return;
        }
        if monotone_type != 0 || self.leaf_is_in_monotone_subtree[leaf] {
            self.leaf_is_in_monotone_subtree[leaf] = true;
            self.leaf_is_in_monotone_subtree[new_leaf] = true;
        }
        self.node_parent[new_leaf - 1] = tree.leaf_parent[leaf];
    }

    /// Called after the split; returns the leaves whose best split must be
    /// recomputed. `split_feature` is the inner feature of the split.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        tree: &Tree,
        is_numerical: bool,
        leaf: usize,
        new_leaf: usize,
        monotone_type: i8,
        split_feature: i32,
        split: &SplitInfo,
        best_split_per_leaf: &[SplitInfo],
    ) -> Vec<usize> {
        if self.method == Method::Basic {
            self.entries[new_leaf] = self.entries[leaf].clone();
            if is_numerical {
                let mid = (split.left_output + split.right_output) / 2.0;
                if monotone_type < 0 {
                    self.entries[leaf].update_min(mid);
                    self.entries[new_leaf].update_max(mid);
                } else if monotone_type > 0 {
                    self.entries[leaf].update_max(mid);
                    self.entries[new_leaf].update_min(mid);
                }
            }
            return Vec::new();
        }
        self.leaves_to_update.clear();
        if self.leaf_is_in_monotone_subtree[leaf] {
            self.entries[new_leaf] = self.entries[leaf].clone();
            if is_numerical {
                if monotone_type < 0 {
                    self.entries[leaf].update_min(split.right_output);
                    self.entries[new_leaf].update_max(split.left_output);
                } else if monotone_type > 0 {
                    self.entries[leaf].update_max(split.right_output);
                    self.entries[new_leaf].update_min(split.left_output);
                }
            }
            let mut path = Path::default();
            self.go_up_to_find_leaves_to_update(
                tree,
                tree.leaf_parent[new_leaf],
                &mut path,
                split_feature,
                split,
                best_split_per_leaf,
            );
        }
        self.leaves_to_update.clone()
    }

    fn go_up_to_find_leaves_to_update(
        &mut self,
        tree: &Tree,
        node: i32,
        path: &mut Path,
        split_feature: i32,
        split: &SplitInfo,
        best: &[SplitInfo],
    ) {
        let parent = self.node_parent[node as usize];
        if parent == -1 {
            return;
        }
        let p = parent as usize;
        let inner = tree.split_feature_inner[p];
        let mono = self.monotone[tree.split_feature[p] as usize];
        let is_in_right = tree.right_child[p] == node;
        let numerical = !tree.is_categorical(p);
        if path.opposite_child_should_be_updated(numerical, inner, is_in_right) {
            if mono != 0 {
                let left_is_curr = tree.left_child[p] == node;
                let opposite = if left_is_curr { tree.right_child[p] } else { tree.left_child[p] };
                let update_max = if mono < 0 { left_is_curr } else { !left_is_curr };
                self.go_down_to_find_leaves_to_update(
                    tree, opposite, path, update_max, split_feature, split, true, true, split.threshold, best,
                );
            }
            path.push(inner, tree.threshold_in_bin[p], is_in_right);
        }
        self.go_up_to_find_leaves_to_update(tree, parent, path, split_feature, split, best);
    }

    #[allow(clippy::too_many_arguments)]
    fn go_down_to_find_leaves_to_update(
        &mut self,
        tree: &Tree,
        node: i32,
        path: &Path,
        update_max: bool,
        split_feature: i32,
        split: &SplitInfo,
        use_left_leaf: bool,
        use_right_leaf: bool,
        split_threshold: u32,
        best: &[SplitInfo],
    ) {
        if node < 0 {
            let leaf = !node as usize;
            if best[leaf].gain == K_MIN_SCORE {
                return;
            }
            let (l, r) = (split.left_output, split.right_output);
            let (mn, mx) = if use_right_leaf && use_left_leaf {
                // std::minmax(right, left)
                if l < r { (l, r) } else { (r, l) }
            } else if use_right_leaf {
                (r, r)
            } else {
                (l, l)
            };
            let changed = if !update_max {
                self.entries[leaf].update_min_changed(mx)
            } else {
                self.entries[leaf].update_max_changed(mn)
            };
            if changed {
                self.leaves_to_update.push(leaf);
            }
            return;
        }
        let n = node as usize;
        let (keep_left, keep_right) = path.keep_going_left_right(tree, n);
        let inner = tree.split_feature_inner[n];
        let threshold = tree.threshold_in_bin[n];
        let mut use_left_for_update_right = true;
        let mut use_right_for_update_left = true;
        if !tree.is_categorical(n) && inner == split_feature {
            if threshold >= split_threshold {
                use_left_for_update_right = false;
            }
            if threshold <= split_threshold {
                use_right_for_update_left = false;
            }
        }
        if keep_left {
            self.go_down_to_find_leaves_to_update(
                tree,
                tree.left_child[n],
                path,
                update_max,
                split_feature,
                split,
                use_left_leaf,
                use_right_for_update_left && use_right_leaf,
                split_threshold,
                best,
            );
        }
        if keep_right {
            self.go_down_to_find_leaves_to_update(
                tree,
                tree.right_child[n],
                path,
                update_max,
                split_feature,
                split,
                use_left_for_update_right && use_left_leaf,
                use_right_leaf,
                split_threshold,
                best,
            );
        }
    }

    /// upstream `AdvancedConstraintEntry::RecomputeConstraintsIfNeeded`
    /// (a no-op for the other methods). If both bounds are flagged only the
    /// minimum is recomputed, and both flags are cleared.
    pub fn recompute_if_needed(&mut self, tree: &Tree, feature: usize, leaf: usize, num_bin: u32) {
        let Entry::Advanced(fs) = &mut self.entries[leaf] else { return };
        let fc = &mut fs[feature];
        if !(fc.min_to_recompute || fc.max_to_recompute) {
            return;
        }
        let is_min = fc.min_to_recompute;
        let mut target = std::mem::take(if is_min { &mut fc.min } else { &mut fc.max });
        target.reset(if is_min { -f64::MAX } else { f64::MAX });
        let mut path = Path::default();
        self.go_up_to_find_constraining_leaves(
            tree, feature as i32, !(leaf as i32), &mut path, &mut target, is_min, 0, num_bin, num_bin,
        );
        let Entry::Advanced(fs) = &mut self.entries[leaf] else { unreachable!() };
        let fc = &mut fs[feature];
        if is_min {
            fc.min = target;
        } else {
            fc.max = target;
        }
        fc.min_to_recompute = false;
        fc.max_to_recompute = false;
    }

    #[allow(clippy::too_many_arguments)]
    fn go_up_to_find_constraining_leaves(
        &self,
        tree: &Tree,
        feature_for_constraint: i32,
        node: i32,
        path: &mut Path,
        fc: &mut MinOrMax,
        min_to_update: bool,
        mut it_start: u32,
        mut it_end: u32,
        last_threshold: u32,
    ) {
        let parent = if node < 0 { tree.leaf_parent[!node as usize] } else { self.node_parent[node as usize] };
        if parent == -1 {
            return;
        }
        let p = parent as usize;
        let inner = tree.split_feature_inner[p];
        let mono = self.monotone[tree.split_feature[p] as usize];
        let is_in_right = tree.right_child[p] == node;
        let numerical = !tree.is_categorical(p);
        let threshold = tree.threshold_in_bin[p];
        if feature_for_constraint == inner && numerical {
            if is_in_right {
                it_start = threshold.max(it_start);
            } else {
                it_end = (threshold + 1).min(it_end);
            }
        }
        if path.opposite_child_should_be_updated(numerical, inner, is_in_right) {
            if mono != 0 {
                let left_is_curr = tree.left_child[p] == node;
                let update_min_in_curr = if mono < 0 { left_is_curr } else { !left_is_curr };
                if update_min_in_curr == min_to_update {
                    let opposite = if left_is_curr { tree.right_child[p] } else { tree.left_child[p] };
                    self.go_down_to_find_constraining_leaves(
                        tree,
                        feature_for_constraint,
                        inner,
                        opposite,
                        min_to_update,
                        it_start,
                        it_end,
                        path,
                        fc,
                        last_threshold,
                    );
                }
            }
            path.push(inner, threshold, is_in_right);
        }
        if parent != 0 {
            self.go_up_to_find_constraining_leaves(
                tree,
                feature_for_constraint,
                parent,
                path,
                fc,
                min_to_update,
                it_start,
                it_end,
                last_threshold,
            );
        }
    }

    /// upstream `LeftRightContainsRelevantInformation`.
    fn left_right_relevant(&self, min_to_update: bool, feature: usize, split_is_inner: bool) -> (bool, bool) {
        if split_is_inner {
            return (true, true);
        }
        let mono = self.monotone[feature];
        if mono == 0 {
            return (true, true);
        }
        if (mono == -1 && min_to_update) || (mono == 1 && !min_to_update) { (true, false) } else { (false, true) }
    }

    #[allow(clippy::too_many_arguments)]
    fn go_down_to_find_constraining_leaves(
        &self,
        tree: &Tree,
        feature_for_constraint: i32,
        root_monotone_feature: i32,
        node: i32,
        min_to_update: bool,
        it_start: u32,
        it_end: u32,
        path: &Path,
        fc: &mut MinOrMax,
        last_threshold: u32,
    ) {
        if node < 0 {
            let extremum = tree.leaf_value[!node as usize];
            update_constraints(fc, extremum, it_start, it_end, min_to_update, last_threshold);
            return;
        }
        let n = node as usize;
        let (keep_left, keep_right) = path.keep_going_left_right(tree, n);
        let inner = tree.split_feature_inner[n];
        let threshold = tree.threshold_in_bin[n];
        let split_is_inner = inner == feature_for_constraint;
        let split_is_monotone_feature = root_monotone_feature == feature_for_constraint;
        let (rel_left, rel_right) = self.left_right_relevant(
            min_to_update,
            tree.split_feature[n] as usize,
            split_is_inner && !split_is_monotone_feature,
        );
        if keep_left && (rel_left || !keep_right) {
            let new_end = if split_is_inner { (threshold + 1).min(it_end) } else { it_end };
            self.go_down_to_find_constraining_leaves(
                tree,
                feature_for_constraint,
                root_monotone_feature,
                tree.left_child[n],
                min_to_update,
                it_start,
                new_end,
                path,
                fc,
                last_threshold,
            );
        }
        if keep_right && (rel_right || !keep_left) {
            let new_start = if split_is_inner { (threshold + 1).max(it_start) } else { it_start };
            self.go_down_to_find_constraining_leaves(
                tree,
                feature_for_constraint,
                root_monotone_feature,
                tree.right_child[n],
                min_to_update,
                new_start,
                it_end,
                path,
                fc,
                last_threshold,
            );
        }
    }
}

/// upstream `AdvancedLeafConstraints::UpdateConstraints`: raise (`use_max`)
/// or lower the bound to `extremum` on bins `[it_start, it_end)`.
fn update_constraints(fc: &mut MinOrMax, extremum: f64, it_start: u32, it_end: u32, use_max: bool, last_threshold: u32) {
    let pick = |a: f64, b: f64| if use_max { cpp_max(a, b) } else { cpp_min(a, b) };
    let beats = |a: f64, b: f64| if use_max { a > b } else { a < b };
    let mut start_done = false;
    let mut end_done = false;
    let mut previous = if use_max { -f64::MAX } else { f64::MAX };
    let mut i = 0usize;
    while i < fc.thresholds.len() {
        let current = fc.constraints[i];
        if fc.thresholds[i] == it_start {
            fc.constraints[i] = pick(extremum, fc.constraints[i]);
            start_done = true;
        }
        if fc.thresholds[i] > it_start {
            if fc.thresholds[i] < it_end {
                fc.constraints[i] = pick(extremum, fc.constraints[i]);
            }
            if !start_done {
                start_done = true;
                if beats(extremum, previous) {
                    fc.constraints.insert(i, extremum);
                    fc.thresholds.insert(i, it_start);
                    i += 1;
                }
            }
        }
        if fc.thresholds[i] == it_end {
            end_done = true;
            break;
        }
        if fc.thresholds[i] > it_end {
            if i != 0 && previous != fc.constraints[i - 1] {
                fc.constraints.insert(i, previous);
                fc.thresholds.insert(i, it_end);
            }
            end_done = true;
            break;
        }
        if i != 0 && fc.constraints[i] == fc.constraints[i - 1] {
            fc.constraints.remove(i);
            fc.thresholds.remove(i);
            i -= 1;
        }
        previous = current;
        i += 1;
    }
    if !start_done {
        if beats(extremum, *fc.constraints.last().expect("non-empty")) {
            fc.constraints.push(extremum);
            fc.thresholds.push(it_start);
        } else {
            end_done = true;
        }
    }
    if !end_done && it_end != last_threshold && previous != *fc.constraints.last().expect("non-empty") {
        fc.constraints.push(previous);
        fc.thresholds.push(it_end);
    }
}

/// upstream `LeafConstraintsBase::ComputeMonotoneSplitGainPenalty`.
pub fn monotone_split_gain_penalty(depth: i32, penalization: f64) -> f64 {
    let d = depth as f64;
    if penalization >= d + 1.0 {
        return K_EPSILON;
    }
    if penalization <= 1.0 {
        return 1.0 - penalization / 2f64.powf(d) + K_EPSILON;
    }
    1.0 - 2f64.powf(penalization - 1.0 - d) + K_EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_constraints_splits_ranges() {
        // upstream comment: adding cstr2 on [1, 2) to cstr1 on [0, inf)
        // gives thresholds [0, 1, 2] and constraints [cstr1, cstr2, cstr1].
        let mut fc = MinOrMax::new(-f64::MAX);
        update_constraints(&mut fc, 5.0, 1, 2, true, 10);
        assert_eq!(fc.thresholds, vec![0, 1, 2]);
        assert_eq!(fc.constraints, vec![-f64::MAX, 5.0, -f64::MAX]);
        // a weaker bound on an overlapping range changes nothing
        update_constraints(&mut fc, 4.0, 1, 2, true, 10);
        assert_eq!(fc.constraints, vec![-f64::MAX, 5.0, -f64::MAX]);
        // up to the last bin: no trailing reset
        let mut fc = MinOrMax::new(f64::MAX);
        update_constraints(&mut fc, -1.0, 3, 10, false, 10);
        assert_eq!(fc.thresholds, vec![0, 3]);
        assert_eq!(fc.constraints, vec![f64::MAX, -1.0]);
    }

    #[test]
    fn cumulative_constraint_tracks_threshold() {
        let min = MinOrMax { constraints: vec![-f64::MAX, 2.0, 1.0], thresholds: vec![0, 3, 6] };
        let max = MinOrMax::new(f64::MAX);
        let mut s = FeatureConstraint::Advanced(&AdvancedFeature {
            min,
            max,
            min_to_recompute: false,
            max_to_recompute: false,
        })
        .init_cumulative(true);
        assert_eq!(s.left().min, 2.0);
        assert_eq!(s.right().min, 1.0);
        s.update(5);
        assert_eq!(s.left().min, 2.0);
        assert_eq!(s.right().min, 2.0);
        s.update(2);
        assert_eq!(s.left().min, -f64::MAX);
        assert_eq!(s.right().min, 2.0);
    }

    #[test]
    fn penalty_matches_upstream_formula() {
        assert_eq!(monotone_split_gain_penalty(0, 1.0), K_EPSILON);
        assert_eq!(monotone_split_gain_penalty(2, 0.0), 1.0 + K_EPSILON);
        assert_eq!(monotone_split_gain_penalty(1, 0.5), 1.0 - 0.25 + K_EPSILON);
        assert_eq!(monotone_split_gain_penalty(3, 2.0), 1.0 - 0.25 + K_EPSILON);
    }
}
