//! Decision tree model.
//!
//! upstream: include/LightGBM/tree.h, src/io/tree.cpp (numerical and
//! categorical splits; no linear trees).

use std::collections::HashMap;

use crate::binning::MissingType;
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::{Dataset, avoid_inf_f32, avoid_inf_f64};
use crate::error::{LgbmError, Result};
use crate::fmt::{fmt_g6, fmt_g17};

const K_CATEGORICAL_MASK: i8 = 1;
const K_DEFAULT_LEFT_MASK: i8 = 2;

#[inline]
pub fn is_zero(v: f64) -> bool {
    v >= -K_ZERO_THRESHOLD && v <= K_ZERO_THRESHOLD
}

#[inline]
pub fn maybe_round_to_zero(v: f64) -> f64 {
    if is_zero(v) { 0.0 } else { v }
}

/// upstream: utils/common.h `ConstructBitset`.
pub fn construct_bitset(vals: impl Iterator<Item = i32>) -> Vec<u32> {
    let mut ret: Vec<u32> = Vec::new();
    for v in vals {
        let (i1, i2) = ((v / 32) as usize, v % 32);
        if ret.len() < i1 + 1 {
            ret.resize(i1 + 1, 0);
        }
        ret[i1] |= 1u32 << i2;
    }
    ret
}

/// upstream: utils/common.h `FindInBitset` (`pos >= 0`).
#[inline]
pub fn find_in_bitset(bits: &[u32], pos: i32) -> bool {
    let i1 = (pos / 32) as usize;
    if i1 >= bits.len() {
        return false;
    }
    (bits[i1] >> (pos % 32)) & 1 == 1
}

/// One element of a TreeSHAP decision path (upstream `Tree::PathElement`).
#[derive(Debug, Clone, Copy, Default)]
pub struct PathElement {
    feature_index: i32,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// upstream: `Tree::ExtendPath`.
fn extend_path(p: &mut [PathElement], unique_depth: usize, zero_fraction: f64, one_fraction: f64, feature_index: i32) {
    p[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    let d1 = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        p[i + 1].pweight += one_fraction * p[i].pweight * (i + 1) as f64 / d1;
        p[i].pweight = zero_fraction * p[i].pweight * (unique_depth - i) as f64 / d1;
    }
}

/// upstream: `Tree::UnwindPath`.
fn unwind_path(p: &mut [PathElement], unique_depth: usize, path_index: usize) {
    let one_fraction = p[path_index].one_fraction;
    let zero_fraction = p[path_index].zero_fraction;
    let mut next_one_portion = p[unique_depth].pweight;
    let d1 = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = p[i].pweight;
            p[i].pweight = next_one_portion * d1 / ((i + 1) as f64 * one_fraction);
            next_one_portion = tmp - p[i].pweight * zero_fraction * (unique_depth - i) as f64 / d1;
        } else {
            p[i].pweight = (p[i].pweight * d1) / (zero_fraction * (unique_depth - i) as f64);
        }
    }
    for i in path_index..unique_depth {
        p[i].feature_index = p[i + 1].feature_index;
        p[i].zero_fraction = p[i + 1].zero_fraction;
        p[i].one_fraction = p[i + 1].one_fraction;
    }
}

/// upstream: `Tree::UnwoundPathSum`.
fn unwound_path_sum(p: &[PathElement], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = p[path_index].one_fraction;
    let zero_fraction = p[path_index].zero_fraction;
    let mut next_one_portion = p[unique_depth].pweight;
    let mut total = 0.0;
    let d1 = (unique_depth + 1) as f64;
    for i in (0..unique_depth).rev() {
        if one_fraction != 0.0 {
            let tmp = next_one_portion * d1 / ((i + 1) as f64 * one_fraction);
            total += tmp;
            next_one_portion = p[i].pweight - tmp * zero_fraction * ((unique_depth - i) as f64 / d1);
        } else {
            total += (p[i].pweight / zero_fraction) / ((unique_depth - i) as f64 / d1);
        }
    }
    total
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub num_leaves: usize,
    pub split_feature_inner: Vec<i32>,
    pub split_feature: Vec<i32>,
    pub split_gain: Vec<f32>,
    pub threshold_in_bin: Vec<u32>,
    pub threshold: Vec<f64>,
    pub decision_type: Vec<i8>,
    pub left_child: Vec<i32>,
    pub right_child: Vec<i32>,
    pub leaf_parent: Vec<i32>,
    pub leaf_value: Vec<f64>,
    pub leaf_weight: Vec<f64>,
    pub leaf_count: Vec<i32>,
    pub leaf_depth: Vec<i32>,
    pub internal_value: Vec<f64>,
    pub internal_weight: Vec<f64>,
    pub internal_count: Vec<i32>,
    pub shrinkage: f64,
    /// Number of categorical splits; node `n` of such a split stores its
    /// index in `threshold`/`threshold_in_bin`.
    pub num_cat: i32,
    /// Bitset word ranges per categorical split (`num_cat + 1` entries).
    pub cat_boundaries: Vec<i32>,
    /// Bitsets of the categories sent left.
    pub cat_threshold: Vec<u32>,
    pub cat_boundaries_inner: Vec<i32>,
    /// Bitsets of the bins sent left (training data only; not saved).
    pub cat_threshold_inner: Vec<u32>,
}

/// Arguments of one numerical split (upstream `Tree::Split`).
#[derive(Debug, Clone, Copy)]
pub struct SplitArgs {
    pub leaf: usize,
    pub feature_inner: i32,
    pub feature_real: i32,
    pub threshold_bin: u32,
    pub threshold: f64,
    pub left_value: f64,
    pub right_value: f64,
    pub left_count: i32,
    pub right_count: i32,
    pub left_weight: f64,
    pub right_weight: f64,
    pub gain: f32,
    pub missing_type: MissingType,
    pub default_left: bool,
}

impl Tree {
    pub fn new(max_leaves: usize) -> Self {
        let m = max_leaves.max(1);
        Self {
            num_leaves: 1,
            split_feature_inner: Vec::with_capacity(m - 1),
            split_feature: Vec::with_capacity(m - 1),
            split_gain: Vec::with_capacity(m - 1),
            threshold_in_bin: Vec::with_capacity(m - 1),
            threshold: Vec::with_capacity(m - 1),
            decision_type: Vec::with_capacity(m - 1),
            left_child: Vec::with_capacity(m - 1),
            right_child: Vec::with_capacity(m - 1),
            leaf_parent: vec![-1],
            leaf_value: vec![0.0],
            leaf_weight: vec![0.0],
            leaf_count: vec![0],
            leaf_depth: vec![0],
            internal_value: Vec::with_capacity(m - 1),
            internal_weight: Vec::with_capacity(m - 1),
            internal_count: Vec::with_capacity(m - 1),
            shrinkage: 1.0,
            num_cat: 0,
            cat_boundaries: vec![0],
            cat_threshold: Vec::new(),
            cat_boundaries_inner: vec![0],
            cat_threshold_inner: Vec::new(),
        }
    }

    /// Categorical split of `leaf` (`a.threshold*` and `a.default_left` are
    /// ignored); returns the index of the new (right) leaf.
    ///
    /// upstream: `Tree::SplitCategorical`.
    pub fn split_categorical(&mut self, a: &SplitArgs, bitset_inner: &[u32], bitset: &[u32]) -> usize {
        let args = SplitArgs { threshold_bin: self.num_cat as u32, threshold: self.num_cat as f64, default_left: false, ..*a };
        let right = self.split(&args);
        let node = self.decision_type.len() - 1;
        self.decision_type[node] |= K_CATEGORICAL_MASK;
        self.num_cat += 1;
        self.cat_boundaries.push(self.cat_boundaries.last().unwrap() + bitset.len() as i32);
        self.cat_threshold.extend_from_slice(bitset);
        self.cat_boundaries_inner.push(self.cat_boundaries_inner.last().unwrap() + bitset_inner.len() as i32);
        self.cat_threshold_inner.extend_from_slice(bitset_inner);
        right
    }

    /// Split `leaf`; returns the index of the new (right) leaf.
    pub fn split(&mut self, a: &SplitArgs) -> usize {
        let leaf = a.leaf;
        let new_node = self.num_leaves - 1;
        let new_leaf = self.num_leaves;
        let parent = self.leaf_parent[leaf];
        if parent >= 0 {
            let p = parent as usize;
            if self.left_child[p] == !(leaf as i32) {
                self.left_child[p] = new_node as i32;
            } else {
                self.right_child[p] = new_node as i32;
            }
        }
        self.split_feature_inner.push(a.feature_inner);
        self.split_feature.push(a.feature_real);
        self.split_gain.push(a.gain);
        self.left_child.push(!(leaf as i32));
        self.right_child.push(!(new_leaf as i32));
        self.leaf_parent[leaf] = new_node as i32;
        self.leaf_parent.push(new_node as i32);
        self.internal_weight.push(a.left_weight + a.right_weight);
        self.internal_value.push(self.leaf_value[leaf]);
        self.internal_count.push(a.left_count + a.right_count);
        self.leaf_value[leaf] = if a.left_value.is_nan() { 0.0 } else { a.left_value };
        self.leaf_weight[leaf] = a.left_weight;
        self.leaf_count[leaf] = a.left_count;
        self.leaf_value.push(if a.right_value.is_nan() { 0.0 } else { a.right_value });
        self.leaf_weight.push(a.right_weight);
        self.leaf_count.push(a.right_count);
        let d = self.leaf_depth[leaf] + 1;
        self.leaf_depth[leaf] = d;
        self.leaf_depth.push(d);

        let mut dt: i8 = 0;
        if a.default_left {
            dt |= K_DEFAULT_LEFT_MASK;
        }
        dt = (dt & 3) | ((a.missing_type as i8) << 2);
        self.decision_type.push(dt);
        self.threshold_in_bin.push(a.threshold_bin);
        self.threshold.push(a.threshold);
        self.num_leaves += 1;
        new_leaf
    }

    pub fn set_leaf_output(&mut self, leaf: usize, v: f64) {
        self.leaf_value[leaf] = maybe_round_to_zero(v);
    }

    pub fn shrink(&mut self, rate: f64) {
        for i in 0..self.num_leaves - 1 {
            self.leaf_value[i] = maybe_round_to_zero(self.leaf_value[i] * rate);
            self.internal_value[i] = maybe_round_to_zero(self.internal_value[i] * rate);
        }
        let last = self.num_leaves - 1;
        self.leaf_value[last] = maybe_round_to_zero(self.leaf_value[last] * rate);
        self.shrinkage *= rate;
    }

    pub fn add_bias(&mut self, v: f64) {
        for i in 0..self.num_leaves - 1 {
            self.leaf_value[i] = maybe_round_to_zero(self.leaf_value[i] + v);
            self.internal_value[i] = maybe_round_to_zero(self.internal_value[i] + v);
        }
        let last = self.num_leaves - 1;
        self.leaf_value[last] = maybe_round_to_zero(self.leaf_value[last] + v);
        self.shrinkage = 1.0;
    }

    pub fn as_constant(&mut self, v: f64, count: i32) {
        self.num_leaves = 1;
        self.shrinkage = 1.0;
        self.leaf_value.truncate(1);
        self.leaf_value[0] = v;
        self.leaf_count.truncate(1);
        self.leaf_count[0] = count;
        self.leaf_weight.truncate(1);
        self.leaf_parent.truncate(1);
        self.leaf_depth.truncate(1);
        for v in [
            &mut self.split_feature_inner,
            &mut self.split_feature,
            &mut self.left_child,
            &mut self.right_child,
            &mut self.internal_count,
        ] {
            v.clear();
        }
        self.split_gain.clear();
        self.threshold_in_bin.clear();
        self.threshold.clear();
        self.decision_type.clear();
        self.internal_value.clear();
        self.internal_weight.clear();
    }

    #[inline]
    fn default_left(&self, node: usize) -> bool {
        self.decision_type[node] & K_DEFAULT_LEFT_MASK > 0
    }

    #[inline]
    fn missing_type(&self, node: usize) -> i8 {
        (self.decision_type[node] >> 2) & 3
    }

    #[inline]
    fn numerical_decision(&self, mut fval: f64, node: usize) -> i32 {
        let mt = self.missing_type(node);
        if fval.is_nan() && mt != MissingType::NaN as i8 {
            fval = 0.0;
        }
        if (mt == MissingType::Zero as i8 && is_zero(fval))
            || (mt == MissingType::NaN as i8 && fval.is_nan())
        {
            return if self.default_left(node) { self.left_child[node] } else { self.right_child[node] };
        }
        if fval <= self.threshold[node] { self.left_child[node] } else { self.right_child[node] }
    }

    #[inline]
    pub fn is_categorical(&self, node: usize) -> bool {
        self.decision_type[node] & K_CATEGORICAL_MASK > 0
    }

    /// Category bitset of categorical node `node`.
    pub fn cat_bitset(&self, node: usize) -> &[u32] {
        let c = self.threshold[node] as usize;
        &self.cat_threshold[self.cat_boundaries[c] as usize..self.cat_boundaries[c + 1] as usize]
    }

    /// upstream: `Tree::CategoricalDecision` (NaN and negatives go right).
    #[inline]
    fn categorical_decision(&self, fval: f64, node: usize) -> i32 {
        if fval.is_nan() {
            return self.right_child[node];
        }
        let v = fval as i32;
        if v >= 0 && find_in_bitset(self.cat_bitset(node), v) {
            self.left_child[node]
        } else {
            self.right_child[node]
        }
    }

    /// upstream: `Tree::Decision`.
    #[inline]
    pub fn decision(&self, fval: f64, node: usize) -> i32 {
        if self.is_categorical(node) {
            self.categorical_decision(fval, node)
        } else {
            self.numerical_decision(fval, node)
        }
    }

    /// Leaf index for one row of raw feature values (indexed by real feature).
    #[inline]
    pub fn get_leaf(&self, row: &[f64]) -> usize {
        if self.num_leaves <= 1 {
            return 0;
        }
        let mut node: i32 = 0;
        while node >= 0 {
            let n = node as usize;
            let f = self.split_feature[n] as usize;
            let v = if f < row.len() { row[f] } else { 0.0 };
            node = self.decision(v, n);
        }
        !node as usize
    }

    #[inline]
    pub fn predict(&self, row: &[f64]) -> f64 {
        self.leaf_value[self.get_leaf(row)]
    }

    #[inline]
    fn data_count(&self, node: i32) -> f64 {
        if node >= 0 { self.internal_count[node as usize] as f64 } else { self.leaf_count[!node as usize] as f64 }
    }

    /// upstream: `Tree::ExpectedValue`.
    pub fn expected_value(&self) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_value[0];
        }
        let total_count = self.internal_count[0] as f64;
        let mut exp_value = 0.0;
        for i in 0..self.num_leaves {
            exp_value += (self.leaf_count[i] as f64 / total_count) * self.leaf_value[i];
        }
        exp_value
    }

    /// Add this tree's SHAP values for `row` to `out[..num_features]` and its
    /// expected value to `out[num_features]`. `path` is scratch space.
    ///
    /// upstream: `Tree::PredictContrib`.
    pub fn predict_contrib(
        &self,
        row: &[f64],
        num_features: usize,
        expected_value: f64,
        out: &mut [f64],
        path: &mut Vec<PathElement>,
    ) {
        out[num_features] += expected_value;
        if self.num_leaves > 1 {
            let max_path_len = self.max_depth() as usize + 1;
            path.clear();
            path.resize(max_path_len * (max_path_len + 1) / 2, PathElement::default());
            self.tree_shap(row, out, 0, 0, path, 0, 1.0, 1.0, -1);
        }
    }

    /// upstream: `Tree::TreeSHAP`; the unique path of depth `unique_depth`
    /// lives at `path[parent + unique_depth..]`.
    #[allow(clippy::too_many_arguments)]
    fn tree_shap(
        &self,
        row: &[f64],
        phi: &mut [f64],
        node: i32,
        mut unique_depth: usize,
        path: &mut [PathElement],
        parent: usize,
        parent_zero_fraction: f64,
        parent_one_fraction: f64,
        parent_feature_index: i32,
    ) {
        let base = parent + unique_depth;
        if unique_depth > 0 {
            path.copy_within(parent..parent + unique_depth, base);
        }
        extend_path(&mut path[base..], unique_depth, parent_zero_fraction, parent_one_fraction, parent_feature_index);

        if node < 0 {
            let unique_path = &path[base..];
            let leaf_value = self.leaf_value[!node as usize];
            for i in 1..=unique_depth {
                let w = unwound_path_sum(unique_path, unique_depth, i);
                let el = unique_path[i];
                phi[el.feature_index as usize] += w * (el.one_fraction - el.zero_fraction) * leaf_value;
            }
        } else {
            let n = node as usize;
            let feature = self.split_feature[n];
            let fval = row.get(feature as usize).copied().unwrap_or(0.0);
            let hot_index = self.decision(fval, n);
            let cold_index = if hot_index == self.left_child[n] { self.right_child[n] } else { self.left_child[n] };
            let w = self.data_count(node);
            let hot_zero_fraction = self.data_count(hot_index) / w;
            let cold_zero_fraction = self.data_count(cold_index) / w;
            let mut incoming_zero_fraction = 1.0;
            let mut incoming_one_fraction = 1.0;

            // undo an earlier split on the same feature so it can be redone here
            let unique_path = &mut path[base..];
            let path_index = (0..=unique_depth).find(|&i| unique_path[i].feature_index == feature);
            if let Some(path_index) = path_index {
                incoming_zero_fraction = unique_path[path_index].zero_fraction;
                incoming_one_fraction = unique_path[path_index].one_fraction;
                unwind_path(unique_path, unique_depth, path_index);
                unique_depth -= 1;
            }

            self.tree_shap(row, phi, hot_index, unique_depth + 1, path, base,
                           hot_zero_fraction * incoming_zero_fraction, incoming_one_fraction, feature);
            self.tree_shap(row, phi, cold_index, unique_depth + 1, path, base,
                           cold_zero_fraction * incoming_zero_fraction, 0.0, feature);
        }
    }

    /// Leaf index of row `i` of a binned dataset. Requires the tree to have
    /// been trained on a dataset sharing `data`'s bin mappers.
    #[inline]
    pub fn get_leaf_binned(&self, data: &Dataset, i: usize) -> usize {
        if self.num_leaves <= 1 {
            return 0;
        }
        let mut node: i32 = 0;
        while node >= 0 {
            let n = node as usize;
            let inner = self.split_feature_inner[n] as usize;
            let m = data.feature_bin_mapper(inner);
            let b = data.feature_bins(inner).get(i);
            if self.is_categorical(n) {
                // upstream: Tree::CategoricalDecisionInner
                let c = self.threshold_in_bin[n] as usize;
                let bits = &self.cat_threshold_inner
                    [self.cat_boundaries_inner[c] as usize..self.cat_boundaries_inner[c + 1] as usize];
                node = if find_in_bitset(bits, b as i32) { self.left_child[n] } else { self.right_child[n] };
                continue;
            }
            let mt = self.missing_type(n);
            let is_default = (mt == MissingType::Zero as i8 && b == m.default_bin)
                || (mt == MissingType::NaN as i8 && b == (m.num_bin - 1) as u32);
            node = if is_default {
                if self.default_left(n) { self.left_child[n] } else { self.right_child[n] }
            } else if b <= self.threshold_in_bin[n] {
                self.left_child[n]
            } else {
                self.right_child[n]
            };
        }
        !node as usize
    }

    /// upstream: `Tree::AddPredictionToScore` on a binned dataset.
    pub fn add_prediction_to_score(&self, data: &Dataset, score: &mut [f64]) {
        if self.num_leaves <= 1 {
            let v = self.leaf_value[0];
            for s in score.iter_mut() {
                *s += v;
            }
            return;
        }
        use rayon::prelude::*;
        score.par_iter_mut().with_min_len(4096).enumerate().for_each(|(i, s)| {
            *s += self.leaf_value[self.get_leaf_binned(data, i)];
        });
    }

    /// upstream: `Tree::AddPredictionToScore(data, used_data_indices, num_data, score)`.
    pub fn add_prediction_to_score_rows(&self, data: &Dataset, rows: &[u32], score: &mut [f64]) {
        if self.num_leaves <= 1 {
            let v = self.leaf_value[0];
            for &i in rows {
                score[i as usize] += v;
            }
            return;
        }
        use rayon::prelude::*;
        let leaves: Vec<usize> =
            rows.par_iter().with_min_len(4096).map(|&i| self.get_leaf_binned(data, i as usize)).collect();
        for (&i, leaf) in rows.iter().zip(leaves) {
            score[i as usize] += self.leaf_value[leaf];
        }
    }

    pub fn max_depth(&self) -> i32 {
        if self.num_leaves <= 1 { 0 } else { *self.leaf_depth.iter().max().unwrap() }
    }

    /// upstream: `Tree::ToString`.
    pub fn to_model_string(&self) -> String {
        fn join<T>(v: &[T], f: impl Fn(&T) -> String) -> String {
            v.iter().map(f).collect::<Vec<_>>().join(" ")
        }
        let n = self.num_leaves;
        let ni = n - 1;
        let mut s = String::new();
        s.push_str(&format!("num_leaves={n}\n"));
        s.push_str(&format!("num_cat={}\n", self.num_cat));
        s.push_str(&format!("split_feature={}\n", join(&self.split_feature[..ni], |x| x.to_string())));
        s.push_str(&format!("split_gain={}\n", join(&self.split_gain[..ni], |x| fmt_g6(*x as f64))));
        s.push_str(&format!("threshold={}\n", join(&self.threshold[..ni], |x| fmt_g17(*x))));
        s.push_str(&format!("decision_type={}\n", join(&self.decision_type[..ni], |x| x.to_string())));
        s.push_str(&format!("left_child={}\n", join(&self.left_child[..ni], |x| x.to_string())));
        s.push_str(&format!("right_child={}\n", join(&self.right_child[..ni], |x| x.to_string())));
        s.push_str(&format!("leaf_value={}\n", join(&self.leaf_value[..n], |x| fmt_g17(*x))));
        // ArrayToString prints at most the stored values; a loaded one-leaf tree stores no weights
        let nw = n.min(self.leaf_weight.len());
        s.push_str(&format!("leaf_weight={}\n", join(&self.leaf_weight[..nw], |x| fmt_g17(*x))));
        s.push_str(&format!("leaf_count={}\n", join(&self.leaf_count[..n], |x| x.to_string())));
        s.push_str(&format!("internal_value={}\n", join(&self.internal_value[..ni], |x| fmt_g6(*x))));
        s.push_str(&format!("internal_weight={}\n", join(&self.internal_weight[..ni], |x| fmt_g6(*x))));
        s.push_str(&format!("internal_count={}\n", join(&self.internal_count[..ni], |x| x.to_string())));
        if self.num_cat > 0 {
            let nb = self.num_cat as usize + 1;
            s.push_str(&format!("cat_boundaries={}\n", join(&self.cat_boundaries[..nb], |x| x.to_string())));
            s.push_str(&format!("cat_threshold={}\n", join(&self.cat_threshold, |x| x.to_string())));
        }
        s.push_str("is_linear=0\n");
        s.push_str(&format!("shrinkage={}\n", fmt_g6(self.shrinkage)));
        s.push('\n');
        s
    }

    /// upstream: `Tree::ToJSON` (stream precision 17, i.e. `%.17g`).
    pub fn to_json(&self) -> String {
        let mut s = format!(
            "\"num_leaves\":{},\n\"num_cat\":{},\n\"shrinkage\":{},\n",
            self.num_leaves,
            self.num_cat,
            fmt_g17(self.shrinkage)
        );
        if self.num_leaves == 1 {
            s.push_str(&format!(
                "\"tree_structure\":{{\"leaf_value\":{}, \n\"leaf_count\":{}}}\n",
                fmt_g17(self.leaf_value[0]),
                self.leaf_count[0]
            ));
        } else {
            s.push_str("\"tree_structure\":");
            self.node_to_json(0, &mut s);
            s.push('\n');
        }
        s
    }

    /// upstream: `Tree::NodeToJSON`.
    fn node_to_json(&self, index: i32, s: &mut String) {
        if index >= 0 {
            let i = index as usize;
            let missing = match self.missing_type(i) {
                0 => "None",
                1 => "Zero",
                _ => "NaN",
            };
            let (threshold, op) = if self.is_categorical(i) {
                let bits = self.cat_bitset(i);
                let cats: Vec<String> = (0..bits.len() as i32 * 32)
                    .filter(|&c| find_in_bitset(bits, c))
                    .map(|c| c.to_string())
                    .collect();
                (format!("\"{}\"", cats.join("||")), "==")
            } else {
                (fmt_g17(avoid_inf_f64(self.threshold[i])), "<=")
            };
            s.push_str(&format!(
                "{{\n\"split_index\":{index},\n\"split_feature\":{},\n\"split_gain\":{},\n\"threshold\":{threshold},\n\
                 \"decision_type\":\"{op}\",\n\"default_left\":{},\n\"missing_type\":\"{missing}\",\n\
                 \"internal_value\":{},\n\"internal_weight\":{},\n\"internal_count\":{},\n\"left_child\":",
                self.split_feature[i],
                fmt_g17(avoid_inf_f32(self.split_gain[i]) as f64),
                self.default_left(i),
                fmt_g17(self.internal_value[i]),
                fmt_g17(self.internal_weight[i]),
                self.internal_count[i],
            ));
            self.node_to_json(self.left_child[i], s);
            s.push_str(",\n\"right_child\":");
            self.node_to_json(self.right_child[i], s);
            s.push_str("\n}");
        } else {
            let leaf = !index as usize;
            s.push_str(&format!(
                "{{\n\"leaf_index\":{leaf},\n\"leaf_value\":{},\n\"leaf_weight\":{},\n\"leaf_count\":{}\n}}",
                fmt_g17(self.leaf_value[leaf]),
                fmt_g17(self.leaf_weight.get(leaf).copied().unwrap_or(0.0)),
                self.leaf_count[leaf],
            ));
        }
    }

    /// Parse a tree block (the lines after `Tree=i`) as produced by upstream
    /// `Tree::ToString`. Returns the tree and the number of bytes consumed.
    pub fn from_model_str(text: &str) -> Result<(Self, usize)> {
        let mut kv: HashMap<&str, &str> = HashMap::new();
        let mut consumed = 0usize;
        let bytes = text.as_bytes();
        let mut read = 0;
        while read < 22 && consumed < bytes.len() {
            if bytes[consumed] == b'\r' || bytes[consumed] == b'\n' {
                break;
            }
            let line_end = text[consumed..].find(['\r', '\n']).map_or(text.len(), |p| consumed + p);
            let line = &text[consumed..line_end];
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| LgbmError::ModelFormat(format!("bad tree line: {line}")))?;
            kv.insert(k, v);
            read += 1;
            consumed = line_end;
            if consumed < bytes.len() && bytes[consumed] == b'\r' {
                consumed += 1;
            }
            if consumed < bytes.len() && bytes[consumed] == b'\n' {
                consumed += 1;
            }
        }
        fn need<'a>(kv: &HashMap<&str, &'a str>, k: &str) -> Result<&'a str> {
            kv.get(k).copied().ok_or_else(|| {
                LgbmError::ModelFormat(format!("Tree model string format error, should contain {k} field"))
            })
        }
        fn arr<T: std::str::FromStr>(s: &str, n: usize, name: &str) -> Result<Vec<T>> {
            let v: Vec<T> = s
                .split_whitespace()
                .map(|t| {
                    t.parse::<T>().map_err(|_| LgbmError::ModelFormat(format!("bad value {t:?} in {name}")))
                })
                .collect::<Result<_>>()?;
            if v.len() != n {
                return Err(LgbmError::ModelFormat(format!(
                    "{name} has {} values, expected {n}",
                    v.len()
                )));
            }
            Ok(v)
        }
        fn arr_f64(s: &str, n: usize, name: &str) -> Result<Vec<f64>> {
            let v: Vec<f64> = s
                .split_whitespace()
                .map(|t| crate::fmt::parse_f64(t).ok_or_else(|| LgbmError::ModelFormat(format!("bad value {t:?} in {name}"))))
                .collect::<Result<_>>()?;
            if v.len() != n {
                return Err(LgbmError::ModelFormat(format!("{name} has {} values, expected {n}", v.len())));
            }
            Ok(v)
        }
        // upstream: StringToArrayFast (legacy Common::Atof)
        fn arr_f64_legacy(s: &str, n: usize, name: &str) -> Result<Vec<f64>> {
            let v: Vec<f64> = s
                .split_whitespace()
                .map(|t| crate::fmt::atof_legacy(t).ok_or_else(|| LgbmError::ModelFormat(format!("bad value {t:?} in {name}"))))
                .collect::<Result<_>>()?;
            if v.len() != n {
                return Err(LgbmError::ModelFormat(format!("{name} has {} values, expected {n}", v.len())));
            }
            Ok(v)
        }
        let num_leaves: usize = need(&kv, "num_leaves")?
            .trim()
            .parse()
            .map_err(|_| LgbmError::ModelFormat("bad num_leaves".into()))?;
        if num_leaves == 0 {
            return Err(LgbmError::ModelFormat("num_leaves must be >= 1".into()));
        }
        let num_cat: i32 = need(&kv, "num_cat")?
            .trim()
            .parse()
            .ok()
            .filter(|&c| c >= 0)
            .ok_or_else(|| LgbmError::ModelFormat("bad num_cat".into()))?;
        if kv.get("is_linear").is_some_and(|v| v.trim() != "0") {
            return Err(LgbmError::Unsupported("linear trees".into()));
        }
        let n = num_leaves;
        let ni = n - 1;
        let mut t = Tree::new(n);
        t.num_leaves = n;
        t.num_cat = num_cat;
        t.leaf_value = arr_f64(need(&kv, "leaf_value")?, n, "leaf_value")?;
        t.shrinkage = kv.get("shrinkage").and_then(|s| crate::fmt::atof_legacy(s.trim())).unwrap_or(1.0);
        t.leaf_count = match kv.get("leaf_count") {
            Some(s) => arr(s, n, "leaf_count")?,
            None => vec![0; n],
        };
        t.leaf_parent = vec![-1; n];
        t.leaf_depth = vec![0; n];
        // upstream Tree(const char*) returns before reading the remaining fields of a one-leaf tree
        t.leaf_weight = Vec::new();
        if n > 1 {
            t.leaf_weight = match kv.get("leaf_weight") {
                Some(s) => arr_f64(s, n, "leaf_weight")?,
                None => vec![0.0; n],
            };
            t.left_child = arr(need(&kv, "left_child")?, ni, "left_child")?;
            t.right_child = arr(need(&kv, "right_child")?, ni, "right_child")?;
            t.split_feature = arr(need(&kv, "split_feature")?, ni, "split_feature")?;
            t.threshold = arr_f64(need(&kv, "threshold")?, ni, "threshold")?;
            t.split_gain = match kv.get("split_gain") {
                Some(s) => arr_f64_legacy(s, ni, "split_gain")?.into_iter().map(|x| x as f32).collect(),
                None => vec![0.0; ni],
            };
            t.internal_count = match kv.get("internal_count") {
                Some(s) => arr(s, ni, "internal_count")?,
                None => vec![0; ni],
            };
            t.internal_value = match kv.get("internal_value") {
                Some(s) => arr_f64_legacy(s, ni, "internal_value")?,
                None => vec![0.0; ni],
            };
            t.internal_weight = match kv.get("internal_weight") {
                Some(s) => arr_f64_legacy(s, ni, "internal_weight")?,
                None => vec![0.0; ni],
            };
            t.decision_type = match kv.get("decision_type") {
                Some(s) => arr(s, ni, "decision_type")?,
                None => vec![0; ni],
            };
            if num_cat > 0 {
                let nb = num_cat as usize + 1;
                t.cat_boundaries = arr(
                    kv.get("cat_boundaries")
                        .ok_or_else(|| LgbmError::ModelFormat("Tree model should contain cat_boundaries field.".into()))?,
                    nb,
                    "cat_boundaries",
                )?;
                if t.cat_boundaries[0] != 0 || t.cat_boundaries.windows(2).any(|w| w[0] > w[1]) {
                    return Err(LgbmError::ModelFormat("cat_boundaries must be non-decreasing from 0".into()));
                }
                t.cat_threshold = arr(
                    kv.get("cat_threshold")
                        .ok_or_else(|| LgbmError::ModelFormat("Tree model should contain cat_threshold field".into()))?,
                    t.cat_boundaries[nb - 1] as usize,
                    "cat_threshold",
                )?;
            }
            for node in 0..ni {
                if t.decision_type[node] & K_CATEGORICAL_MASK > 0 {
                    let c = t.threshold[node];
                    if !(c >= 0.0 && (c as i32) < num_cat) {
                        return Err(LgbmError::ModelFormat(format!("categorical split index {c} out of range")));
                    }
                }
            }
            for &c in t.left_child.iter().chain(&t.right_child) {
                let ok = if c >= 0 { (c as usize) < ni } else { ((!c) as usize) < n };
                if !ok {
                    return Err(LgbmError::ModelFormat(format!("child index {c} out of range")));
                }
            }
            t.split_feature_inner = vec![-1; ni];
            t.threshold_in_bin = vec![0; ni];
            t.recompute_leaf_depths(0, 0);
            for node in 0..ni {
                for c in [t.left_child[node], t.right_child[node]] {
                    if c < 0 {
                        t.leaf_parent[(!c) as usize] = node as i32;
                    }
                }
            }
        }
        Ok((t, consumed))
    }

    fn recompute_leaf_depths(&mut self, node: i32, depth: i32) {
        if node < 0 {
            self.leaf_depth[(!node) as usize] = depth;
        } else {
            let n = node as usize;
            let (l, r) = (self.left_child[n], self.right_child[n]);
            self.recompute_leaf_depths(l, depth + 1);
            self.recompute_leaf_depths(r, depth + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_split_tree() -> Tree {
        let mut t = Tree::new(3);
        let base = SplitArgs {
            leaf: 0,
            feature_inner: 0,
            feature_real: 0,
            threshold_bin: 3,
            threshold: 1.5,
            left_value: -1.0,
            right_value: 1.0,
            left_count: 10,
            right_count: 20,
            left_weight: 10.0,
            right_weight: 20.0,
            gain: 5.0,
            missing_type: MissingType::NaN,
            default_left: true,
        };
        t.split(&base);
        t.split(&SplitArgs {
            leaf: 1,
            feature_inner: 1,
            feature_real: 2,
            threshold: 0.25,
            left_value: 0.5,
            right_value: 2.0,
            left_count: 5,
            right_count: 15,
            missing_type: MissingType::None,
            default_left: false,
            ..base
        });
        t
    }

    #[test]
    fn split_structure_matches_upstream_convention() {
        let t = two_split_tree();
        assert_eq!(t.num_leaves, 3);
        assert_eq!(t.left_child, vec![-1, -2]);
        assert_eq!(t.right_child, vec![1, -3]);
        assert_eq!(t.decision_type, vec![2 | (2 << 2), 0]);
        assert_eq!(t.leaf_depth, vec![1, 2, 2]);
    }

    #[test]
    fn predict_with_missing() {
        let t = two_split_tree();
        assert_eq!(t.get_leaf(&[f64::NAN, 0.0, 0.0]), 0); // NaN default-left
        assert_eq!(t.get_leaf(&[1.0, 0.0, 0.0]), 0);
        assert_eq!(t.get_leaf(&[2.0, 0.0, 0.1]), 1);
        assert_eq!(t.get_leaf(&[2.0, 0.0, f64::NAN]), 1); // NaN -> 0 for None missing
        assert_eq!(t.get_leaf(&[2.0, 0.0, 0.3]), 2);
    }

    #[test]
    fn categorical_split_roundtrip() {
        let mut t = two_split_tree();
        let args = SplitArgs {
            leaf: 2,
            feature_inner: 2,
            feature_real: 1,
            threshold_bin: 0,
            threshold: 0.0,
            left_value: 3.0,
            right_value: 4.0,
            left_count: 7,
            right_count: 8,
            left_weight: 7.0,
            right_weight: 8.0,
            gain: 1.0,
            missing_type: MissingType::NaN,
            default_left: true,
        };
        // categories 1 and 33 go left
        t.split_categorical(&args, &construct_bitset([2, 5].into_iter()), &construct_bitset([1, 33].into_iter()));
        assert_eq!(t.num_cat, 1);
        assert_eq!(t.decision_type[2], 1 | (2 << 2));
        assert_eq!(t.cat_threshold, vec![2, 2]);
        let leaf = |c: f64| t.get_leaf(&[2.0, c, 0.3]);
        assert_eq!((leaf(1.0), leaf(33.9), leaf(2.0), leaf(-1.0), leaf(f64::NAN)), (2, 2, 3, 3, 3));
        let s = t.to_model_string();
        assert!(s.contains("num_cat=1\n") && s.contains("cat_boundaries=0 2\ncat_threshold=2 2\n"));
        let (u, _) = Tree::from_model_str(&s).unwrap();
        assert_eq!(u.to_model_string(), s);
        assert_eq!(u.get_leaf(&[2.0, 33.0, 0.3]), 2);
        assert!(t.to_json().contains("\"threshold\":\"1||33\",\n\"decision_type\":\"==\""));
    }

    #[test]
    fn text_roundtrip() {
        let mut t = two_split_tree();
        t.shrink(0.1);
        let s = t.to_model_string();
        let (u, used) = Tree::from_model_str(&s).unwrap();
        assert_eq!(used, s.len() - 1);
        assert_eq!(u.to_model_string(), s);
        for row in [[f64::NAN, 0.0, 0.0], [2.0, 0.0, 0.3], [2.0, 0.0, 0.1]] {
            assert_eq!(u.predict(&row), t.predict(&row));
        }
    }
}
