//! Decision tree model.
//!
//! upstream: include/LightGBM/tree.h, src/io/tree.cpp (numerical splits only).

use std::collections::HashMap;

use crate::binning::MissingType;
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::Dataset;
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
        }
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
            node = self.numerical_decision(v, n);
        }
        !node as usize
    }

    #[inline]
    pub fn predict(&self, row: &[f64]) -> f64 {
        self.leaf_value[self.get_leaf(row)]
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
        s.push_str("num_cat=0\n");
        s.push_str(&format!("split_feature={}\n", join(&self.split_feature[..ni], |x| x.to_string())));
        s.push_str(&format!("split_gain={}\n", join(&self.split_gain[..ni], |x| fmt_g6(*x as f64))));
        s.push_str(&format!("threshold={}\n", join(&self.threshold[..ni], |x| fmt_g17(*x))));
        s.push_str(&format!("decision_type={}\n", join(&self.decision_type[..ni], |x| x.to_string())));
        s.push_str(&format!("left_child={}\n", join(&self.left_child[..ni], |x| x.to_string())));
        s.push_str(&format!("right_child={}\n", join(&self.right_child[..ni], |x| x.to_string())));
        s.push_str(&format!("leaf_value={}\n", join(&self.leaf_value[..n], |x| fmt_g17(*x))));
        s.push_str(&format!("leaf_weight={}\n", join(&self.leaf_weight[..n], |x| fmt_g17(*x))));
        s.push_str(&format!("leaf_count={}\n", join(&self.leaf_count[..n], |x| x.to_string())));
        s.push_str(&format!("internal_value={}\n", join(&self.internal_value[..ni], |x| fmt_g6(*x))));
        s.push_str(&format!("internal_weight={}\n", join(&self.internal_weight[..ni], |x| fmt_g6(*x))));
        s.push_str(&format!("internal_count={}\n", join(&self.internal_count[..ni], |x| x.to_string())));
        s.push_str("is_linear=0\n");
        s.push_str(&format!("shrinkage={}\n", fmt_g6(self.shrinkage)));
        s.push('\n');
        s
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
        let num_leaves: usize = need(&kv, "num_leaves")?
            .trim()
            .parse()
            .map_err(|_| LgbmError::ModelFormat("bad num_leaves".into()))?;
        if num_leaves == 0 {
            return Err(LgbmError::ModelFormat("num_leaves must be >= 1".into()));
        }
        let num_cat: i32 = need(&kv, "num_cat")?.trim().parse().unwrap_or(-1);
        if num_cat != 0 {
            return Err(LgbmError::Unsupported("models with categorical splits".into()));
        }
        if kv.get("is_linear").is_some_and(|v| v.trim() != "0") {
            return Err(LgbmError::Unsupported("linear trees".into()));
        }
        let n = num_leaves;
        let ni = n - 1;
        let mut t = Tree::new(n);
        t.num_leaves = n;
        t.leaf_value = arr_f64(need(&kv, "leaf_value")?, n, "leaf_value")?;
        t.shrinkage = kv.get("shrinkage").and_then(|s| crate::fmt::parse_f64(s.trim())).unwrap_or(1.0);
        t.leaf_count = match kv.get("leaf_count") {
            Some(s) => arr(s, n, "leaf_count")?,
            None => vec![0; n],
        };
        t.leaf_weight = match kv.get("leaf_weight") {
            Some(s) => arr_f64(s, n, "leaf_weight")?,
            None => vec![0.0; n],
        };
        t.leaf_parent = vec![-1; n];
        t.leaf_depth = vec![0; n];
        if n > 1 {
            t.left_child = arr(need(&kv, "left_child")?, ni, "left_child")?;
            t.right_child = arr(need(&kv, "right_child")?, ni, "right_child")?;
            t.split_feature = arr(need(&kv, "split_feature")?, ni, "split_feature")?;
            t.threshold = arr_f64(need(&kv, "threshold")?, ni, "threshold")?;
            t.split_gain = match kv.get("split_gain") {
                Some(s) => arr_f64(s, ni, "split_gain")?.into_iter().map(|x| x as f32).collect(),
                None => vec![0.0; ni],
            };
            t.internal_count = match kv.get("internal_count") {
                Some(s) => arr(s, ni, "internal_count")?,
                None => vec![0; ni],
            };
            t.internal_value = match kv.get("internal_value") {
                Some(s) => arr_f64(s, ni, "internal_value")?,
                None => vec![0.0; ni],
            };
            t.internal_weight = match kv.get("internal_weight") {
                Some(s) => arr_f64(s, ni, "internal_weight")?,
                None => vec![0.0; ni],
            };
            t.decision_type = match kv.get("decision_type") {
                Some(s) => arr(s, ni, "decision_type")?,
                None => vec![0; ni],
            };
            if t.decision_type.iter().any(|d| d & K_CATEGORICAL_MASK > 0) {
                return Err(LgbmError::Unsupported("models with categorical splits".into()));
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
