//! Forced splits: splits read from a JSON tree and applied breadth-first at
//! the top of every tree, before the leaf-wise search.
//!
//! upstream: src/treelearner/serial_tree_learner.cpp (`ForceSplits`,
//! `FindAllForceFeatures`) and src/boosting/gbdt.cpp (loading
//! `forcedsplits_filename`, `CheckForcedSplitFeatures`). The JSON is read with
//! json11's semantics: a missing or mistyped key reads as null, 0 or empty.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use serde_json::Value;

use super::SerialTreeLearner;
use super::split::{SplitInfo, gather_info_for_threshold};
use crate::error::{LgbmError, Result};
use crate::tree::Tree;

static NULL: Value = Value::Null;

/// json11 `operator[]` on an object key.
fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
    v.get(key).unwrap_or(&NULL)
}

/// json11 `object_items().count(key) > 0`.
fn has(v: &Value, key: &str) -> bool {
    v.as_object().is_some_and(|o| o.contains_key(key))
}

/// json11 `int_value()`: `static_cast<int>` of a number, else 0.
fn int_value(v: &Value) -> i32 {
    v.as_f64().map_or(0, |x| x as i32)
}

/// json11 `number_value()`.
fn number_value(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Load `forcedsplits_filename` as `GBDT::Init` does (a missing or invalid
/// file gives no forced splits) and run `CheckForcedSplitFeatures`, which
/// upstream also runs on the null document of an unset file.
pub(crate) fn load_forced_splits(path: &str, max_feature_idx: i32) -> Result<Option<Arc<Value>>> {
    let json = read_json(path);
    let mut q = VecDeque::from([&json]);
    while let Some(node) = q.pop_front() {
        let feature = int_value(get(node, "feature"));
        if feature > max_feature_idx {
            return Err(LgbmError::InvalidParameter(format!(
                "Forced splits file includes feature index {feature}, but maximum feature index in dataset is \
                 {max_feature_idx}"
            )));
        }
        for key in ["left", "right"] {
            if has(node, key) {
                q.push_back(get(node, key));
            }
        }
    }
    Ok((!json.is_null()).then(|| Arc::new(json)))
}

/// Reload `forcedsplits_filename` as `GBDT::ResetConfig` does: no
/// `CheckForcedSplitFeatures`.
pub(crate) fn reload_forced_splits(path: &str) -> Option<Arc<Value>> {
    let json = read_json(path);
    (!json.is_null()).then(|| Arc::new(json))
}

fn read_json(path: &str) -> Value {
    if path.is_empty() {
        return Value::Null;
    }
    std::fs::read_to_string(path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or(Value::Null)
}

fn unsupported(what: &str) -> LgbmError {
    LgbmError::Unsupported(format!("forced splits {what}"))
}

impl SerialTreeLearner {
    pub fn set_forced_split(&mut self, json: Option<Arc<Value>>) {
        self.forced_split = json;
    }

    /// upstream `FindAllForceFeatures`: the inner features of every node.
    fn force_features(&self, root: &Value) -> Vec<bool> {
        let mut used = vec![false; self.data.num_features()];
        let mut q = VecDeque::from([root]);
        while let Some(node) = q.pop_front() {
            if let Some(inner) = self.inner_of(int_value(get(node, "feature"))) {
                used[inner] = true;
            }
            for key in ["left", "right"] {
                if has(node, key) {
                    q.push_back(get(node, key));
                }
            }
        }
        used
    }

    fn inner_of(&self, feature: i32) -> Option<usize> {
        usize::try_from(feature).ok().and_then(|f| self.data.inner_feature_index(f))
    }

    /// The forced split of `node` on the smaller or larger leaf of the last
    /// split, from the histograms of the last histogram build (which belong
    /// to other leaves when that build was skipped, as upstream).
    fn gather_forced(&self, node: &Value, smaller: bool) -> Result<SplitInfo> {
        let feature = int_value(get(node, "feature"));
        let inner = self.inner_of(feature).ok_or_else(|| {
            unsupported(&format!(
                "on feature {feature}, which is not used for training (upstream reads out of bounds)"
            ))
        })?;
        let (ls, slot) = if smaller { (self.smaller, self.hist_slots.0) } else { (self.larger, self.hist_slots.1) };
        let hist = usize::try_from(slot)
            .ok()
            .and_then(|s| self.hist_pool[s].as_ref())
            .ok_or_else(|| unsupported("on a leaf without a histogram buffer"))?;
        if !hist.fresh[inner] && self.data.num_feature_groups() < self.data.num_features() {
            // upstream builds bundled features group-wise, overwriting
            // left-over histograms of features this port does not touch
            return Err(unsupported(
                "on a histogram left over from an earlier build when features are bundled",
            ));
        }
        let bin = self.data.feature_bin_mapper(inner).value_to_bin(number_value(get(node, "threshold")));
        let v = self.slots.views[inner];
        let mut split = SplitInfo::default();
        gather_info_for_threshold(
            &hist.data[v.start..v.start + v.len],
            &self.metas[inner],
            &self.params,
            ls.sum_gradients,
            ls.sum_hessians,
            bin,
            ls.num_data,
            ls.weight,
            &mut split,
        );
        split.feature = feature;
        Ok(split)
    }

    /// upstream `SerialTreeLearner::ForceSplits`: the number of splits made,
    /// or `num_leaves` when training of the tree should stop.
    pub(super) fn force_splits(
        &mut self,
        tree: &mut Tree,
        grad: &[f32],
        hess: &[f32],
        left_leaf: &mut i32,
        right_leaf: &mut i32,
    ) -> Result<usize> {
        let Some(root) = self.forced_split.clone() else { return Ok(0) };
        let force = self.force_features(&root);
        let mut result = 0;
        *left_leaf = 0;
        let mut q: VecDeque<(&Value, i32)> = VecDeque::from([(&*root, 0)]);
        // the children of the last split, whose splits are gathered next
        let mut left: &Value = &root;
        let mut right: &Value = &NULL;
        let mut left_smaller = true;
        let mut force_split_map: HashMap<i32, SplitInfo> = HashMap::new();
        let mut abort = false;
        while let Some(&(node, current)) = q.front() {
            if self.before_find_best_split(tree, *left_leaf, *right_leaf) {
                self.find_best_splits(tree, grad, hess, *left_leaf, *right_leaf, Some(&force))?;
            }
            for (child, leaf, use_smaller) in [(left, *left_leaf, left_smaller), (right, *right_leaf, !left_smaller)] {
                if child.is_null() {
                    continue;
                }
                let split = self.gather_forced(child, use_smaller)?;
                if split.gain < 0.0 {
                    force_split_map.remove(&leaf);
                } else {
                    force_split_map.insert(leaf, split);
                }
            }
            q.pop_front();
            // bfs order: the split was gathered when its parent was split
            let Some(split) = force_split_map.get(&current) else {
                abort = true;
                break;
            };
            if tree.num_leaves >= self.num_leaves {
                return Err(unsupported("with more splits than num_leaves - 1 (upstream writes past the tree)"));
            }
            self.best_split_per_leaf[current as usize] = split.clone();
            let (l, r) = self.split(tree, current as usize)?;
            (*left_leaf, *right_leaf) = (l, r);
            left_smaller = self.smaller.leaf == l;
            left = &NULL;
            right = &NULL;
            for (key, child, leaf) in [("left", &mut left, l), ("right", &mut right, r)] {
                if has(node, key) {
                    *child = get(node, key);
                    if has(child, "feature") && has(child, "threshold") {
                        q.push_back((*child, leaf));
                    }
                }
            }
            result += 1;
        }
        if abort {
            let best_leaf = self.best_leaf();
            if self.best_split_per_leaf[best_leaf].gain <= 0.0 {
                return Ok(self.num_leaves);
            }
            let (l, r) = self.split(tree, best_leaf)?;
            (*left_leaf, *right_leaf) = (l, r);
            result += 1;
        }
        Ok(result)
    }
}
