//! Threshold search on one feature histogram.
//!
//! upstream: src/treelearner/feature_histogram.hpp
//! (`FindBestThresholdSequentially`, `GetSplitGains`, `GetLeafGain`,
//! `CalculateSplittedLeafOutput`, `FuncForNumricalL3`) and
//! feature_histogram.cpp (`FindBestThresholdCategoricalInner`). Quantized
//! gradients and feature penalties are not implemented, so their template
//! branches are omitted. `USE_MC` is a `Some` feature constraint.

use crate::binning::{BinType, MissingType};
use crate::consts::{K_EPSILON, K_MIN_SCORE};
use crate::learner::constraints::{BasicConstraint, FeatureConstraint, ScanConstraint};
use crate::random::Random;

/// Per-feature histogram metadata (upstream `FeatureMetainfo`).
#[derive(Debug, Clone, Copy)]
pub struct FeatureMeta {
    pub num_bin: i32,
    pub missing_type: MissingType,
    /// 1 if bin 0 is the most frequent bin (and therefore not stored).
    pub offset: i32,
    pub default_bin: u32,
    pub most_freq_bin: u32,
    pub bin_type: BinType,
    pub monotone_type: i8,
}

/// Regularization and stopping parameters used during split search.
#[derive(Debug, Clone, Copy)]
pub struct SplitParams {
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub max_delta_step: f64,
    pub path_smooth: f64,
    pub min_data_in_leaf: i32,
    pub min_sum_hessian_in_leaf: f64,
    pub min_gain_to_split: f64,
    pub max_cat_to_onehot: i32,
    pub max_cat_threshold: i32,
    pub cat_l2: f64,
    pub cat_smooth: f64,
    pub min_data_per_group: i32,
}

impl SplitParams {
    fn use_l1(&self) -> bool {
        self.lambda_l1 > 0.0
    }
    fn use_max_output(&self) -> bool {
        self.max_delta_step > 0.0
    }
    fn use_smoothing(&self) -> bool {
        self.path_smooth > K_EPSILON
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SplitInfo {
    /// Real (column) feature index; -1 if none.
    pub feature: i32,
    pub threshold: u32,
    /// Bins sent left by a categorical split (empty for numerical splits).
    pub cat_threshold: Vec<u32>,
    pub left_output: f64,
    pub right_output: f64,
    pub gain: f64,
    pub left_count: i32,
    pub right_count: i32,
    pub left_sum_gradient: f64,
    pub left_sum_hessian: f64,
    pub right_sum_gradient: f64,
    pub right_sum_hessian: f64,
    pub default_left: bool,
    pub monotone_type: i8,
}

impl Default for SplitInfo {
    fn default() -> Self {
        Self {
            feature: -1,
            threshold: 0,
            cat_threshold: Vec::new(),
            left_output: 0.0,
            right_output: 0.0,
            gain: K_MIN_SCORE,
            left_count: 0,
            right_count: 0,
            left_sum_gradient: 0.0,
            left_sum_hessian: 0.0,
            right_sum_gradient: 0.0,
            right_sum_hessian: 0.0,
            default_left: true,
            monotone_type: 0,
        }
    }
}

impl SplitInfo {
    pub fn reset(&mut self) {
        self.feature = -1;
        self.gain = K_MIN_SCORE;
    }

    /// upstream `SplitInfo::operator>`: higher gain, then smaller feature index.
    #[inline]
    pub fn better_than(&self, other: &SplitInfo) -> bool {
        if self.gain != other.gain {
            return self.gain > other.gain;
        }
        let a = if self.feature == -1 { i32::MAX } else { self.feature };
        let b = if other.feature == -1 { i32::MAX } else { other.feature };
        a < b
    }
}

#[inline]
fn sign(x: f64) -> f64 {
    ((x > 0.0) as i32 - (x < 0.0) as i32) as f64
}

#[inline]
pub fn threshold_l1(s: f64, l1: f64) -> f64 {
    let reg = (s.abs() - l1).max(0.0);
    sign(s) * reg
}

/// upstream `CalculateSplittedLeafOutput<USE_L1, USE_MAX_OUTPUT, USE_SMOOTHING>`.
#[inline]
fn leaf_output_t(
    g: f64,
    h: f64,
    p: &SplitParams,
    use_l1: bool,
    use_max: bool,
    use_smooth: bool,
    num_data: i32,
    parent_output: f64,
) -> f64 {
    let mut ret = if use_l1 {
        -threshold_l1(g, p.lambda_l1) / (h + p.lambda_l2)
    } else {
        -g / (h + p.lambda_l2)
    };
    if use_max && p.max_delta_step > 0.0 && ret.abs() > p.max_delta_step {
        ret = sign(ret) * p.max_delta_step;
    }
    if use_smooth {
        let n = num_data as f64;
        ret = ret * (n / p.path_smooth) / (n / p.path_smooth + 1.0)
            + parent_output / (n / p.path_smooth + 1.0);
    }
    ret
}

/// Leaf output with the configured regularization (used for split children).
#[inline]
pub fn leaf_output(g: f64, h: f64, p: &SplitParams, num_data: i32, parent_output: f64) -> f64 {
    leaf_output_t(g, h, p, p.use_l1(), p.use_max_output(), p.use_smoothing(), num_data, parent_output)
}

/// Refit output: upstream `FitByExistingTree` calls
/// `CalculateSplittedLeafOutput<true, true, USE_SMOOTHING>`.
#[inline]
pub fn refit_leaf_output(g: f64, h: f64, p: &SplitParams, use_smooth: bool, num_data: i32, parent_output: f64) -> f64 {
    leaf_output_t(g, h, p, true, true, use_smooth, num_data, parent_output)
}

/// Root output: upstream calls `CalculateSplittedLeafOutput<true, true, true, false>`.
#[inline]
pub fn root_output(g: f64, h: f64, p: &SplitParams, num_data: i32) -> f64 {
    leaf_output_t(g, h, p, true, true, false, num_data, 0.0)
}

#[inline]
fn leaf_gain_given_output(g: f64, h: f64, p: &SplitParams, output: f64) -> f64 {
    let sg = if p.use_l1() { threshold_l1(g, p.lambda_l1) } else { g };
    -(2.0 * sg * output + (h + p.lambda_l2) * output * output)
}

#[inline]
fn leaf_gain(g: f64, h: f64, p: &SplitParams, num_data: i32, parent_output: f64) -> f64 {
    if !p.use_max_output() && !p.use_smoothing() {
        if p.use_l1() {
            let sg = threshold_l1(g, p.lambda_l1);
            (sg * sg) / (h + p.lambda_l2)
        } else {
            (g * g) / (h + p.lambda_l2)
        }
    } else {
        let out = leaf_output(g, h, p, num_data, parent_output);
        leaf_gain_given_output(g, h, p, out)
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn split_gain(
    lg: f64,
    lh: f64,
    rg: f64,
    rh: f64,
    p: &SplitParams,
    lc: i32,
    rc: i32,
    parent_output: f64,
    mc: Option<&ScanConstraint>,
    monotone_type: i8,
) -> f64 {
    let Some(c) = mc else {
        return leaf_gain(lg, lh, p, lc, parent_output) + leaf_gain(rg, rh, p, rc, parent_output);
    };
    let left = c.left().clamp(leaf_output(lg, lh, p, lc, parent_output));
    let right = c.right().clamp(leaf_output(rg, rh, p, rc, parent_output));
    if (monotone_type > 0 && left > right) || (monotone_type < 0 && left < right) {
        return 0.0;
    }
    leaf_gain_given_output(lg, lh, p, left) + leaf_gain_given_output(rg, rh, p, right)
}

#[inline]
fn round_int(x: f64) -> i32 {
    (x + 0.5f32 as f64) as i32
}

/// Find the best threshold for one feature.
///
/// `hist` is interleaved `[g0, h0, g1, h1, ...]` with `num_bin - offset`
/// entries; `sum_hessian` is the leaf's raw Hessian sum. Returns whether
/// any candidate beat the no-split gain (upstream `is_splittable_`).
///
/// `extra_rand` is the feature's extra-trees generator (upstream
/// `FeatureMetainfo::rand`, `USE_RAND`): one threshold is drawn per call and
/// only that candidate is evaluated.
///
/// `mc` is the leaf's constraint on this feature when any monotone
/// constraint is configured (upstream `USE_MC`).
#[allow(clippy::too_many_arguments)]
pub fn find_best_threshold(
    hist: &[f64],
    meta: &FeatureMeta,
    p: &SplitParams,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_output: f64,
    mc: Option<FeatureConstraint<'_>>,
    extra_rand: Option<&mut Random>,
    out: &mut SplitInfo,
) -> bool {
    out.default_left = true;
    out.gain = K_MIN_SCORE;
    let sum_hessian = sum_hessian + 2.0 * K_EPSILON;
    if meta.bin_type == BinType::Categorical {
        return find_best_threshold_categorical(
            hist, meta, p, sum_gradient, sum_hessian, num_data, parent_output, mc, extra_rand, out,
        );
    }
    // upstream: BeforeNumerical
    out.monotone_type = meta.monotone_type;
    let min_gain_shift =
        leaf_gain(sum_gradient, sum_hessian, p, num_data, parent_output) + p.min_gain_to_split;
    let rand_threshold = extra_rand.map(|r| if meta.num_bin - 2 > 0 { r.next_int(0, meta.num_bin - 2) } else { 0 });
    let mut splittable = false;
    let mut run = |reverse: bool, skip_default: bool, na_as_missing: bool, out: &mut SplitInfo| {
        splittable |= scan(
            hist, meta, p, sum_gradient, sum_hessian, num_data, min_gain_shift, parent_output,
            reverse, skip_default, na_as_missing, rand_threshold, mc, out,
        );
    };
    if meta.num_bin > 2 && meta.missing_type != MissingType::None {
        if meta.missing_type == MissingType::Zero {
            run(true, true, false, out);
            run(false, true, false, out);
        } else {
            run(true, false, true, out);
            run(false, false, true, out);
        }
    } else {
        run(true, false, false, out);
        if meta.missing_type == MissingType::NaN {
            out.default_left = false;
        }
    }
    splittable
}

/// upstream `FindBestThresholdCategoricalInner` (no MC). One-vs-rest for at
/// most `max_cat_to_onehot` bins; otherwise bins with enough data are sorted
/// by `sum_grad / (sum_hess + cat_smooth)` and prefixes are scanned from both
/// ends with `lambda_l2 + cat_l2`. `sum_hessian` already includes `2 * kEpsilon`.
#[allow(clippy::too_many_arguments)]
fn find_best_threshold_categorical(
    hist: &[f64],
    meta: &FeatureMeta,
    p: &SplitParams,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    parent_output: f64,
    mc: Option<FeatureConstraint<'_>>,
    extra_rand: Option<&mut Random>,
    out: &mut SplitInfo,
) -> bool {
    let g = |t: i32| hist[2 * t as usize];
    let h = |t: i32| hist[2 * t as usize + 1];
    let mut is_splittable = false;
    out.default_left = false;
    let mc = mc.map(|c| c.init_cumulative(true));
    let mc = mc.as_ref();
    let mut best_gain = K_MIN_SCORE;
    let mut best_left_count: i32 = 0;
    let mut best_sum_left_gradient = 0.0f64;
    let mut best_sum_left_hessian = 0.0f64;
    // Without smoothing the shift uses the plain lambda_l2, the children the
    // categorical one (upstream keeps this for backwards compatibility).
    let gain_shift = if p.use_smoothing() {
        leaf_gain_given_output(sum_gradient, sum_hessian, p, parent_output)
    } else {
        leaf_gain(sum_gradient, sum_hessian, p, num_data, 0.0)
    };
    let min_gain_shift = gain_shift + p.min_gain_to_split;
    let offset = meta.offset;
    let bin_start = 1 - offset;
    let bin_end = meta.num_bin - offset;
    let mut used_bin: i32 = -1;
    let mut sorted_idx: Vec<i32> = Vec::new();
    let mut pc = *p;
    let use_onehot = meta.num_bin <= p.max_cat_to_onehot;
    let mut best_threshold: i32 = -1;
    let mut best_dir: i32 = 1;
    let cnt_factor = num_data as f64 / sum_hessian;
    let mut rand_threshold = 0;
    let mut extra_rand = extra_rand;
    if use_onehot {
        if let Some(r) = extra_rand.as_deref_mut() {
            if bin_end - bin_start > 0 {
                rand_threshold = r.next_int(bin_start, bin_end);
            }
        }
        for t in bin_start..bin_end {
            let grad = g(t);
            let hess = h(t);
            let cnt = round_int(hess * cnt_factor);
            if cnt < p.min_data_in_leaf || hess < p.min_sum_hessian_in_leaf {
                continue;
            }
            let other_count = num_data - cnt;
            if other_count < p.min_data_in_leaf {
                continue;
            }
            let sum_other_hessian = sum_hessian - hess - K_EPSILON;
            if sum_other_hessian < p.min_sum_hessian_in_leaf {
                continue;
            }
            let sum_other_gradient = sum_gradient - grad;
            if extra_rand.is_some() && t != rand_threshold {
                continue;
            }
            let current_gain = split_gain(
                sum_other_gradient,
                sum_other_hessian,
                grad,
                hess + K_EPSILON,
                &pc,
                other_count,
                cnt,
                parent_output,
                mc,
                0,
            );
            if current_gain <= min_gain_shift {
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                best_threshold = t;
                best_sum_left_gradient = grad;
                best_sum_left_hessian = hess + K_EPSILON;
                best_left_count = cnt;
                best_gain = current_gain;
            }
        }
    } else {
        for i in bin_start..bin_end {
            if round_int(h(i) * cnt_factor) as f64 >= p.cat_smooth {
                sorted_idx.push(i);
            }
        }
        used_bin = sorted_idx.len() as i32;
        pc.lambda_l2 += p.cat_l2;
        let ctr = |i: i32| g(i) / (h(i) + p.cat_smooth);
        // std::stable_sort with `ctr(i) < ctr(j)`
        sorted_idx.sort_by(|&i, &j| {
            let (a, b) = (ctr(i), ctr(j));
            if a < b {
                std::cmp::Ordering::Less
            } else if b < a {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        let max_num_cat = p.max_cat_threshold.min((used_bin + 1) / 2);
        let max_threshold = (max_num_cat.min(used_bin) - 1).max(0);
        if let Some(r) = extra_rand.as_deref_mut() {
            if max_threshold > 0 {
                rand_threshold = r.next_int(0, max_threshold);
            }
        }
        is_splittable = false;
        for (dir, start) in [(1i32, 0i32), (-1, used_bin - 1)] {
            let mut start_pos = start;
            let mut cnt_cur_group: i32 = 0;
            let mut sum_left_gradient = 0.0f64;
            let mut sum_left_hessian = K_EPSILON;
            let mut left_count: i32 = 0;
            let mut i = 0;
            while i < used_bin && i < max_num_cat {
                let t = sorted_idx[start_pos as usize];
                start_pos += dir;
                let grad = g(t);
                let hess = h(t);
                let cnt = round_int(hess * cnt_factor);
                sum_left_gradient += grad;
                sum_left_hessian += hess;
                left_count += cnt;
                cnt_cur_group += cnt;
                if left_count < p.min_data_in_leaf || sum_left_hessian < p.min_sum_hessian_in_leaf {
                    i += 1;
                    continue;
                }
                let right_count = num_data - left_count;
                if right_count < p.min_data_in_leaf || right_count < p.min_data_per_group {
                    break;
                }
                let sum_right_hessian = sum_hessian - sum_left_hessian;
                if sum_right_hessian < p.min_sum_hessian_in_leaf {
                    break;
                }
                if cnt_cur_group < p.min_data_per_group {
                    i += 1;
                    continue;
                }
                cnt_cur_group = 0;
                let sum_right_gradient = sum_gradient - sum_left_gradient;
                if extra_rand.is_some() && i != rand_threshold {
                    i += 1;
                    continue;
                }
                let current_gain = split_gain(
                    sum_left_gradient,
                    sum_left_hessian,
                    sum_right_gradient,
                    sum_right_hessian,
                    &pc,
                    left_count,
                    right_count,
                    parent_output,
                    mc,
                    0,
                );
                if current_gain <= min_gain_shift {
                    i += 1;
                    continue;
                }
                is_splittable = true;
                if current_gain > best_gain {
                    best_left_count = left_count;
                    best_sum_left_gradient = sum_left_gradient;
                    best_sum_left_hessian = sum_left_hessian;
                    best_threshold = i;
                    best_gain = current_gain;
                    best_dir = dir;
                }
                i += 1;
            }
        }
    }

    if is_splittable {
        let (lc, rc) = mc.map_or((BasicConstraint::default(), BasicConstraint::default()), |c| (c.left(), c.right()));
        let clamp = |v: f64, c: BasicConstraint| if mc.is_some() { c.clamp(v) } else { v };
        out.left_output = clamp(
            leaf_output(best_sum_left_gradient, best_sum_left_hessian, &pc, best_left_count, parent_output),
            lc,
        );
        out.left_count = best_left_count;
        out.left_sum_gradient = best_sum_left_gradient;
        out.left_sum_hessian = best_sum_left_hessian - K_EPSILON;
        out.right_output = clamp(
            leaf_output(
                sum_gradient - best_sum_left_gradient,
                sum_hessian - best_sum_left_hessian,
                &pc,
                num_data - best_left_count,
                parent_output,
            ),
            rc,
        );
        out.monotone_type = 0;
        out.right_count = num_data - best_left_count;
        out.right_sum_gradient = sum_gradient - best_sum_left_gradient;
        out.right_sum_hessian = sum_hessian - best_sum_left_hessian - K_EPSILON;
        out.gain = best_gain - min_gain_shift;
        out.cat_threshold = if use_onehot {
            vec![(best_threshold + offset) as u32]
        } else if best_dir == 1 {
            sorted_idx[..=best_threshold as usize].iter().map(|&t| (t + offset) as u32).collect()
        } else {
            (0..=best_threshold as usize).map(|i| (sorted_idx[used_bin as usize - 1 - i] + offset) as u32).collect()
        };
    }
    is_splittable
}

/// upstream `FindBestThresholdSequentially` (`rand_threshold` is `Some`
/// under `USE_RAND`). The cumulative constraint only moves in the reverse
/// scan, as upstream.
#[allow(clippy::too_many_arguments)]
fn scan(
    hist: &[f64],
    meta: &FeatureMeta,
    p: &SplitParams,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    min_gain_shift: f64,
    parent_output: f64,
    reverse: bool,
    skip_default_bin: bool,
    na_as_missing: bool,
    rand_threshold: Option<i32>,
    mc: Option<FeatureConstraint<'_>>,
    out: &mut SplitInfo,
) -> bool {
    let offset = meta.offset;
    let g = |t: i32| hist[2 * t as usize];
    let h = |t: i32| hist[2 * t as usize + 1];
    let mut is_splittable = false;
    let mut best_sum_left_gradient = f64::NAN;
    let mut best_sum_left_hessian = f64::NAN;
    let mut best_gain = K_MIN_SCORE;
    let mut best_left_count: i32 = 0;
    let mut best_threshold: u32 = meta.num_bin as u32;
    let cnt_factor = num_data as f64 / sum_hessian;
    let min_data = p.min_data_in_leaf;
    let min_hess = p.min_sum_hessian_in_leaf;
    let update_needed = mc.is_some_and(|c| c.different_depending_on_threshold());
    let mut cur = mc.map(|c| c.init_cumulative(reverse));
    let mut best_left_c = BasicConstraint::default();
    let mut best_right_c = BasicConstraint::default();
    let mono = meta.monotone_type;

    if reverse {
        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = K_EPSILON;
        let mut right_count: i32 = 0;
        let mut t = meta.num_bin - 1 - offset - na_as_missing as i32;
        let t_end = 1 - offset;
        while t >= t_end {
            if skip_default_bin && (t + offset) == meta.default_bin as i32 {
                t -= 1;
                continue;
            }
            let gr = g(t);
            let hs = h(t);
            let cnt = round_int(hs * cnt_factor);
            sum_right_gradient += gr;
            sum_right_hessian += hs;
            right_count += cnt;
            if right_count < min_data || sum_right_hessian < min_hess {
                t -= 1;
                continue;
            }
            let left_count = num_data - right_count;
            if left_count < min_data {
                break;
            }
            let sum_left_hessian = sum_hessian - sum_right_hessian;
            if sum_left_hessian < min_hess {
                break;
            }
            let sum_left_gradient = sum_gradient - sum_right_gradient;
            if rand_threshold.is_some_and(|r| t - 1 + offset != r) {
                t -= 1;
                continue;
            }
            if update_needed {
                cur.as_mut().expect("constraint present").update(t + offset);
            }
            let current_gain = split_gain(
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                p,
                left_count,
                right_count,
                parent_output,
                cur.as_ref(),
                mono,
            );
            if current_gain <= min_gain_shift {
                t -= 1;
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                if let Some(c) = cur.as_ref() {
                    best_right_c = c.right();
                    best_left_c = c.left();
                    if best_right_c.min > best_right_c.max || best_left_c.min > best_left_c.max {
                        t -= 1;
                        continue;
                    }
                }
                best_left_count = left_count;
                best_sum_left_gradient = sum_left_gradient;
                best_sum_left_hessian = sum_left_hessian;
                best_threshold = (t - 1 + offset) as u32;
                best_gain = current_gain;
            }
            t -= 1;
        }
    } else {
        let mut sum_left_gradient = 0.0f64;
        let mut sum_left_hessian = K_EPSILON;
        let mut left_count: i32 = 0;
        let mut t: i32 = 0;
        let t_end = meta.num_bin - 2 - offset;
        if na_as_missing && offset == 1 {
            sum_left_gradient = sum_gradient;
            sum_left_hessian = sum_hessian - K_EPSILON;
            left_count = num_data;
            for i in 0..(meta.num_bin - offset) {
                let gr = g(i);
                let hs = h(i);
                let cnt = round_int(hs * cnt_factor);
                sum_left_gradient -= gr;
                sum_left_hessian -= hs;
                left_count -= cnt;
            }
            t = -1;
        }
        while t <= t_end {
            if skip_default_bin && (t + offset) == meta.default_bin as i32 {
                t += 1;
                continue;
            }
            if t >= 0 {
                sum_left_gradient += g(t);
                sum_left_hessian += h(t);
                left_count += round_int(h(t) * cnt_factor);
            }
            if left_count < min_data || sum_left_hessian < min_hess {
                t += 1;
                continue;
            }
            let right_count = num_data - left_count;
            if right_count < min_data {
                break;
            }
            let sum_right_hessian = sum_hessian - sum_left_hessian;
            if sum_right_hessian < min_hess {
                break;
            }
            let sum_right_gradient = sum_gradient - sum_left_gradient;
            if rand_threshold.is_some_and(|r| t + offset != r) {
                t += 1;
                continue;
            }
            let current_gain = split_gain(
                sum_left_gradient,
                sum_left_hessian,
                sum_right_gradient,
                sum_right_hessian,
                p,
                left_count,
                right_count,
                parent_output,
                cur.as_ref(),
                mono,
            );
            if current_gain <= min_gain_shift {
                t += 1;
                continue;
            }
            is_splittable = true;
            if current_gain > best_gain {
                if let Some(c) = cur.as_ref() {
                    best_right_c = c.right();
                    best_left_c = c.left();
                    if best_right_c.min > best_right_c.max || best_left_c.min > best_left_c.max {
                        t += 1;
                        continue;
                    }
                }
                best_left_count = left_count;
                best_sum_left_gradient = sum_left_gradient;
                best_sum_left_hessian = sum_left_hessian;
                best_threshold = (t + offset) as u32;
                best_gain = current_gain;
            }
            t += 1;
        }
    }

    if is_splittable && best_gain > out.gain + min_gain_shift {
        let clamp = |v: f64, c: BasicConstraint| if mc.is_some() { c.clamp(v) } else { v };
        out.threshold = best_threshold;
        out.left_output = clamp(
            leaf_output(best_sum_left_gradient, best_sum_left_hessian, p, best_left_count, parent_output),
            best_left_c,
        );
        out.left_count = best_left_count;
        out.left_sum_gradient = best_sum_left_gradient;
        out.left_sum_hessian = best_sum_left_hessian - K_EPSILON;
        out.right_output = clamp(
            leaf_output(
                sum_gradient - best_sum_left_gradient,
                sum_hessian - best_sum_left_hessian,
                p,
                num_data - best_left_count,
                parent_output,
            ),
            best_right_c,
        );
        out.right_count = num_data - best_left_count;
        out.right_sum_gradient = sum_gradient - best_sum_left_gradient;
        out.right_sum_hessian = sum_hessian - best_sum_left_hessian - K_EPSILON;
        out.gain = best_gain - min_gain_shift;
        out.default_left = reverse;
    }
    is_splittable
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SplitParams {
        SplitParams {
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            max_delta_step: 0.0,
            path_smooth: 0.0,
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 1e-3,
            min_gain_to_split: 0.0,
            max_cat_to_onehot: 4,
            max_cat_threshold: 32,
            cat_l2: 10.0,
            cat_smooth: 10.0,
            min_data_per_group: 100,
        }
    }

    #[test]
    fn finds_obvious_split() {
        // 4 bins, offset 0; grads strongly negative in bins 0-1, positive in 2-3.
        let hist = [-5.0, 5.0, -5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let meta = FeatureMeta {
            num_bin: 4,
            missing_type: MissingType::None,
            offset: 0,
            default_bin: 0,
            most_freq_bin: 0,
            bin_type: BinType::Numerical,
            monotone_type: 0,
        };
        let mut out = SplitInfo::default();
        let ok = find_best_threshold(&hist, &meta, &params(), 0.0, 20.0, 20, 0.0, None, None, &mut out);
        assert!(ok);
        assert_eq!(out.threshold, 1);
        assert_eq!(out.left_count, 10);
        assert!((out.left_output - 1.0).abs() < 1e-12);
        assert!((out.right_output + 1.0).abs() < 1e-12);
        // gain = 100/10 + 100/10 - 0
        assert!((out.gain - 20.0).abs() < 1e-9);
    }

    #[test]
    fn categorical_one_hot_and_sorted() {
        let meta = FeatureMeta {
            num_bin: 3,
            missing_type: MissingType::NaN,
            offset: 0,
            default_bin: 1,
            most_freq_bin: 1,
            bin_type: BinType::Categorical,
            monotone_type: 0,
        };
        let hist = [0.0, 10.0, -10.0, 10.0, 10.0, 10.0];
        let mut out = SplitInfo::default();
        assert!(find_best_threshold(&hist, &meta, &params(), 0.0, 30.0, 30, 0.0, None, None, &mut out));
        assert_eq!(out.cat_threshold, vec![1]);
        assert!(!out.default_left);
        assert_eq!(out.left_count, 10);

        // many-vs-many: 6 bins, sorted by ctr; the two most negative go left
        let meta = FeatureMeta { num_bin: 6, offset: 1, most_freq_bin: 0, default_bin: 0, ..meta };
        let hist = [5.0, 100.0, -50.0, 100.0, 2.0, 100.0, -40.0, 100.0, 3.0, 100.0];
        let p = SplitParams { min_data_per_group: 1, cat_smooth: 1.0, ..params() };
        let mut out = SplitInfo::default();
        assert!(find_best_threshold(&hist, &meta, &p, -80.0, 600.0, 600, 0.0, None, None, &mut out));
        assert_eq!(out.cat_threshold, vec![2, 4]);
    }

    #[test]
    fn tie_break_prefers_smaller_feature() {
        let a = SplitInfo { feature: 3, gain: 1.0, ..Default::default() };
        let b = SplitInfo { feature: 1, gain: 1.0, ..Default::default() };
        assert!(b.better_than(&a));
        assert!(!a.better_than(&b));
        let none = SplitInfo::default();
        assert!(a.better_than(&none));
    }

    #[test]
    fn l1_and_max_delta_step() {
        let p = SplitParams { lambda_l1: 1.0, max_delta_step: 0.5, ..params() };
        assert_eq!(threshold_l1(3.0, 1.0), 2.0);
        assert_eq!(threshold_l1(-0.5, 1.0), 0.0);
        let o = leaf_output(-10.0, 2.0, &p, 10, 0.0);
        assert_eq!(o, 0.5);
    }
}
