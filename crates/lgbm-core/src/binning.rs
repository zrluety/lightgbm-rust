//! Feature discretization (bin mappers).
//!
//! upstream: src/io/bin.cpp (`GreedyFindBin`, `FindBinWithZeroAsOneBin`,
//! `BinMapper::FindBin`, `NeedFilter`) and include/LightGBM/bin.h
//! (`BinMapper::ValueToBin`). Ported line-by-line, including integer/float
//! promotion rules, so bin boundaries and category bins match upstream
//! bit-for-bit.

use std::collections::HashMap;

use crate::consts::{K_SPARSE_THRESHOLD, K_ZERO_THRESHOLD};
use crate::error::Result;
use crate::fmt::fmt_g17;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingType {
    None = 0,
    Zero = 1,
    NaN = 2,
}

impl MissingType {
    pub fn from_i8(v: i8) -> Self {
        match v {
            1 => MissingType::Zero,
            2 => MissingType::NaN,
            _ => MissingType::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinType {
    Numerical,
    Categorical,
}

/// Parameters controlling bin construction for one feature.
#[derive(Debug, Clone, Copy)]
pub struct BinParams {
    pub max_bin: i32,
    pub min_data_in_bin: i32,
    pub min_split_data: i32,
    pub pre_filter: bool,
    pub use_missing: bool,
    pub zero_as_missing: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinMapper {
    pub num_bin: i32,
    pub missing_type: MissingType,
    pub bin_upper_bound: Vec<f64>,
    pub is_trivial: bool,
    pub sparse_rate: f64,
    pub bin_type: BinType,
    pub min_val: f64,
    pub max_val: f64,
    pub default_bin: u32,
    pub most_freq_bin: u32,
    /// Category of each bin (bin 0 is the `-1` "other/NaN" bin); empty for
    /// numerical features.
    pub bin_2_categorical: Vec<i32>,
    /// Inverse of `bin_2_categorical`.
    pub categorical_2_bin: HashMap<i32, u32>,
}

impl Default for BinMapper {
    fn default() -> Self {
        Self {
            num_bin: 1,
            missing_type: MissingType::None,
            bin_upper_bound: vec![f64::INFINITY],
            is_trivial: true,
            sparse_rate: 1.0,
            bin_type: BinType::Numerical,
            min_val: 0.0,
            max_val: 0.0,
            default_bin: 0,
            most_freq_bin: 0,
            bin_2_categorical: Vec::new(),
            categorical_2_bin: HashMap::new(),
        }
    }
}

#[inline]
fn get_double_upper_bound(a: f64) -> f64 {
    a.next_up()
}

#[inline]
fn check_double_equal_ordered(a: f64, b: f64) -> bool {
    b <= a.next_up()
}

/// upstream: src/io/bin.cpp `NeedFilter`.
fn need_filter(cnt_in_bin: &[i32], total_cnt: i32, filter_cnt: i32, bin_type: BinType) -> bool {
    if bin_type == BinType::Numerical {
        let mut sum_left = 0;
        for &c in &cnt_in_bin[..cnt_in_bin.len() - 1] {
            sum_left += c;
            if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                return false;
            }
        }
    } else {
        if cnt_in_bin.len() > 2 {
            return false;
        }
        for &sum_left in &cnt_in_bin[..cnt_in_bin.len() - 1] {
            if sum_left >= filter_cnt && total_cnt - sum_left >= filter_cnt {
                return false;
            }
        }
    }
    true
}

/// upstream: src/io/bin.cpp `GreedyFindBin`.
pub fn greedy_find_bin(
    distinct_values: &[f64],
    counts: &[i32],
    max_bin: i32,
    total_cnt: usize,
    min_data_in_bin: i32,
) -> Vec<f64> {
    let num_distinct_values = distinct_values.len() as i32;
    let mut bin_upper_bound = Vec::new();
    assert!(max_bin > 0);
    if num_distinct_values == 0 {
        // reached from FindBinWithPredefinedBin for an empty forced interval
        return vec![f64::INFINITY];
    }
    if num_distinct_values <= max_bin {
        let mut cur_cnt_inbin = 0;
        for i in 0..(num_distinct_values - 1) as usize {
            cur_cnt_inbin += counts[i];
            if cur_cnt_inbin >= min_data_in_bin {
                let val = get_double_upper_bound((distinct_values[i] + distinct_values[i + 1]) / 2.0);
                if bin_upper_bound
                    .last()
                    .is_none_or(|&last| !check_double_equal_ordered(last, val))
                {
                    bin_upper_bound.push(val);
                    cur_cnt_inbin = 0;
                }
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    } else {
        let mut max_bin = max_bin;
        if min_data_in_bin > 0 {
            max_bin = max_bin.min((total_cnt / min_data_in_bin as usize) as i32);
            max_bin = max_bin.max(1);
        }
        let mut mean_bin_size = total_cnt as f64 / max_bin as f64;
        let mut rest_bin_cnt = max_bin;
        let mut rest_sample_cnt = total_cnt as i32;
        let n = num_distinct_values as usize;
        let mut is_big_count_value = vec![false; n];
        for i in 0..n {
            if counts[i] as f64 >= mean_bin_size {
                is_big_count_value[i] = true;
                rest_bin_cnt -= 1;
                rest_sample_cnt -= counts[i];
            }
        }
        mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
        let mut upper_bounds = vec![f64::INFINITY; max_bin as usize];
        let mut lower_bounds = vec![f64::INFINITY; max_bin as usize];

        let mut bin_cnt: i32 = 0;
        lower_bounds[0] = distinct_values[0];
        let mut cur_cnt_inbin = 0;
        for i in 0..n - 1 {
            if !is_big_count_value[i] {
                rest_sample_cnt -= counts[i];
            }
            cur_cnt_inbin += counts[i];
            let half = (mean_bin_size * 0.5f32 as f64).max(1.0);
            if is_big_count_value[i]
                || cur_cnt_inbin as f64 >= mean_bin_size
                || (is_big_count_value[i + 1] && cur_cnt_inbin as f64 >= half)
            {
                upper_bounds[bin_cnt as usize] = distinct_values[i];
                bin_cnt += 1;
                lower_bounds[bin_cnt as usize] = distinct_values[i + 1];
                if bin_cnt >= max_bin - 1 {
                    break;
                }
                cur_cnt_inbin = 0;
                if !is_big_count_value[i] {
                    rest_bin_cnt -= 1;
                    mean_bin_size = rest_sample_cnt as f64 / rest_bin_cnt as f64;
                }
            }
        }
        bin_cnt += 1;
        for i in 0..(bin_cnt - 1) as usize {
            let val = get_double_upper_bound((upper_bounds[i] + lower_bounds[i + 1]) / 2.0);
            if bin_upper_bound
                .last()
                .is_none_or(|&last| !check_double_equal_ordered(last, val))
            {
                bin_upper_bound.push(val);
            }
        }
        bin_upper_bound.push(f64::INFINITY);
    }
    bin_upper_bound
}

/// upstream: src/io/bin.cpp `FindBinWithZeroAsOneBin` (no forced bounds).
pub fn find_bin_with_zero_as_one_bin(
    distinct_values: &[f64],
    counts: &[i32],
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
) -> Vec<f64> {
    let n = distinct_values.len();
    let mut bin_upper_bound: Vec<f64> = Vec::new();
    let mut left_cnt_data = 0i32;
    let mut cnt_zero = 0i32;
    let mut right_cnt_data = 0i32;
    for i in 0..n {
        if distinct_values[i] <= -K_ZERO_THRESHOLD {
            left_cnt_data += counts[i];
        } else if distinct_values[i] > K_ZERO_THRESHOLD {
            right_cnt_data += counts[i];
        } else {
            cnt_zero += counts[i];
        }
    }
    let left_cnt = distinct_values
        .iter()
        .position(|&v| v > -K_ZERO_THRESHOLD)
        .unwrap_or(n);

    if left_cnt > 0 && max_bin > 1 {
        let denom = total_sample_cnt.wrapping_sub(cnt_zero as usize) as f64;
        let mut left_max_bin = (left_cnt_data as f64 / denom * (max_bin - 1) as f64) as i32;
        left_max_bin = left_max_bin.max(1);
        bin_upper_bound = greedy_find_bin(
            &distinct_values[..left_cnt],
            &counts[..left_cnt],
            left_max_bin,
            left_cnt_data as usize,
            min_data_in_bin,
        );
        if let Some(last) = bin_upper_bound.last_mut() {
            *last = -K_ZERO_THRESHOLD;
        }
    }

    let right_start = distinct_values[left_cnt..]
        .iter()
        .position(|&v| v > K_ZERO_THRESHOLD)
        .map(|p| p + left_cnt);

    let right_max_bin = max_bin - 1 - bin_upper_bound.len() as i32;
    match right_start {
        Some(rs) if right_max_bin > 0 => {
            let right_bounds = greedy_find_bin(
                &distinct_values[rs..],
                &counts[rs..],
                right_max_bin,
                right_cnt_data as usize,
                min_data_in_bin,
            );
            bin_upper_bound.push(K_ZERO_THRESHOLD);
            bin_upper_bound.extend(right_bounds);
        }
        _ => bin_upper_bound.push(f64::INFINITY),
    }
    debug_assert!(bin_upper_bound.len() <= max_bin as usize);
    bin_upper_bound
}

/// upstream: src/io/bin.cpp `FindBinWithPredefinedBin`: zero bounds and the
/// forced bounds first, then the free bins shared out over the forced
/// intervals by sample count and filled by `GreedyFindBin`.
pub fn find_bin_with_predefined_bin(
    distinct_values: &[f64],
    counts: &[i32],
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Result<Vec<f64>> {
    let n = distinct_values.len();
    let mut bin_upper_bound: Vec<f64> = Vec::new();
    let left_cnt = distinct_values.iter().position(|&v| v > -K_ZERO_THRESHOLD).unwrap_or(n);
    let right_start = distinct_values[left_cnt..].iter().position(|&v| v > K_ZERO_THRESHOLD).map(|p| p + left_cnt);

    if max_bin == 2 {
        bin_upper_bound.push(if left_cnt == 0 { K_ZERO_THRESHOLD } else { -K_ZERO_THRESHOLD });
    } else if max_bin >= 3 {
        if left_cnt > 0 {
            bin_upper_bound.push(-K_ZERO_THRESHOLD);
        }
        if right_start.is_some() {
            bin_upper_bound.push(K_ZERO_THRESHOLD);
        }
    }
    bin_upper_bound.push(f64::INFINITY);

    // forced bounds, excluding zeros (the zero bounds are already in)
    let max_to_insert = max_bin - bin_upper_bound.len() as i32;
    let mut num_inserted = 0;
    for &b in forced_upper_bounds {
        if num_inserted >= max_to_insert {
            break;
        }
        if b.abs() > K_ZERO_THRESHOLD {
            bin_upper_bound.push(b);
            num_inserted += 1;
        }
    }
    let by_value = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    bin_upper_bound.sort_by(by_value);

    let free_bins = max_bin - bin_upper_bound.len() as i32;
    let mut bounds_to_add: Vec<f64> = Vec::new();
    let mut value_ind = 0usize;
    for i in 0..bin_upper_bound.len() {
        let mut cnt_in_bin = 0i32;
        let bin_start = value_ind;
        while value_ind < n && distinct_values[value_ind] < bin_upper_bound[i] {
            cnt_in_bin += counts[value_ind];
            value_ind += 1;
        }
        let bins_remaining = max_bin - bin_upper_bound.len() as i32 - bounds_to_add.len() as i32;
        // std::lround: half away from zero
        let num_sub_bins = (cnt_in_bin as f64 * free_bins as f64 / total_sample_cnt as f64).round() as i32;
        let mut num_sub_bins = num_sub_bins.min(bins_remaining) + 1;
        if i == bin_upper_bound.len() - 1 {
            num_sub_bins = bins_remaining + 1;
        }
        let new_upper_bounds = greedy_find_bin(
            &distinct_values[bin_start..value_ind],
            &counts[bin_start..value_ind],
            num_sub_bins,
            cnt_in_bin as usize,
            min_data_in_bin,
        );
        // the last bound is +inf
        bounds_to_add.extend_from_slice(&new_upper_bounds[..new_upper_bounds.len() - 1]);
    }
    bin_upper_bound.extend(bounds_to_add);
    bin_upper_bound.sort_by(by_value);
    if bin_upper_bound.len() > max_bin as usize {
        return Err(crate::error::LgbmError::InvalidData(
            "Check failed: (bin_upper_bound.size()) <= (static_cast<size_t>(max_bin))".into(),
        ));
    }
    Ok(bin_upper_bound)
}

/// upstream: the `FindBinWithZeroAsOneBin` overload taking forced bounds.
fn find_bin_with_forced(
    distinct_values: &[f64],
    counts: &[i32],
    max_bin: i32,
    total_sample_cnt: usize,
    min_data_in_bin: i32,
    forced_upper_bounds: &[f64],
) -> Result<Vec<f64>> {
    if forced_upper_bounds.is_empty() {
        Ok(find_bin_with_zero_as_one_bin(distinct_values, counts, max_bin, total_sample_cnt, min_data_in_bin))
    } else {
        find_bin_with_predefined_bin(
            distinct_values, counts, max_bin, total_sample_cnt, min_data_in_bin, forced_upper_bounds,
        )
    }
}

impl BinMapper {
    /// Build a bin mapper from sampled values.
    ///
    /// `values` must contain only the sampled values with
    /// `|v| > kZeroThreshold` or `NaN` (as upstream's dataset constructors
    /// pass them); `total_sample_cnt` counts all sampled rows including zeros.
    ///
    /// upstream: src/io/bin.cpp `BinMapper::FindBin`.
    pub fn find_bin(
        values: &[f64],
        total_sample_cnt: usize,
        bin_type: BinType,
        p: &BinParams,
    ) -> Result<Self> {
        Self::find_bin_logged(values, total_sample_cnt, bin_type, p, &[], &mut Vec::new())
    }

    /// Like [`BinMapper::find_bin`], with upstream's forced upper bounds for
    /// a numerical feature, appending upstream's `Log::Warning`s to `warnings`.
    pub fn find_bin_logged(
        values: &[f64],
        total_sample_cnt: usize,
        bin_type: BinType,
        p: &BinParams,
        forced_upper_bounds: &[f64],
        warnings: &mut Vec<String>,
    ) -> Result<Self> {
        let num_sample_values_in = values.len() as i32;
        let mut vals: Vec<f64> = values.iter().copied().filter(|v| !v.is_nan()).collect();
        let non_na_cnt = vals.len() as i32;
        let mut na_cnt = 0i32;
        let mut missing_type = if !p.use_missing {
            MissingType::None
        } else if p.zero_as_missing {
            MissingType::Zero
        } else if non_na_cnt == num_sample_values_in {
            MissingType::None
        } else {
            na_cnt = num_sample_values_in - non_na_cnt;
            MissingType::NaN
        };
        let num_sample_values = non_na_cnt;
        let zero_cnt = (total_sample_cnt as i64 - num_sample_values as i64 - na_cnt as i64) as i32;

        // stable sort; NaNs were removed so total order is fine
        vals.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));

        let mut distinct_values: Vec<f64> = Vec::new();
        let mut counts: Vec<i32> = Vec::new();
        if num_sample_values == 0 || (vals[0] > 0.0 && zero_cnt > 0) {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }
        if num_sample_values > 0 {
            distinct_values.push(vals[0]);
            counts.push(1);
        }
        for i in 1..num_sample_values as usize {
            if !check_double_equal_ordered(vals[i - 1], vals[i]) {
                if vals[i - 1] < 0.0 && vals[i] > 0.0 {
                    distinct_values.push(0.0);
                    counts.push(zero_cnt);
                }
                distinct_values.push(vals[i]);
                counts.push(1);
            } else {
                *distinct_values.last_mut().unwrap() = vals[i];
                *counts.last_mut().unwrap() += 1;
            }
        }
        if num_sample_values > 0 && vals[num_sample_values as usize - 1] < 0.0 && zero_cnt > 0 {
            distinct_values.push(0.0);
            counts.push(zero_cnt);
        }
        let min_val = distinct_values[0];
        let max_val = *distinct_values.last().unwrap();

        if bin_type == BinType::Categorical {
            return Ok(Self::finish_categorical(
                &distinct_values, &counts, na_cnt, missing_type, min_val, max_val, total_sample_cnt, p, warnings,
            ));
        }

        let forced = forced_upper_bounds;
        let mut bin_upper_bound = match missing_type {
            MissingType::Zero => {
                let b = find_bin_with_forced(
                    &distinct_values, &counts, p.max_bin, total_sample_cnt, p.min_data_in_bin, forced,
                )?;
                if b.len() == 2 {
                    missing_type = MissingType::None;
                }
                b
            }
            MissingType::None => find_bin_with_forced(
                &distinct_values, &counts, p.max_bin, total_sample_cnt, p.min_data_in_bin, forced,
            )?,
            MissingType::NaN => {
                let mut b = find_bin_with_forced(
                    &distinct_values,
                    &counts,
                    p.max_bin - 1,
                    total_sample_cnt - na_cnt as usize,
                    p.min_data_in_bin,
                    forced,
                )?;
                b.push(f64::NAN);
                b
            }
        };
        let num_bin = bin_upper_bound.len() as i32;
        let mut cnt_in_bin = vec![0i32; num_bin as usize];
        let mut i_bin = 0usize;
        for i in 0..distinct_values.len() {
            // NaN upper bound compares false, matching upstream's loop.
            while distinct_values[i] > bin_upper_bound[i_bin] && (i_bin as i32) < num_bin - 1 {
                i_bin += 1;
            }
            cnt_in_bin[i_bin] += counts[i];
        }
        if missing_type == MissingType::NaN {
            cnt_in_bin[num_bin as usize - 1] = na_cnt;
        }
        debug_assert!(num_bin <= p.max_bin);

        let mut m = BinMapper {
            num_bin,
            missing_type,
            bin_upper_bound: std::mem::take(&mut bin_upper_bound),
            is_trivial: true,
            sparse_rate: 1.0,
            bin_type,
            min_val,
            max_val,
            ..Default::default()
        };
        m.finish(&cnt_in_bin, total_sample_cnt, p);
        Ok(m)
    }

    /// upstream: the categorical branch of `BinMapper::FindBin`.
    #[allow(clippy::too_many_arguments)]
    fn finish_categorical(
        distinct_values: &[f64],
        counts: &[i32],
        mut na_cnt: i32,
        missing_type: MissingType,
        min_val: f64,
        max_val: f64,
        total_sample_cnt: usize,
        p: &BinParams,
        warnings: &mut Vec<String>,
    ) -> Self {
        let mut m = BinMapper {
            missing_type,
            bin_type: BinType::Categorical,
            min_val,
            max_val,
            ..Default::default()
        };
        // upstream converts with static_cast<int> (truncation toward zero)
        let mut distinct_values_int: Vec<i32> = Vec::new();
        let mut counts_int: Vec<i32> = Vec::new();
        for (&v, &c) in distinct_values.iter().zip(counts) {
            let val = v as i32;
            if val < 0 {
                na_cnt += c;
                warnings.push("Met negative value in categorical features, will convert it to NaN".into());
            } else if distinct_values_int.last() != Some(&val) {
                distinct_values_int.push(val);
                counts_int.push(c);
            } else {
                *counts_int.last_mut().unwrap() += c;
            }
        }
        let mut cnt_in_bin: Vec<i32> = Vec::new();
        let rest_cnt = (total_sample_cnt as i64 - na_cnt as i64) as i32;
        if rest_cnt > 0 {
            const SPARSE_RATIO: i32 = 100;
            if distinct_values_int.last().unwrap() / SPARSE_RATIO > distinct_values_int.len() as i32 {
                warnings.push(
                    "Met categorical feature which contains sparse values. \
                     Consider renumbering to consecutive integers started from zero"
                        .into(),
                );
            }
            // upstream: Common::SortForPair(.., is_reverse = true) — stable, by count descending
            let mut pairs: Vec<(i32, i32)> = counts_int.iter().copied().zip(distinct_values_int.iter().copied()).collect();
            pairs.sort_by_key(|p| std::cmp::Reverse(p.0));
            let (counts_int, distinct_values_int): (Vec<i32>, Vec<i32>) = pairs.into_iter().unzip();
            // upstream: RoundInt((total_sample_cnt - na_cnt) * 0.99f), a float product
            let cut_cnt = ((total_sample_cnt.wrapping_sub(na_cnt as usize) as f32 * 0.99f32) as f64 + 0.5f32 as f64) as i32;
            let mut distinct_cnt = distinct_values_int.len() as i32;
            if na_cnt > 0 {
                distinct_cnt += 1;
            }
            let max_bin = distinct_cnt.min(p.max_bin);
            m.bin_2_categorical.push(-1);
            m.categorical_2_bin.insert(-1, 0);
            cnt_in_bin.push(0);
            m.num_bin = 1;
            let mut used_cnt = 0i32;
            let mut cur_cat_idx = 0usize;
            while cur_cat_idx < distinct_values_int.len() && (used_cnt < cut_cnt || m.num_bin < max_bin) {
                if counts_int[cur_cat_idx] < p.min_data_in_bin && cur_cat_idx > 1 {
                    break;
                }
                m.bin_2_categorical.push(distinct_values_int[cur_cat_idx]);
                m.categorical_2_bin.insert(distinct_values_int[cur_cat_idx], m.num_bin as u32);
                used_cnt += counts_int[cur_cat_idx];
                cnt_in_bin.push(counts_int[cur_cat_idx]);
                m.num_bin += 1;
                cur_cat_idx += 1;
            }
            // MissingType::None means every category got its own bin
            m.missing_type = if cur_cat_idx == distinct_values_int.len() && na_cnt == 0 {
                MissingType::None
            } else {
                MissingType::NaN
            };
            cnt_in_bin[0] = (total_sample_cnt as i64 - used_cnt as i64) as i32;
        }
        m.finish(&cnt_in_bin, total_sample_cnt, p);
        m
    }

    /// upstream: the tail of `BinMapper::FindBin` (trivial check, pre-filter,
    /// default / most frequent bin, sparse rate).
    fn finish(&mut self, cnt_in_bin: &[i32], total_sample_cnt: usize, p: &BinParams) {
        let m = self;
        m.is_trivial = m.num_bin <= 1;
        if !m.is_trivial
            && p.pre_filter
            && need_filter(cnt_in_bin, total_sample_cnt as i32, p.min_split_data, m.bin_type)
        {
            m.is_trivial = true;
        }
        if !m.is_trivial {
            m.default_bin = m.value_to_bin(0.0);
            let mut arg_max = 0usize;
            for i in 1..cnt_in_bin.len() {
                if cnt_in_bin[i] > cnt_in_bin[arg_max] {
                    arg_max = i;
                }
            }
            m.most_freq_bin = arg_max as u32;
            let max_sparse_rate = cnt_in_bin[arg_max] as f64 / total_sample_cnt as f64;
            if m.most_freq_bin != m.default_bin && max_sparse_rate < K_SPARSE_THRESHOLD {
                m.most_freq_bin = m.default_bin;
            }
            m.sparse_rate = cnt_in_bin[m.most_freq_bin as usize] as f64 / total_sample_cnt as f64;
        } else {
            m.sparse_rate = 1.0;
        }
    }

    /// upstream: include/LightGBM/bin.h `BinMapper::ValueToBin`.
    #[inline]
    pub fn value_to_bin(&self, value: f64) -> u32 {
        if self.bin_type == BinType::Categorical {
            if value.is_nan() {
                return 0;
            }
            // negative values (and unseen categories) go to the "other" bin 0
            let v = value as i32;
            return if v < 0 { 0 } else { self.categorical_2_bin.get(&v).copied().unwrap_or(0) };
        }
        let mut value = value;
        if value.is_nan() {
            if self.missing_type == MissingType::NaN {
                return (self.num_bin - 1) as u32;
            }
            value = 0.0;
        }
        let mut l: i32 = 0;
        let mut r: i32 = self.num_bin - 1;
        if self.missing_type == MissingType::NaN {
            r -= 1;
        }
        while l < r {
            let m = (r + l - 1) / 2;
            if value <= self.bin_upper_bound[m as usize] {
                r = m;
            } else {
                l = m + 1;
            }
        }
        l as u32
    }

    /// upstream: `BinMapper::BinToValue` — upper bound of a numerical bin,
    /// or the category of a categorical bin.
    pub fn bin_to_value(&self, bin: u32) -> f64 {
        if self.bin_type == BinType::Categorical {
            self.bin_2_categorical[bin as usize] as f64
        } else {
            self.bin_upper_bound[bin as usize]
        }
    }

    /// upstream: include/LightGBM/bin.h `bin_info_string`.
    pub fn bin_info_string(&self) -> String {
        if self.bin_type == BinType::Categorical {
            self.bin_2_categorical.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(":")
        } else {
            format!("[{}:{}]", fmt_g17(self.min_val), fmt_g17(self.max_val))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(max_bin: i32) -> BinParams {
        BinParams {
            max_bin,
            min_data_in_bin: 3,
            min_split_data: 0,
            pre_filter: false,
            use_missing: true,
            zero_as_missing: false,
        }
    }

    #[test]
    fn small_distinct_values_get_midpoint_bounds() {
        let vals: Vec<f64> = (1..=10).flat_map(|v| std::iter::repeat_n(v as f64, 5)).collect();
        let m = BinMapper::find_bin(&vals, vals.len(), BinType::Numerical, &params(255)).unwrap();
        assert_eq!(m.missing_type, MissingType::None);
        // zero bound, then midpoints 1.5 .. 9.5, then +inf
        assert_eq!(m.bin_upper_bound[0], K_ZERO_THRESHOLD);
        assert_eq!(m.bin_upper_bound[1], 1.5f64.next_up());
        assert_eq!(*m.bin_upper_bound.last().unwrap(), f64::INFINITY);
        assert_eq!(m.value_to_bin(1.0), 1);
        assert_eq!(m.value_to_bin(10.0), m.num_bin as u32 - 1);
        assert_eq!(m.value_to_bin(-5.0), 0);
    }

    #[test]
    fn nan_gets_last_bin() {
        let mut vals: Vec<f64> = (1..=50).map(|v| v as f64).collect();
        vals.push(f64::NAN);
        vals.push(f64::NAN);
        let m = BinMapper::find_bin(&vals, vals.len(), BinType::Numerical, &params(255)).unwrap();
        assert_eq!(m.missing_type, MissingType::NaN);
        assert_eq!(m.value_to_bin(f64::NAN), m.num_bin as u32 - 1);
        assert!(m.bin_upper_bound.last().unwrap().is_nan());
    }

    #[test]
    fn greedy_respects_max_bin() {
        let vals: Vec<f64> = (0..10_000).map(|v| (v as f64 * 0.37).sin() + 2.0).collect();
        let m = BinMapper::find_bin(&vals, vals.len(), BinType::Numerical, &params(16)).unwrap();
        assert!(m.num_bin <= 16);
        for w in m.bin_upper_bound.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn categorical_bins_by_count() {
        // category 3 x30, 1 x20, 2 x10, -1 x5 (-> NaN), zeros x15 (implicit)
        let mut vals = Vec::new();
        vals.extend(std::iter::repeat_n(3.0, 30));
        vals.extend(std::iter::repeat_n(1.0, 20));
        vals.extend(std::iter::repeat_n(2.7, 10));
        vals.extend(std::iter::repeat_n(-1.0, 5));
        let total = vals.len() + 15;
        let mut w = Vec::new();
        let m = BinMapper::find_bin_logged(&vals, total, BinType::Categorical, &params(255), &[], &mut w).unwrap();
        assert_eq!(m.bin_2_categorical, vec![-1, 3, 1, 0, 2]);
        assert_eq!(m.missing_type, MissingType::NaN);
        assert_eq!(m.bin_info_string(), "-1:3:1:0:2");
        assert_eq!(m.value_to_bin(2.9), 4);
        assert_eq!(m.value_to_bin(0.0), 3);
        assert_eq!(m.value_to_bin(-3.0), 0);
        assert_eq!(m.value_to_bin(f64::NAN), 0);
        assert_eq!(m.value_to_bin(7.0), 0);
        assert_eq!(m.bin_to_value(2), 1.0);
        assert_eq!(m.default_bin, 3);
        assert!(w[0].starts_with("Met negative value"));
    }

    #[test]
    fn forced_bounds_are_kept() {
        // upstream examples/regression/forced_bins.json on x = 0, 0.01, ..., 0.99 with max_bin = 5
        let vals: Vec<f64> = (1..100).map(|i| i as f64 * 0.01).collect();
        let m = BinMapper::find_bin_logged(&vals, 100, BinType::Numerical, &params(5), &[0.3, 0.35, 0.4], &mut Vec::new())
            .unwrap();
        assert_eq!(m.bin_upper_bound, vec![K_ZERO_THRESHOLD, 0.3, 0.35, 0.4, f64::INFINITY]);
        // no free bins are left, so an empty forced interval adds nothing
        let b = find_bin_with_predefined_bin(&[1.0, 2.0], &[5, 5], 4, 10, 1, &[-3.0, 5.0]).unwrap();
        assert_eq!(b, vec![-3.0, K_ZERO_THRESHOLD, 5.0, f64::INFINITY]);
    }

    #[test]
    fn constant_feature_is_trivial() {
        let m = BinMapper::find_bin(&[], 100, BinType::Numerical, &params(255)).unwrap();
        assert!(m.is_trivial);
    }
}
