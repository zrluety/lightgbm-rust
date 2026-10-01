//! upstream: src/objective/regression_objective.hpp `PercentileFun` and
//! `WeightedPercentileFun`.
//!
//! Both are generic over the value type `T` (`label_t = f32` when applied to
//! labels, `double` when applied to residuals); the interpolation arithmetic
//! mixes `T` and `double` exactly as the C++ macros do.

use std::cmp::Ordering;

pub trait PercentileValue: Copy + PartialOrd {
    /// `a - b` in `T`.
    fn sub(self, other: Self) -> Self;
    fn to_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
}

impl PercentileValue for f32 {
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
}

impl PercentileValue for f64 {
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
}

fn desc<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    b.partial_cmp(a).unwrap_or(Ordering::Equal)
}

fn first_max<T: PartialOrd + Copy>(v: &[T]) -> T {
    let mut m = v[0];
    for &x in &v[1..] {
        if x > m {
            m = x;
        }
    }
    m
}

fn first_min<T: PartialOrd + Copy>(v: &[T]) -> T {
    let mut m = v[0];
    for &x in &v[1..] {
        if x < m {
            m = x;
        }
    }
    m
}

/// `PercentileFun`: the `alpha` quantile of `cnt` values read by `read`,
/// interpolated between neighbouring order statistics.
///
/// Upstream selects with `ArrayArgs::ArgMaxAtK`; only order statistics enter
/// the result, so any exact selection gives the same value.
pub fn percentile<T: PercentileValue>(cnt: usize, read: impl Fn(usize) -> T, alpha: f64) -> T {
    if cnt <= 1 {
        return read(0);
    }
    let mut v: Vec<T> = (0..cnt).map(&read).collect();
    let float_pos = (cnt - 1) as f64 * (1.0 - alpha);
    let pos = float_pos as i64 + 1;
    if pos < 1 {
        return first_max(&v);
    }
    if pos as usize >= cnt {
        return first_min(&v);
    }
    let pos = pos as usize;
    let bias = float_pos - (pos - 1) as f64;
    // v1: the (pos-1)-th largest (0-based), v2: the pos-th largest.
    v.select_nth_unstable_by(pos - 1, desc);
    let v1 = v[pos - 1];
    let v2 = first_max(&v[pos..]);
    T::from_f64(v1.to_f64() - v1.sub(v2).to_f64() * bias)
}

/// `WeightedPercentileFun`: the weighted `alpha` quantile.
pub fn weighted_percentile<T: PercentileValue>(
    cnt: usize,
    read: impl Fn(usize) -> T,
    weight: impl Fn(usize) -> f64,
    alpha: f64,
) -> T {
    if cnt <= 1 {
        return read(0);
    }
    let mut sorted: Vec<usize> = (0..cnt).collect();
    // std::stable_sort with `data_reader(a) < data_reader(b)`
    sorted.sort_by(|&a, &b| read(a).partial_cmp(&read(b)).unwrap_or(Ordering::Equal));
    let mut cdf = vec![0.0f64; cnt];
    cdf[0] = weight(sorted[0]);
    for i in 1..cnt {
        cdf[i] = cdf[i - 1] + weight(sorted[i]);
    }
    let threshold = cdf[cnt - 1] * alpha;
    // std::upper_bound: first element > threshold
    let pos = cdf.partition_point(|&c| c <= threshold).min(cnt - 1);
    if pos == 0 || pos == cnt - 1 {
        return read(sorted[pos]);
    }
    let v1 = read(sorted[pos - 1]);
    let v2 = read(sorted[pos]);
    if cdf[pos] - cdf[pos - 1] >= 1.0 {
        T::from_f64((threshold - cdf[pos - 1]) / (cdf[pos] - cdf[pos - 1]) * v2.sub(v1).to_f64() + v1.to_f64())
    } else {
        v1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_and_extremes() {
        let v = [3.0f64, 1.0, 4.0, 1.0, 5.0];
        assert_eq!(percentile(5, |i| v[i], 0.5), 3.0);
        assert_eq!(percentile(5, |i| v[i], 1.0), 5.0);
        assert_eq!(percentile(5, |i| v[i], 0.0), 1.0);
        // even count interpolates: float_pos = 1.5, values 4 and 3
        let w = [1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(percentile(4, |i| w[i], 0.5), 2.5);
        assert_eq!(percentile(1, |i| w[i], 0.3), 1.0);
    }

    #[test]
    fn weighted_matches_unweighted_shape() {
        let v = [1.0f64, 2.0, 3.0, 4.0];
        // cdf = 1,2,3,4; threshold 2 -> pos 2 (first > 2), interpolate 2..3 by 0
        assert_eq!(weighted_percentile(4, |i| v[i], |_| 1.0, 0.5), 2.0);
        // heavier weight on 4 moves the median up to the last element
        assert_eq!(weighted_percentile(4, |i| v[i], |i| if i == 3 { 10.0 } else { 1.0 }, 0.5), 4.0);
    }
}
