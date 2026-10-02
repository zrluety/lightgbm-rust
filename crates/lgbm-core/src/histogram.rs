//! Per-feature gradient/Hessian histograms.
//!
//! upstream: `Dataset::FixHistogram` (src/io/dataset.cpp). Histograms are
//! built per feature group (see [`crate::bin::construct_histogram`] and
//! [`crate::multi_val_bin`]); a feature's view of a leaf buffer is
//! `[g0, h0, g1, h1, ...]` with `num_bin - offset` entries, where
//! `offset = 1` when bin 0 is the most frequent bin (that bin is never
//! stored). Rows in the most frequent bin are not stored and are recovered
//! afterwards from the leaf totals, exactly as upstream does.

use crate::binning::BinMapper;

/// Histogram geometry of one feature (subset of upstream `FeatureMetainfo`).
#[derive(Debug, Clone, Copy)]
pub struct HistLayout {
    pub num_bin: i32,
    pub offset: i32,
    pub most_freq_bin: u32,
}

impl HistLayout {
    pub fn of(m: &BinMapper) -> Self {
        Self {
            num_bin: m.num_bin,
            offset: if m.most_freq_bin == 0 { 1 } else { 0 },
            most_freq_bin: m.most_freq_bin,
        }
    }

    /// Number of f64 slots (gradient + Hessian per stored bin).
    pub fn len(&self) -> usize {
        2 * (self.num_bin - self.offset) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// upstream: `Dataset::FixHistogram` — fill the most frequent bin from the
/// leaf totals (only needed when that bin is stored, i.e. `mfb > 0`).
pub fn fix(layout: &HistLayout, sum_g: f64, sum_h: f64, hist: &mut [f64]) {
    let mfb = layout.most_freq_bin as usize;
    if mfb > 0 {
        let mut g = sum_g;
        let mut h = sum_h;
        for i in 0..layout.num_bin as usize {
            if i != mfb {
                g -= hist[2 * i];
                h -= hist[2 * i + 1];
            }
        }
        hist[2 * mfb] = g;
        hist[2 * mfb + 1] = h;
    }
}

/// Histogram subtraction: `larger = parent - smaller`, in place on `parent`.
pub fn subtract(parent: &mut [f64], smaller: &[f64]) {
    for (a, b) in parent.iter_mut().zip(smaller) {
        *a -= *b;
    }
}

/// upstream `Dataset::FixHistogramInt` on packed integer sums (one `i64`
/// per bin: gradient in the high half, hessian in the low half).
pub fn fix_int(layout: &HistLayout, int_sum: i64, hist: &mut [i64]) {
    let mfb = layout.most_freq_bin as usize;
    if mfb > 0 {
        let mut s = int_sum;
        for (i, &v) in hist.iter().enumerate().take(layout.num_bin as usize) {
            if i != mfb {
                s = s.wrapping_sub(v);
            }
        }
        hist[mfb] = s;
    }
}

/// [`subtract`] on packed integer sums.
pub fn subtract_int(parent: &mut [i64], smaller: &[i64]) {
    for (a, b) in parent.iter_mut().zip(smaller) {
        *a = a.wrapping_sub(*b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_recovers_most_frequent_bin() {
        // bins 0,1,1,2 with gradients 1,2,3,4 and mfb = 1: bin 1 is not stored
        let layout = HistLayout { num_bin: 3, offset: 0, most_freq_bin: 1 };
        let mut hist = vec![1.0, 1.0, 0.0, 0.0, 4.0, 1.0];
        fix(&layout, 10.0, 4.0, &mut hist);
        assert_eq!(hist, vec![1.0, 1.0, 5.0, 2.0, 4.0, 1.0]);
    }

    #[test]
    fn offset_one_leaves_the_histogram_alone() {
        let layout = HistLayout { num_bin: 3, offset: 1, most_freq_bin: 0 };
        assert_eq!(layout.len(), 4);
        let mut hist = vec![2.0, 1.0, 3.0, 1.0];
        fix(&layout, 7.0, 3.0, &mut hist);
        assert_eq!(hist, vec![2.0, 1.0, 3.0, 1.0]);
    }
}
