//! Per-feature gradient/Hessian histograms.
//!
//! upstream: `Dataset::ConstructHistograms`, `DenseBin::ConstructHistogram`,
//! and `Dataset::FixHistogram` (src/io/dataset.cpp, src/io/dense_bin.hpp).
//!
//! Layout: `[g0, h0, g1, h1, ...]` with `num_bin - offset` entries, where
//! `offset = 1` when bin 0 is the most frequent bin (that bin is never
//! stored). Rows in the most frequent bin are skipped during accumulation
//! and recovered afterwards from the leaf totals, exactly as upstream does.
//! Accumulation runs in data-index order, which fixes the summation order.

use crate::binning::BinMapper;
use crate::dataset::BinColumn;

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

/// Accumulate rows into `hist`. With `indices = Some(idx)`, `og[k]`/`oh[k]`
/// are the gradients of row `idx[k]` (already gathered); otherwise row `k`.
pub fn construct(
    col: &BinColumn,
    indices: Option<&[u32]>,
    og: &[f32],
    oh: &[f32],
    layout: &HistLayout,
    hist: &mut [f64],
) {
    fn run<T: Copy + Into<u32>>(
        bins: &[T],
        indices: Option<&[u32]>,
        og: &[f32],
        oh: &[f32],
        mfb: u32,
        offset: u32,
        hist: &mut [f64],
    ) {
        let mut add = |b: u32, k: usize| {
            if b != mfb {
                let t = ((b - offset) as usize) << 1;
                hist[t] += og[k] as f64;
                hist[t + 1] += oh[k] as f64;
            }
        };
        match indices {
            Some(idx) => {
                for (k, &i) in idx.iter().enumerate() {
                    add(bins[i as usize].into(), k);
                }
            }
            None => {
                for (k, &b) in bins.iter().enumerate() {
                    add(b.into(), k);
                }
            }
        }
    }
    let mfb = layout.most_freq_bin;
    let off = layout.offset as u32;
    match col {
        BinColumn::U8(v) => run(v, indices, og, oh, mfb, off, hist),
        BinColumn::U16(v) => run(v, indices, og, oh, mfb, off, hist),
        BinColumn::U32(v) => run(v, indices, og, oh, mfb, off, hist),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fix_recovers_most_frequent_bin() {
        // bins: 0,1,1,2 with mfb = 1 -> rows in bin 1 skipped then recovered
        let col = BinColumn::U8(vec![0, 1, 1, 2]);
        let layout = HistLayout { num_bin: 3, offset: 0, most_freq_bin: 1 };
        let g = [1.0f32, 2.0, 3.0, 4.0];
        let h = [1.0f32; 4];
        let mut hist = vec![0.0; layout.len()];
        construct(&col, None, &g, &h, &layout, &mut hist);
        assert_eq!(hist, vec![1.0, 1.0, 0.0, 0.0, 4.0, 1.0]);
        fix(&layout, 10.0, 4.0, &mut hist);
        assert_eq!(hist, vec![1.0, 1.0, 5.0, 2.0, 4.0, 1.0]);
    }

    #[test]
    fn offset_one_drops_bin_zero() {
        let col = BinColumn::U8(vec![0, 0, 1, 2]);
        let layout = HistLayout { num_bin: 3, offset: 1, most_freq_bin: 0 };
        let g = [1.0f32, 1.0, 2.0, 3.0];
        let h = [1.0f32; 4];
        let mut hist = vec![0.0; layout.len()];
        construct(&col, Some(&[0, 2, 3]), &[g[0], g[2], g[3]], &[h[0], h[2], h[3]], &layout, &mut hist);
        assert_eq!(hist, vec![2.0, 1.0, 3.0, 1.0]);
    }
}
