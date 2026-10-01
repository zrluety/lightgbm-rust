//! Row-major bins of all features and row-wise histogram construction.
//!
//! upstream: src/io/multi_val_dense_bin.hpp (`MultiValDenseBin`),
//! include/LightGBM/train_share_states.h (`MultiValBinWrapper::
//! ConstructHistograms`), src/io/train_share_states.cpp (`HistMerge`,
//! `InitTrain`), and include/LightGBM/utils/threading.h (`BlockInfo`).
//!
//! Every feature owns `1 + num_bin - offset` consecutive histogram slots:
//! slot 0 collects the rows in the most frequent bin (discarded; that bin is
//! recovered by `histogram::fix`), and bin `b` maps to slot `b - offset + 1`.
//! The total slot count equals upstream's row-wise `num_bin`, so the block
//! partition (and therefore the floating-point summation order) matches
//! upstream for every thread count: rows are split into
//! `min(threads, ceil(cnt / min_block_size))` blocks, each block is summed in
//! row order into its own buffer, and buffers are added in block order.

use rayon::prelude::*;

use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::{BinColumn, Dataset};
use crate::threading::{SharedMut, ThreadTeam};

const K_ALIGNED_SIZE: usize = 32;

#[derive(Debug, Clone)]
enum RowData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

/// Per-feature location of a feature's stored bins inside a flat histogram.
#[derive(Debug, Clone, Copy)]
pub struct FeatureView {
    /// Index of the first f64 (gradient of the first stored bin).
    pub start: usize,
    /// Number of f64 values (`2 * (num_bin - offset)`).
    pub len: usize,
}

/// Slot layout of the flat histogram shared by the row-wise and col-wise paths.
#[derive(Debug, Clone)]
pub struct HistSlots {
    pub views: Vec<FeatureView>,
    /// First slot of each feature (its most-frequent-bin slot).
    pub base: Vec<u32>,
    pub num_slots: usize,
}

impl HistSlots {
    pub fn new(data: &Dataset) -> Self {
        let nf = data.num_features();
        let mut base = Vec::with_capacity(nf);
        let mut views = Vec::with_capacity(nf);
        let mut cur = 0usize;
        for f in 0..nf {
            let m = data.feature_bin_mapper(f);
            let offset = usize::from(m.most_freq_bin == 0);
            let stored = m.num_bin as usize - offset;
            base.push(cur as u32);
            views.push(FeatureView { start: 2 * (cur + 1), len: 2 * stored });
            cur += 1 + stored;
        }
        Self { views, base, num_slots: cur }
    }

    /// Length of a flat histogram buffer in f64 values.
    pub fn buf_len(&self) -> usize {
        2 * self.num_slots
    }

    /// Disjoint mutable per-feature views into a flat histogram.
    pub fn split_mut<'a>(&self, mut buf: &'a mut [f64]) -> Vec<&'a mut [f64]> {
        let mut out = Vec::with_capacity(self.views.len());
        let mut pos = 0usize;
        for v in &self.views {
            let (_, rest) = buf.split_at_mut(v.start - pos);
            let (view, rest) = rest.split_at_mut(v.len);
            out.push(view);
            buf = rest;
            pos = v.start + v.len;
        }
        out
    }
}

/// upstream: `MultiValDenseBin` (dense features only).
#[derive(Debug, Clone)]
pub struct MultiValBin {
    data: RowData,
    num_feature: usize,
    num_data: usize,
    base: Vec<u32>,
    num_slots: usize,
    min_block_size: usize,
}

/// upstream: `Threading::BlockInfo(num_threads, cnt, min_cnt_per_block, ...)`.
pub fn block_info(num_threads: usize, cnt: usize, min_cnt_per_block: usize) -> (usize, usize) {
    let nblock = num_threads.min(cnt.div_ceil(min_cnt_per_block)).max(1);
    if nblock > 1 {
        let size = cnt.div_ceil(nblock);
        (nblock, size.div_ceil(K_ALIGNED_SIZE) * K_ALIGNED_SIZE)
    } else {
        (1, cnt)
    }
}

impl MultiValBin {
    pub fn new(data: &Dataset, slots: &HistSlots) -> Self {
        let n = data.num_data();
        let nf = data.num_features();
        let mfb: Vec<u32> = (0..nf).map(|f| data.feature_bin_mapper(f).most_freq_bin).collect();
        let max_slot = (0..nf)
            .map(|f| data.feature_bin_mapper(f).num_bin as usize - usize::from(mfb[f] == 0))
            .max()
            .unwrap_or(0);
        fn fill<T: Copy + Send + Sync + Default>(
            data: &Dataset,
            mfb: &[u32],
            conv: impl Fn(u32) -> T + Sync,
        ) -> Vec<T> {
            let (n, nf) = (data.num_data(), mfb.len());
            let mut out = vec![T::default(); n * nf];
            if nf == 0 {
                return out;
            }
            const ROWS: usize = 4096;
            out.par_chunks_mut(ROWS * nf).enumerate().for_each(|(c, chunk)| {
                let r0 = c * ROWS;
                let rows = chunk.len() / nf;
                for (f, &m) in mfb.iter().enumerate() {
                    let offset = u32::from(m == 0);
                    let mut put = |r: usize, b: u32| {
                        chunk[r * nf + f] = conv(if b == m { 0 } else { b - offset + 1 });
                    };
                    match data.feature_bins(f) {
                        BinColumn::U8(v) => v[r0..r0 + rows].iter().enumerate().for_each(|(r, &b)| put(r, b as u32)),
                        BinColumn::U16(v) => v[r0..r0 + rows].iter().enumerate().for_each(|(r, &b)| put(r, b as u32)),
                        BinColumn::U32(v) => v[r0..r0 + rows].iter().enumerate().for_each(|(r, &b)| put(r, b)),
                    }
                }
            });
            out
        }
        let rows = if max_slot <= u8::MAX as usize {
            RowData::U8(fill(data, &mfb, |s| s as u8))
        } else if max_slot <= u16::MAX as usize {
            RowData::U16(fill(data, &mfb, |s| s as u16))
        } else {
            RowData::U32(fill(data, &mfb, |s| s))
        };
        // upstream: MultiValBinWrapper::InitTrain
        let mbs = ((0.3f32 * slots.num_slots as f32) as f64 / (nf as f64 + K_ZERO_THRESHOLD))
            as i64
            + 1;
        let min_block_size = mbs.clamp(32, 1024) as usize;
        Self {
            data: rows,
            num_feature: nf,
            num_data: n,
            base: slots.base.clone(),
            num_slots: slots.num_slots,
            min_block_size,
        }
    }

    pub fn min_block_size(&self) -> usize {
        self.min_block_size
    }

    /// Accumulate the rows `indices` (all rows when `None`) block-wise: block
    /// 0 into `out`, block `b > 0` into `bufs[b - 1]` (all overwritten).
    /// Returns the number of blocks; the caller completes the histogram with
    /// [`merge_blocks`] (upstream `HistMerge`). `gh[i]` is `[grad, hess]` of
    /// row `i` (interleaved so an indexed row touches one gradient line).
    pub fn construct_blocks(
        &self,
        team: &ThreadTeam,
        indices: Option<&[u32]>,
        gh: &[[f32; 2]],
        out: &mut [f64],
        bufs: &mut Vec<Vec<f64>>,
    ) -> usize {
        let cnt = indices.map_or(self.num_data, |ix| ix.len());
        let (nblock, bsize) = block_info(team.num_threads(), cnt, self.min_block_size);
        let len = 2 * self.num_slots;
        assert_eq!(out.len(), len);
        if nblock == 1 {
            out.fill(0.0);
            self.block(indices, 0, cnt, gh, out);
            return 1;
        }
        if bufs.len() < nblock - 1 {
            bufs.resize_with(nblock - 1, Vec::new);
        }
        for b in bufs.iter_mut().take(nblock - 1) {
            b.resize(len, 0.0);
        }
        let targets: Vec<SharedMut<f64>> = std::iter::once(SharedMut::new(out))
            .chain(bufs.iter_mut().take(nblock - 1).map(|b| SharedMut::new(b)))
            .collect();
        team.for_each(nblock, cnt * self.num_feature + nblock * len, |b| {
            // SAFETY: each block writes only its own buffer.
            let dst = unsafe { targets[b].slice(0, len) };
            dst.fill(0.0);
            let start = (b * bsize).min(cnt);
            let end = (start + bsize).min(cnt);
            if start < end {
                self.block(indices, start, end, gh, dst);
            }
        });
        nblock
    }

    fn block(&self, indices: Option<&[u32]>, start: usize, end: usize, gh: &[[f32; 2]], out: &mut [f64]) {
        match &self.data {
            RowData::U8(d) => self.block_typed(d, indices, start, end, gh, out),
            RowData::U16(d) => self.block_typed(d, indices, start, end, gh, out),
            RowData::U32(d) => self.block_typed(d, indices, start, end, gh, out),
        }
    }

    #[inline(always)]
    fn block_typed<T: Copy + Into<u32>>(
        &self,
        data: &[T],
        indices: Option<&[u32]>,
        start: usize,
        end: usize,
        gh: &[[f32; 2]],
        out: &mut [f64],
    ) {
        let nf = self.num_feature;
        let base = &self.base[..nf];
        assert!(out.len() >= 2 * self.num_slots && gh.len() >= self.num_data);
        let mut add_row = |i: usize| {
            let row = &data[i * nf..(i + 1) * nf];
            // SAFETY: i < num_data (checked by the slice above and the
            // assert on gh), and every slot is < num_slots by construction.
            let [g, h] = unsafe { *gh.get_unchecked(i) };
            let (gi, hi) = (g as f64, h as f64);
            for (&b, &o) in row.iter().zip(base) {
                let t = ((o + b.into()) as usize) << 1;
                unsafe {
                    *out.get_unchecked_mut(t) += gi;
                    *out.get_unchecked_mut(t + 1) += hi;
                }
            }
        };
        match indices {
            Some(ix) => {
                let ix = &ix[start..end];
                // upstream: MultiValDenseBin::ConstructHistogramInner prefetches
                // the row (and its gradients) 32 / sizeof(VAL_T) positions ahead.
                let pf = 32 / std::mem::size_of::<T>();
                let split = ix.len().saturating_sub(pf);
                for k in 0..split {
                    let p = ix[k + pf] as usize;
                    prefetch(data.as_ptr().wrapping_add(p * nf));
                    prefetch(gh.as_ptr().wrapping_add(p));
                    add_row(ix[k] as usize);
                }
                ix[split..].iter().for_each(|&i| add_row(i as usize));
            }
            None => (start..end).for_each(add_row),
        }
    }
}

/// upstream `HistMerge` restricted to `dst = out[start..start + dst.len()]`:
/// adds blocks `1..nblock` in block order, bin by bin.
pub fn merge_blocks(dst: &mut [f64], bufs: &[Vec<f64>], start: usize, nblock: usize) {
    let len = dst.len();
    for src in &bufs[..nblock.saturating_sub(1)] {
        for (d, s) in dst.iter_mut().zip(&src[start..start + len]) {
            *d += *s;
        }
    }
}

#[inline(always)]
fn prefetch<T>(p: *const T) {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: prefetch is a hint and never faults, even for invalid addresses.
    unsafe {
        use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        _mm_prefetch(p as *const i8, _MM_HINT_T0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = p;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_info_matches_upstream() {
        assert_eq!(block_info(1, 1000, 77), (1, 1000));
        assert_eq!(block_info(16, 1000, 77), (13, 96));
        assert_eq!(block_info(16, 1_000_000, 77), (16, 62_528));
        assert_eq!(block_info(4, 50, 77), (1, 50));
    }
}
