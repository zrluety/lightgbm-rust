//! Histogram layout and row-wise (multi-value) bins.
//!
//! upstream: src/io/train_share_states.cpp (`CalcBinOffsets`,
//! `MultiValBinWrapper::InitTrain`, `CopyMultiValBinSubset`, `HistMerge`,
//! `HistMove`), include/LightGBM/train_share_states.h
//! (`MultiValBinWrapper::ConstructHistograms`), src/io/dataset.cpp
//! (`GetMultiBinFromSparseFeatures`, `GetMultiBinFromAllFeatures`,
//! `PushDataToMultiValBin`), src/io/multi_val_dense_bin.hpp,
//! src/io/multi_val_sparse_bin.hpp, src/io/bin.cpp (`CreateMultiValBin`) and
//! include/LightGBM/utils/threading.h (`BlockInfo`).
//!
//! A leaf's histogram buffer is laid out as upstream's: each feature's
//! stored bins start at upstream's `feature_hist_offsets_`. Col-wise, the
//! groups are built one by one into their regions (and the multi-value group
//! row-wise into its region); row-wise, one multi-value bin over all groups
//! fills the whole buffer. Rows are split into upstream's blocks (from the
//! thread count and the bin's `num_bin` and elements per row), each block is
//! summed in row order, and blocks are added in block order, so every bin's
//! summation order matches upstream for the same thread count.

use rayon::prelude::*;

use crate::bin::{AnyCursor, RawCursor};
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::Dataset;
use crate::feature_group::{FeatureCursor, MULTI_VAL_BIN_SPARSE_THRESHOLD};
use crate::threading::{SharedMut, ThreadTeam};

const K_ALIGNED_SIZE: usize = 32;

/// Location of a feature's stored bins inside a leaf's histogram buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureView {
    /// Index of the first f64 (gradient of the first stored bin).
    pub start: usize,
    /// Number of f64 values (`2 * (num_bin - offset)`).
    pub len: usize,
}

/// upstream `TrainingShareStates::CalcBinOffsets` and `group_bin_boundaries_`.
#[derive(Debug, Clone)]
pub struct HistOffsets {
    /// By inner feature.
    pub views: Vec<FeatureView>,
    /// upstream `num_hist_total_bin_`.
    pub num_total_bin: usize,
    /// upstream `group_bin_boundaries_` (col-wise group regions), one more than groups.
    pub group_start: Vec<usize>,
    /// upstream `offsets`: column offsets of the multi-value bin.
    pub mv_offsets: Vec<u32>,
    pub col_wise: bool,
}

impl HistOffsets {
    pub fn new(data: &Dataset, col_wise: bool) -> Self {
        let groups = data.feature_groups();
        let mut group_start = vec![0usize];
        for g in groups {
            group_start.push(group_start.last().unwrap() + g.num_total_bin as usize);
        }
        let mut fho = vec![0u32; data.num_features()];
        let mut offsets = Vec::new();
        let total = if col_wise {
            let mut cur = 0u32;
            let mut hist_cur = 0u32;
            for (gi, g) in groups.iter().enumerate() {
                if g.is_multi_val {
                    if g.is_dense_multi_val {
                        for (i, s) in g.subs.iter().enumerate() {
                            if gi == 0 && i == 0 && s.most_freq_bin > 0 {
                                cur += 1;
                                hist_cur += 1;
                            }
                            offsets.push(cur);
                            fho[s.inner] = hist_cur + u32::from(s.most_freq_bin == 0);
                            hist_cur += s.num_bin;
                            cur += s.num_bin;
                        }
                    } else {
                        cur += 1;
                        hist_cur += 1;
                        for s in &g.subs {
                            offsets.push(cur);
                            fho[s.inner] = hist_cur;
                            let nb = s.num_bin - u32::from(s.most_freq_bin == 0);
                            hist_cur += nb;
                            cur += nb;
                        }
                    }
                    offsets.push(cur);
                    debug_assert_eq!(cur, g.num_total_bin);
                } else {
                    for (i, s) in g.subs.iter().enumerate() {
                        fho[s.inner] = hist_cur + g.bin_offsets[i];
                    }
                    hist_cur += g.num_total_bin;
                }
            }
            hist_cur
        } else if row_wise_sparse_rate(data) >= MULTI_VAL_BIN_SPARSE_THRESHOLD {
            let mut cur = 1u32;
            let mut hist_cur = 1u32;
            for g in groups {
                if g.is_multi_val {
                    for s in &g.subs {
                        offsets.push(cur);
                        fho[s.inner] = hist_cur;
                        let nb = s.num_bin - u32::from(s.most_freq_bin == 0);
                        cur += nb;
                        hist_cur += nb;
                    }
                } else {
                    offsets.push(cur);
                    cur += g.num_total_bin - 1;
                    for (i, s) in g.subs.iter().enumerate() {
                        fho[s.inner] = hist_cur + g.bin_offsets[i] - 1;
                    }
                    hist_cur += g.num_total_bin - 1;
                }
            }
            offsets.push(cur);
            hist_cur
        } else {
            let mut cur = 0u32;
            let mut hist_cur = 0u32;
            for (gi, g) in groups.iter().enumerate() {
                if g.is_multi_val {
                    for (i, s) in g.subs.iter().enumerate() {
                        if gi == 0 && i == 0 && s.most_freq_bin > 0 {
                            cur += 1;
                            hist_cur += 1;
                        }
                        offsets.push(cur);
                        fho[s.inner] = hist_cur + u32::from(s.most_freq_bin == 0);
                        cur += s.num_bin;
                        hist_cur += s.num_bin;
                    }
                } else {
                    offsets.push(cur);
                    cur += g.num_total_bin;
                    for (i, s) in g.subs.iter().enumerate() {
                        fho[s.inner] = hist_cur + g.bin_offsets[i];
                    }
                    hist_cur += g.num_total_bin;
                }
            }
            offsets.push(cur);
            hist_cur
        };
        let views = (0..data.num_features())
            .map(|f| {
                let m = data.feature_bin_mapper(f);
                let stored = m.num_bin as usize - usize::from(m.most_freq_bin == 0);
                FeatureView { start: 2 * fho[f] as usize, len: 2 * stored }
            })
            .collect();
        Self { views, num_total_bin: total as usize, group_start, mv_offsets: offsets, col_wise }
    }

    /// Length of a leaf's histogram buffer in f64 values.
    pub fn buf_len(&self) -> usize {
        2 * self.num_total_bin
    }

    /// Disjoint mutable per-feature views into a histogram buffer.
    pub fn split_mut<'a>(&self, mut buf: &'a mut [f64]) -> Vec<&'a mut [f64]> {
        let mut order: Vec<usize> = (0..self.views.len()).collect();
        order.sort_by_key(|&f| self.views[f].start);
        let mut out: Vec<Option<&'a mut [f64]>> = (0..self.views.len()).map(|_| None).collect();
        let mut pos = 0usize;
        for f in order {
            let v = self.views[f];
            let (_, rest) = std::mem::take(&mut buf).split_at_mut(v.start - pos);
            let (view, rest) = rest.split_at_mut(v.len);
            out[f] = Some(view);
            buf = rest;
            pos = v.start + v.len;
        }
        out.into_iter().map(Option::unwrap).collect()
    }
}

/// `1 - sum_dense_ratio` of upstream's row-wise layout.
fn row_wise_sparse_rate(data: &Dataset) -> f64 {
        let mut sum_dense_ratio = 0.0f64;
        let mut ncol = 0usize;
        for g in data.feature_groups() {
            ncol += if g.is_multi_val { g.subs.len() } else { 1 };
            for s in &g.subs {
                sum_dense_ratio += 1.0 - s.sparse_rate;
            }
        }
        sum_dense_ratio /= ncol as f64;
        1.0 - sum_dense_ratio
}

/// The sparse rate upstream's `GetShareStates` logs for a forced layout:
/// `GetMultiBinFromSparseFeatures` (col-wise, `None` without a multi-value
/// group) or `GetMultiBinFromAllFeatures` (row-wise).
pub fn share_state_sparse_rate(data: &Dataset, col_wise: bool) -> Option<f64> {
    if col_wise {
        let g = &data.feature_groups()[data.multi_val_group()?];
        let mut sum = 0.0f64;
        for s in &g.subs {
            sum += s.sparse_rate;
        }
        Some(sum / g.subs.len() as f64)
    } else {
        Some(row_wise_sparse_rate(data))
    }
}

#[derive(Debug, Clone)]
enum Vals {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

trait BinVal: Copy + Default + Send + Sync {
    fn of_u32(x: u32) -> Self;
}

macro_rules! bin_val {
    ($($t:ty),*) => {$(
        impl BinVal for $t {
            #[inline(always)]
            fn of_u32(x: u32) -> Self {
                x as $t
            }
        }
    )*};
}
bin_val!(u8, u16, u32);

const FILL_CHUNK: usize = 4096;

type Readers<'a, 'd> = &'a (dyn Fn(usize) -> Vec<ColumnReader<'d>> + Sync);

/// Row-major bins of every column.
fn fill_dense<T: BinVal>(n: usize, nc: usize, readers: Readers<'_, '_>) -> Vec<T> {
    let mut all = vec![T::default(); n * nc];
    if nc > 0 {
        all.par_chunks_mut(FILL_CHUNK * nc).enumerate().for_each(|(c, chunk)| {
            let s = c * FILL_CHUNK;
            let mut it = readers(s);
            for (r, row) in chunk.chunks_exact_mut(nc).enumerate() {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = T::of_u32(it[j].get(s + r));
                }
            }
        });
    }
    all
}

/// Row pointers and the non-most-frequent bins of each row (with offsets).
fn fill_sparse<T: BinVal>(
    n: usize,
    columns: &[Column],
    col_offsets: &[u32],
    readers: Readers<'_, '_>,
) -> (Vec<usize>, Vec<T>) {
    let parts: Vec<(Vec<u32>, Vec<T>)> = (0..n.div_ceil(FILL_CHUNK))
        .into_par_iter()
        .map(|c| {
            let (s, e) = (c * FILL_CHUNK, ((c + 1) * FILL_CHUNK).min(n));
            let mut it = readers(s);
            let mut cnt = Vec::with_capacity(e - s);
            let mut vals = Vec::new();
            for i in s..e {
                let before = vals.len();
                for (j, col) in columns.iter().enumerate() {
                    let b = it[j].get(i);
                    if b == col.most_freq_bin {
                        continue;
                    }
                    vals.push(T::of_u32(b + col_offsets[j] - u32::from(col.most_freq_bin == 0)));
                }
                cnt.push((vals.len() - before) as u32);
            }
            (cnt, vals)
        })
        .collect();
    let mut row_ptr = Vec::with_capacity(n + 1);
    row_ptr.push(0usize);
    let mut all = Vec::with_capacity(parts.iter().map(|p| p.1.len()).sum());
    for (cnt, vals) in parts {
        for c in cnt {
            row_ptr.push(row_ptr.last().unwrap() + c as usize);
        }
        all.extend(vals);
    }
    (row_ptr, all)
}

/// One column of a multi-value bin: a whole group, or one feature of the
/// multi-value group.
#[derive(Debug, Clone, Copy)]
struct Column {
    group: usize,
    sub: Option<usize>,
    most_freq_bin: u32,
}

enum ColumnReader<'a> {
    Group(AnyCursor<'a>),
    Feature(FeatureCursor<'a>),
}

impl ColumnReader<'_> {
    #[inline]
    fn get(&mut self, i: usize) -> u32 {
        match self {
            ColumnReader::Group(c) => c.get(i),
            ColumnReader::Feature(c) => c.get(i),
        }
    }
}

/// upstream `MultiValDenseBin` / `MultiValSparseBin`.
#[derive(Debug, Clone)]
pub struct MultiValBin {
    sparse: bool,
    vals: Vals,
    /// Sparse: start of each row's values (`num_data + 1`).
    row_ptr: Vec<usize>,
    columns: Vec<Column>,
    /// Dense: offset of each column's values in the histogram.
    offsets: Vec<u32>,
    num_data: usize,
    num_bin: usize,
    num_element_per_row: f64,
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

/// upstream `MultiValBinWrapper::InitTrain`: the minimum rows per block.
fn min_block_size(num_bin: usize, num_element_per_row: f64) -> usize {
    let mbs = ((0.3f32 * num_bin as f32) as f64 / (num_element_per_row + K_ZERO_THRESHOLD)) as i32;
    (mbs + 1).clamp(32, 1024) as usize
}

/// What one tree's histogram builds use (upstream `InitTrain` /
/// `CopyMultiValBinSubset`).
#[derive(Debug, Clone)]
pub struct TreePlan {
    min_block_size: usize,
    /// Set when only the by-tree features' columns are built (upstream's
    /// sub-column bin).
    subcol: Option<Subcol>,
}

#[derive(Debug, Clone)]
struct Subcol {
    /// The used columns (only read by a dense bin).
    columns: Vec<usize>,
    /// Bin ranges `(start, len)` that upstream's `HistMove` copies back.
    moves: Vec<(usize, usize)>,
}

impl MultiValBin {
    /// The col-wise multi-value bin of the multi-value group (`None`
    /// without one), or the row-wise bin of all groups.
    ///
    /// upstream: `GetMultiBinFromSparseFeatures` / `GetMultiBinFromAllFeatures`.
    pub fn new(data: &Dataset, offsets: &HistOffsets) -> Option<Self> {
        let groups = data.feature_groups();
        let mut columns = Vec::new();
        let sparse_rate = if offsets.col_wise {
            let g = data.multi_val_group()?;
            for (s, sub) in groups[g].subs.iter().enumerate() {
                columns.push(Column { group: g, sub: Some(s), most_freq_bin: sub.most_freq_bin });
            }
            share_state_sparse_rate(data, true)?
        } else {
            if data.num_features() == 0 {
                return None;
            }
            for (g, grp) in groups.iter().enumerate() {
                if grp.is_multi_val {
                    for (s, sub) in grp.subs.iter().enumerate() {
                        columns.push(Column { group: g, sub: Some(s), most_freq_bin: sub.most_freq_bin });
                    }
                } else {
                    columns.push(Column { group: g, sub: None, most_freq_bin: 0 });
                }
            }
            row_wise_sparse_rate(data)
        };
        let n = data.num_data();
        let nc = columns.len();
        let col_offsets = &offsets.mv_offsets;
        let num_bin = *col_offsets.last().expect("offsets") as usize;
        let readers = |start: usize| -> Vec<ColumnReader<'_>> {
            columns
                .iter()
                .map(|c| match c.sub {
                    None => ColumnReader::Group(groups[c.group].bins[0].any_cursor(start)),
                    Some(s) => ColumnReader::Feature(groups[c.group].feature_bins(s).cursor(start)),
                })
                .collect()
        };
        let sparse = sparse_rate >= MULTI_VAL_BIN_SPARSE_THRESHOLD;
        if sparse {
            let (row_ptr, vals, nnz) = if num_bin <= 256 {
                let (p, v) = fill_sparse::<u8>(n, &columns, col_offsets, &readers);
                let nnz = v.len();
                (p, Vals::U8(v), nnz)
            } else if num_bin <= 65536 {
                let (p, v) = fill_sparse::<u16>(n, &columns, col_offsets, &readers);
                let nnz = v.len();
                (p, Vals::U16(v), nnz)
            } else {
                let (p, v) = fill_sparse::<u32>(n, &columns, col_offsets, &readers);
                let nnz = v.len();
                (p, Vals::U32(v), nnz)
            };
            Some(Self {
                sparse,
                vals,
                row_ptr,
                columns,
                offsets: Vec::new(),
                num_data: n,
                num_bin,
                num_element_per_row: nnz as f64 / n as f64,
            })
        } else {
            let max_bin = col_offsets.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0);
            let vals = if max_bin <= 256 {
                Vals::U8(fill_dense(n, nc, &readers))
            } else if max_bin <= 65536 {
                Vals::U16(fill_dense(n, nc, &readers))
            } else {
                Vals::U32(fill_dense(n, nc, &readers))
            };
            Some(Self {
                sparse,
                vals,
                row_ptr: Vec::new(),
                columns,
                offsets: col_offsets.clone(),
                num_data: n,
                num_bin,
                num_element_per_row: nc as f64,
            })
        }
    }

    pub fn num_bin(&self) -> usize {
        self.num_bin
    }

    /// upstream `MultiValBinWrapper::InitTrain` with `is_feature_used` (by
    /// inner feature; upstream's by-tree mask).
    pub fn plan(&self, data: &Dataset, is_feature_used: &[bool]) -> TreePlan {
        let groups = data.feature_groups();
        let mut contained: Vec<usize> = self.columns.iter().map(|c| c.group).collect();
        contained.dedup();
        let mut sum_used_dense_ratio = 0.0f64;
        let mut sum_dense_ratio = 0.0f64;
        let mut num_used = 0usize;
        let mut total = 0usize;
        let mut used_columns = Vec::new();
        for &g in &contained {
            let grp = &groups[g];
            if grp.is_multi_val {
                for s in &grp.subs {
                    let dense_rate = 1.0 - s.sparse_rate;
                    if is_feature_used[s.inner] {
                        num_used += 1;
                        used_columns.push(total);
                        sum_used_dense_ratio += dense_rate;
                    }
                    sum_dense_ratio += dense_rate;
                    total += 1;
                }
            } else {
                let mut is_group_used = false;
                let mut dense_rate = 0.0f64;
                for s in &grp.subs {
                    if is_feature_used[s.inner] {
                        is_group_used = true;
                    }
                    dense_rate += 1.0 - s.sparse_rate;
                }
                if is_group_used {
                    num_used += 1;
                    used_columns.push(total);
                    sum_used_dense_ratio += dense_rate;
                }
                sum_dense_ratio += dense_rate;
                total += 1;
            }
        }
        const K_SUBFEATURE_THRESHOLD: f64 = 0.6;
        if sum_used_dense_ratio >= sum_dense_ratio * K_SUBFEATURE_THRESHOLD {
            return TreePlan { min_block_size: min_block_size(self.num_bin, self.num_element_per_row), subcol: None };
        }
        let offset = u32::from(self.sparse);
        let mut num_total_bin = offset;
        let mut new_num_total_bin = offset;
        let mut moves = Vec::new();
        for &g in &contained {
            let grp = &groups[g];
            if grp.is_multi_val {
                for (j, s) in grp.subs.iter().enumerate() {
                    if g == 0 && j == 0 && s.most_freq_bin > 0 {
                        num_total_bin = 1;
                    }
                    let mut cur = s.num_bin;
                    if s.most_freq_bin == 0 {
                        cur -= offset;
                    }
                    num_total_bin += cur;
                    if is_feature_used[s.inner] {
                        new_num_total_bin += cur;
                        moves.push(((num_total_bin - cur) as usize, cur as usize));
                    }
                }
            } else {
                let is_group_used = grp.subs.iter().any(|s| is_feature_used[s.inner]);
                let cur = grp.num_total_bin - offset;
                num_total_bin += cur;
                if is_group_used {
                    new_num_total_bin += cur;
                    moves.push(((num_total_bin - cur) as usize, cur as usize));
                }
            }
        }
        let num_element_per_row = if self.sparse { sum_used_dense_ratio } else { num_used as f64 };
        TreePlan {
            min_block_size: min_block_size(new_num_total_bin as usize, num_element_per_row),
            subcol: Some(Subcol { columns: used_columns, moves }),
        }
    }

    /// upstream `MultiValBinWrapper::ConstructHistograms` into `origin` (the
    /// bin's region of a leaf buffer, `2 * num_bin` values): the rows
    /// `indices` (all rows when `None`) are summed block-wise, block 0 into
    /// `origin` (into `scratch` with a sub-column plan) and block `b > 0`
    /// into `bufs[b - 1]`; the blocks are added in block order (`HistMerge`)
    /// and, with a sub-column plan, the used ranges copied to `origin`
    /// (`HistMove`). `gh[i]` is `[grad, hess]` of row `i`.
    #[allow(clippy::too_many_arguments)]
    pub fn construct(
        &self,
        team: &ThreadTeam,
        plan: &TreePlan,
        indices: Option<&[u32]>,
        gh: &[[f32; 2]],
        origin: &mut [f64],
        scratch: &mut Vec<f64>,
        bufs: &mut Vec<Vec<f64>>,
    ) {
        let cnt = indices.map_or(self.num_data, |ix| ix.len());
        let (nblock, bsize) = block_info(team.num_threads(), cnt, plan.min_block_size);
        let len = 2 * self.num_bin;
        assert_eq!(origin.len(), len);
        let cols: Option<&[usize]> = plan.subcol.as_ref().map(|s| s.columns.as_slice());
        if plan.subcol.is_some() {
            scratch.resize(len, 0.0);
        }
        let target: &mut [f64] = if plan.subcol.is_some() { &mut scratch[..len] } else { &mut *origin };
        if nblock == 1 {
            target.fill(0.0);
            self.block(indices, 0, cnt, gh, target, cols);
        } else {
            if bufs.len() < nblock - 1 {
                bufs.resize_with(nblock - 1, Vec::new);
            }
            for b in bufs.iter_mut().take(nblock - 1) {
                b.resize(len, 0.0);
            }
            let targets: Vec<SharedMut<f64>> = std::iter::once(SharedMut::new(&mut *target))
                .chain(bufs.iter_mut().take(nblock - 1).map(|b| SharedMut::new(&mut b[..len])))
                .collect();
            let per_row = cols.map_or(self.num_element_per_row, |c| c.len() as f64) as usize + 1;
            team.for_each(nblock, cnt * per_row + nblock * len, |b| {
                // SAFETY: each block writes only its own buffer.
                let dst = unsafe { targets[b].slice(0, len) };
                dst.fill(0.0);
                let start = (b * bsize).min(cnt);
                let end = (start + bsize).min(cnt);
                if start < end {
                    self.block(indices, start, end, gh, dst, cols);
                }
            });
            drop(targets);
            let ranges: Vec<(usize, usize)> = match &plan.subcol {
                Some(s) => s.moves.iter().map(|&(st, l)| (2 * st, 2 * l)).collect(),
                None => vec![(0, len)],
            };
            merge(team, target, &bufs[..nblock - 1], &ranges);
        }
        if let Some(s) = &plan.subcol {
            for &(st, l) in &s.moves {
                origin[2 * st..2 * (st + l)].copy_from_slice(&scratch[2 * st..2 * (st + l)]);
            }
        }
    }

    fn block(&self, indices: Option<&[u32]>, start: usize, end: usize, gh: &[[f32; 2]], out: &mut [f64], cols: Option<&[usize]>) {
        match &self.vals {
            Vals::U8(v) => self.block_typed(v, indices, start, end, gh, out, cols),
            Vals::U16(v) => self.block_typed(v, indices, start, end, gh, out, cols),
            Vals::U32(v) => self.block_typed(v, indices, start, end, gh, out, cols),
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn block_typed<T: Copy + Into<u32>>(
        &self,
        data: &[T],
        indices: Option<&[u32]>,
        start: usize,
        end: usize,
        gh: &[[f32; 2]],
        out: &mut [f64],
        cols: Option<&[usize]>,
    ) {
        assert!(out.len() >= 2 * self.num_bin && gh.len() >= self.num_data);
        let nc = self.columns.len();
        let offsets = &self.offsets;
        let row_ptr = &self.row_ptr;
        // One loop per layout, so the row body is inlined into it.
        match (self.sparse, cols) {
            (true, _) => rows_loop(indices, start, end, data, gh, |i| row_ptr[i], |i, gi, hi| {
                for &v in &data[row_ptr[i]..row_ptr[i + 1]] {
                    add_gh(out, (v.into() as usize) << 1, gi, hi);
                }
            }),
            (false, None) => rows_loop(indices, start, end, data, gh, |i| i * nc, |i, gi, hi| {
                for (&b, &o) in data[i * nc..(i + 1) * nc].iter().zip(offsets) {
                    add_gh(out, ((b.into() + o) as usize) << 1, gi, hi);
                }
            }),
            (false, Some(cols)) => rows_loop(indices, start, end, data, gh, |i| i * nc, |i, gi, hi| {
                let row = &data[i * nc..(i + 1) * nc];
                for &j in cols {
                    add_gh(out, ((row[j].into() + offsets[j]) as usize) << 1, gi, hi);
                }
            }),
        }
    }
}

#[inline(always)]
fn add_gh(out: &mut [f64], t: usize, gi: f64, hi: f64) {
    debug_assert!(t + 1 < out.len());
    // SAFETY: every stored value plus its offset is below num_bin, and the
    // callers assert `out.len() >= 2 * num_bin`.
    unsafe {
        *out.get_unchecked_mut(t) += gi;
        *out.get_unchecked_mut(t + 1) += hi;
    }
}

/// Rows `start..end` (of `indices`, or the row numbers themselves) in order,
/// calling `body(row, g, h)`; with indices, the row data (at `row_start`) and
/// gradients are prefetched 32 / sizeof(VAL_T) positions ahead, as upstream.
#[inline(always)]
fn rows_loop<T>(
    indices: Option<&[u32]>,
    start: usize,
    end: usize,
    data: &[T],
    gh: &[[f32; 2]],
    row_start: impl Fn(usize) -> usize,
    mut body: impl FnMut(usize, f64, f64),
) {
    let mut row = |i: usize| {
        // SAFETY: i < num_data <= gh.len() (asserted by the callers).
        let [g, h] = unsafe { *gh.get_unchecked(i) };
        body(i, g as f64, h as f64);
    };
    match indices {
        Some(ix) => {
            let ix = &ix[start..end];
            let pf = 32 / std::mem::size_of::<T>();
            let split = ix.len().saturating_sub(pf);
            for k in 0..split {
                let p = ix[k + pf] as usize;
                prefetch(data.as_ptr().wrapping_add(row_start(p)));
                prefetch(gh.as_ptr().wrapping_add(p));
                row(ix[k] as usize);
            }
            ix[split..].iter().for_each(|&i| row(i as usize));
        }
        None => (start..end).for_each(row),
    }
}

impl MultiValBin {
    /// [`MultiValBin::construct`] with packed integer gradients and hessians
    /// (`gh[row]`, upstream `ConstructHistogramsInt32`): one `i64` per bin in
    /// `origin` (`num_bin` values). Every column is summed; integer sums do
    /// not depend on the block layout.
    pub fn construct_int(
        &self,
        team: &ThreadTeam,
        indices: Option<&[u32]>,
        gh: &[i64],
        origin: &mut [i64],
        bufs: &mut Vec<Vec<i64>>,
    ) {
        let cnt = indices.map_or(self.num_data, |ix| ix.len());
        let (nblock, bsize) = block_info(team.num_threads(), cnt, min_block_size(self.num_bin, self.num_element_per_row));
        let len = self.num_bin;
        assert_eq!(origin.len(), len);
        origin.fill(0);
        if nblock == 1 {
            self.block_int(indices, 0, cnt, gh, origin);
            return;
        }
        if bufs.len() < nblock - 1 {
            bufs.resize_with(nblock - 1, Vec::new);
        }
        for b in bufs.iter_mut().take(nblock - 1) {
            b.clear();
            b.resize(len, 0);
        }
        let targets: Vec<SharedMut<i64>> = std::iter::once(SharedMut::new(&mut *origin))
            .chain(bufs.iter_mut().take(nblock - 1).map(|b| SharedMut::new(&mut b[..len])))
            .collect();
        let per_row = self.num_element_per_row as usize + 1;
        team.for_each(nblock, cnt * per_row + nblock * len, |b| {
            // SAFETY: each block writes only its own buffer.
            let dst = unsafe { targets[b].slice(0, len) };
            let start = (b * bsize).min(cnt);
            let end = (start + bsize).min(cnt);
            if start < end {
                self.block_int(indices, start, end, gh, dst);
            }
        });
        drop(targets);
        for src in &bufs[..nblock - 1] {
            for (o, v) in origin.iter_mut().zip(src) {
                *o = o.wrapping_add(*v);
            }
        }
    }

    fn block_int(&self, indices: Option<&[u32]>, start: usize, end: usize, gh: &[i64], out: &mut [i64]) {
        match &self.vals {
            Vals::U8(v) => self.block_int_typed(v, indices, start, end, gh, out),
            Vals::U16(v) => self.block_int_typed(v, indices, start, end, gh, out),
            Vals::U32(v) => self.block_int_typed(v, indices, start, end, gh, out),
        }
    }

    fn block_int_typed<T: Copy + Into<u32>>(
        &self,
        data: &[T],
        indices: Option<&[u32]>,
        start: usize,
        end: usize,
        gh: &[i64],
        out: &mut [i64],
    ) {
        let nc = self.columns.len();
        let mut add_row = |i: usize| {
            let v = gh[i];
            if self.sparse {
                for &b in &data[self.row_ptr[i]..self.row_ptr[i + 1]] {
                    let t = &mut out[b.into() as usize];
                    *t = t.wrapping_add(v);
                }
            } else {
                for (&b, &o) in data[i * nc..(i + 1) * nc].iter().zip(&self.offsets) {
                    let t = &mut out[(b.into() + o) as usize];
                    *t = t.wrapping_add(v);
                }
            }
        };
        match indices {
            Some(ix) => ix[start..end].iter().for_each(|&i| add_row(i as usize)),
            None => (start..end).for_each(add_row),
        }
    }
}

/// upstream `HistMerge`: `dst += bufs[0] + bufs[1] + ...` (in that order)
/// over the f64 ranges `(start, len)`, split into bin blocks across threads.
fn merge(team: &ThreadTeam, dst: &mut [f64], bufs: &[Vec<f64>], ranges: &[(usize, usize)]) {
    const PIECE: usize = 1 << 12;
    let pieces: Vec<(usize, usize)> = ranges
        .iter()
        .flat_map(|&(s, l)| (0..l.div_ceil(PIECE)).map(move |k| (s + k * PIECE, PIECE.min(l - k * PIECE))))
        .collect();
    let total: usize = ranges.iter().map(|r| r.1).sum();
    let d = SharedMut::new(dst);
    team.for_each(pieces.len(), total * bufs.len(), |p| {
        let (s, l) = pieces[p];
        // SAFETY: pieces are disjoint.
        let out = unsafe { d.slice(s, l) };
        for src in bufs {
            for (o, v) in out.iter_mut().zip(&src[s..s + l]) {
                *o += *v;
            }
        }
    });
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

    #[test]
    fn min_block_size_matches_upstream() {
        assert_eq!(min_block_size(1000, 10.0), 32);
        assert_eq!(min_block_size(100_000, 2.0), 1024);
        assert_eq!(min_block_size(5000, 3.0), 501);
    }
}
