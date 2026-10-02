//! Feature groups: the bins of bundled features share one stored column.
//!
//! upstream: include/LightGBM/feature_group.h (`FeatureGroup`: the two
//! constructors, `PushData`, `SubFeatureIterator`, `Split`, `CopySubrow`),
//! and the push loops of src/c_api.cpp (`LGBM_DatasetCreateFromMats`,
//! `LGBM_DatasetCreateFromCSR`, `LGBM_DatasetCreateFromCSC`) and
//! include/LightGBM/dataset.h (`PushOneRow`, `FinishOneRow`).
//!
//! A non-multi-value group stores, per row, `bin_offsets[sub] + bin - (mfb == 0)`
//! of the one feature whose bin is not its most frequent bin, or 0. When two
//! features of a group both leave their most frequent bin in a row (EFB
//! allows a few such conflicts), the value pushed last wins and the other
//! feature reads its most frequent bin, as upstream. A multi-value group
//! stores each feature in its own bin (`bin - (mfb == 0) + 1`, or 0).

use rayon::prelude::*;

use crate::bin::{Bin, BinKind, DenseWriter, RawCursor};
use crate::binning::BinMapper;
use crate::consts::K_SPARSE_THRESHOLD;

/// upstream `MultiValBin::multi_val_bin_sparse_threshold` (`0.25f`).
pub const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25f32 as f64;

/// The bin-mapper facts a group needs about one of its features.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SubFeature {
    /// This engine's inner feature index (column order).
    pub inner: usize,
    pub num_bin: u32,
    pub most_freq_bin: u32,
    pub default_bin: u32,
    pub sparse_rate: f64,
}

impl SubFeature {
    pub fn of(inner: usize, m: &BinMapper) -> Self {
        Self {
            inner,
            num_bin: m.num_bin as u32,
            most_freq_bin: m.most_freq_bin,
            default_bin: m.default_bin,
            sparse_rate: m.sparse_rate,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureGroup {
    pub(crate) subs: Vec<SubFeature>,
    pub(crate) is_multi_val: bool,
    pub(crate) is_dense_multi_val: bool,
    pub(crate) is_sparse: bool,
    pub(crate) bin_offsets: Vec<u32>,
    pub(crate) num_total_bin: u32,
    /// One bin, or one per feature of a multi-value group.
    pub(crate) bins: Vec<Bin>,
}

impl FeatureGroup {
    /// upstream `FeatureGroup(num_feature, is_multi_val, bin_mappers, num_data, group_id)`
    /// (the bins are left empty for the builder).
    pub fn new(subs: Vec<SubFeature>, is_multi_val: bool, group_id: usize) -> Self {
        let n = subs.len();
        let mut sum_sparse_rate = 0.0f64;
        for s in &subs {
            sum_sparse_rate += s.sparse_rate;
        }
        sum_sparse_rate /= n as f64;
        let mut offset = 1u32;
        let mut is_dense_multi_val = false;
        if sum_sparse_rate < MULTI_VAL_BIN_SPARSE_THRESHOLD && is_multi_val {
            offset = 0;
            is_dense_multi_val = true;
        }
        let mut total = offset;
        if group_id == 0 && n > 0 && is_dense_multi_val && subs[0].most_freq_bin > 0 {
            total = 1;
        }
        let mut bin_offsets = vec![total];
        for s in &subs {
            let mut nb = s.num_bin;
            if s.most_freq_bin == 0 {
                nb -= offset;
            }
            total += nb;
            bin_offsets.push(total);
        }
        Self { subs, is_multi_val, is_dense_multi_val, is_sparse: false, bin_offsets, num_total_bin: total, bins: Vec::new() }
    }

    /// upstream `FeatureGroup(bin_mappers, num_data)`: one feature, stored
    /// sparsely when its sparse rate is at least `kSparseThreshold`
    /// (`Dataset::CreateValid`).
    pub fn new_single(sub: SubFeature) -> Self {
        let mut total = 1u32;
        let mut nb = sub.num_bin;
        if sub.most_freq_bin == 0 {
            nb -= 1;
        }
        total += nb;
        Self {
            is_sparse: sub.sparse_rate >= K_SPARSE_THRESHOLD,
            subs: vec![sub],
            is_multi_val: false,
            is_dense_multi_val: false,
            bin_offsets: vec![1, total],
            num_total_bin: total,
            bins: Vec::new(),
        }
    }

    /// upstream `CreateBinData`: the storage kind of each bin.
    pub fn bin_kinds(&self) -> Vec<BinKind> {
        if self.is_multi_val {
            self.subs
                .iter()
                .map(|s| {
                    let nb = s.num_bin + u32::from(s.most_freq_bin != 0);
                    if s.sparse_rate >= K_SPARSE_THRESHOLD { BinKind::sparse(nb) } else { BinKind::dense(nb) }
                })
                .collect()
        } else if self.is_sparse {
            vec![BinKind::sparse(self.num_total_bin)]
        } else {
            vec![BinKind::dense(self.num_total_bin)]
        }
    }

    /// Number of distinct stored values of bin `k` (all below this).
    pub fn bin_value_count(&self, k: usize) -> u32 {
        if self.is_multi_val {
            let s = &self.subs[k];
            s.num_bin + u32::from(s.most_freq_bin != 0)
        } else {
            self.num_total_bin
        }
    }

    pub fn num_feature(&self) -> usize {
        self.subs.len()
    }

    /// The stored value upstream `PushData` writes for bin `bin` of feature
    /// `sub`, or `None` for its most frequent bin (not pushed).
    #[inline(always)]
    pub fn push_value(&self, sub: usize, bin: u32) -> Option<u32> {
        let s = &self.subs[sub];
        if bin == s.most_freq_bin {
            return None;
        }
        let b = bin - u32::from(s.most_freq_bin == 0);
        Some(if self.is_multi_val { b + 1 } else { b + self.bin_offsets[sub] })
    }

    /// Feature `sub`'s bins (upstream `SubFeatureIterator`).
    pub fn feature_bins(&self, sub: usize) -> FeatureBins<'_> {
        let s = &self.subs[sub];
        let (k, min_bin, max_bin) = if self.is_multi_val {
            (sub, 1, s.num_bin - 1 + u32::from(s.most_freq_bin != 0))
        } else {
            (0, self.bin_offsets[sub], self.bin_offsets[sub + 1] - 1)
        };
        FeatureBins {
            bin: &self.bins[k],
            min_bin,
            max_bin,
            most_freq_bin: s.most_freq_bin,
            offset: u32::from(s.most_freq_bin == 0),
            value_count: self.bin_value_count(k),
        }
    }

    /// upstream `FeatureGroup::CopySubrow`.
    pub fn copy_subrow(&self, used: &[u32]) -> Self {
        Self { bins: self.bins.iter().map(|b| b.copy_subrow(used)).collect(), ..self.clone_layout() }
    }

    fn clone_layout(&self) -> Self {
        Self {
            subs: self.subs.clone(),
            is_multi_val: self.is_multi_val,
            is_dense_multi_val: self.is_dense_multi_val,
            is_sparse: self.is_sparse,
            bin_offsets: self.bin_offsets.clone(),
            num_total_bin: self.num_total_bin,
            bins: Vec::new(),
        }
    }
}

/// The stored bins of one feature and how to read its bin from them.
#[derive(Clone, Copy)]
pub struct FeatureBins<'a> {
    pub bin: &'a Bin,
    pub min_bin: u32,
    pub max_bin: u32,
    pub most_freq_bin: u32,
    offset: u32,
    value_count: u32,
}

impl<'a> FeatureBins<'a> {
    /// upstream `BinIterator::Get`: the feature's bin from a stored value.
    #[inline(always)]
    pub fn decode(&self, raw: u32) -> u32 {
        if raw >= self.min_bin && raw <= self.max_bin { raw - self.min_bin + self.offset } else { self.most_freq_bin }
    }

    /// A forward reader of the feature's bins from row `start` on.
    pub fn cursor(&self, start: usize) -> FeatureCursor<'a> {
        FeatureCursor { raw: self.bin.any_cursor(start), view: *self }
    }

    /// `feature_lut[bin]` for every value the bin can store.
    pub fn raw_lut(&self, feature_lut: &[bool]) -> Vec<bool> {
        (0..self.value_count).map(|raw| feature_lut[self.decode(raw) as usize]).collect()
    }
}

pub struct FeatureCursor<'a> {
    raw: crate::bin::AnyCursor<'a>,
    view: FeatureBins<'a>,
}

impl FeatureCursor<'_> {
    #[inline]
    pub fn get(&mut self, i: usize) -> u32 {
        self.view.decode(self.raw.get(i))
    }
}

/// Rows per parallel task when pushing (even, so 4-bit bins split cleanly).
const PUSH_CHUNK: usize = 1 << 14;

#[derive(Debug, Clone, Copy)]
struct FeatPush {
    group: u32,
    sub: u32,
    slot: u32,
}

enum Slot {
    Dense(Bin),
    Sparse { kind: BinKind, pairs: Vec<(u32, u32)> },
}

/// Fills the bins of a set of feature groups with upstream's push semantics.
pub struct GroupsBuilder<'a> {
    groups: Vec<FeatureGroup>,
    mappers: Vec<&'a BinMapper>,
    feats: Vec<FeatPush>,
    slots: Vec<Slot>,
    /// Inner features of each slot in ascending (column) order.
    slot_features: Vec<Vec<usize>>,
    /// upstream `feature_need_push_zeros_` (upstream inner order).
    need_push_zeros: Vec<usize>,
    num_data: usize,
}

impl<'a> GroupsBuilder<'a> {
    /// `groups` in upstream group order; `mappers[inner]` is each inner feature's mapper.
    pub fn new(groups: Vec<FeatureGroup>, mappers: Vec<&'a BinMapper>, num_data: usize) -> Self {
        let nf = mappers.len();
        let mut feats = vec![FeatPush { group: 0, sub: 0, slot: 0 }; nf];
        let mut slots = Vec::new();
        let mut slot_features = Vec::new();
        let mut need_push_zeros = Vec::new();
        for (g, grp) in groups.iter().enumerate() {
            let kinds = grp.bin_kinds();
            let first = slots.len();
            for &k in &kinds {
                slots.push(if k.is_sparse() {
                    Slot::Sparse { kind: k, pairs: Vec::new() }
                } else {
                    Slot::Dense(Bin::dense_zeros(k, num_data))
                });
                slot_features.push(Vec::new());
            }
            for (s, sub) in grp.subs.iter().enumerate() {
                let slot = first + if grp.is_multi_val { s } else { 0 };
                feats[sub.inner] = FeatPush { group: g as u32, sub: s as u32, slot: slot as u32 };
                slot_features[slot].push(sub.inner);
                if sub.default_bin != sub.most_freq_bin {
                    need_push_zeros.push(sub.inner);
                }
            }
        }
        for f in &mut slot_features {
            f.sort_unstable();
        }
        Self { groups, mappers, feats, slots, slot_features, need_push_zeros, num_data }
    }

    #[inline(always)]
    fn push_value(&self, f: usize, value: f64) -> Option<u32> {
        let p = self.feats[f];
        self.groups[p.group as usize].push_value(p.sub as usize, self.mappers[f].value_to_bin(value))
    }

    /// Column-wise pushes (dense matrices and CSC input): `column(f, rows,
    /// sink)` calls `sink(row, value)` for the pushes of inner feature `f` in
    /// `rows`, in push order. Within a group, features are pushed in column
    /// order. With `chunked`, each group's rows are split across threads
    /// (`column` must then honour `rows`); otherwise `rows` is all rows.
    pub fn push_columns(
        &mut self,
        chunked: bool,
        column: impl Fn(usize, std::ops::Range<usize>, &mut dyn FnMut(usize, f64)) + Sync,
    ) {
        self.push_columns_with(chunked, column, |this, f, v| this.push_value(f, v));
    }

    /// Like [`GroupsBuilder::push_columns`], pushing bins instead of values.
    pub fn push_bin_columns(&mut self, column: impl Fn(usize, &mut dyn FnMut(usize, u32)) + Sync) {
        self.push_columns_with(
            false,
            |f, _rows, sink| column(f, sink),
            |this, f, bin| {
                let p = this.feats[f];
                this.groups[p.group as usize].push_value(p.sub as usize, bin)
            },
        );
    }

    fn push_columns_with<V>(
        &mut self,
        chunked: bool,
        column: impl Fn(usize, std::ops::Range<usize>, &mut dyn FnMut(usize, V)) + Sync,
        to_raw: impl Fn(&Self, usize, V) -> Option<u32> + Sync,
    ) {
        let n = self.num_data;
        let this = &*self;
        let filled: Vec<Option<Vec<(u32, u32)>>> = (0..self.slots.len())
            .into_par_iter()
            .map(|slot| match &this.slots[slot] {
                Slot::Dense(_) => None,
                Slot::Sparse { .. } => {
                    let feats = &this.slot_features[slot];
                    let (nchunk, step) = if chunked { (n.div_ceil(PUSH_CHUNK), PUSH_CHUNK) } else { (1, n) };
                    let ranges: Vec<std::ops::Range<usize>> =
                        (0..nchunk).map(|c| c * step..((c + 1) * step).min(n)).collect();
                    let parts: Vec<Vec<(u32, u32)>> = ranges
                        .into_par_iter()
                        .map(|rows| {
                            let mut out = Vec::new();
                            for &f in feats {
                                column(f, rows.clone(), &mut |r, v| {
                                    if let Some(raw) = to_raw(this, f, v) {
                                        out.push((r as u32, raw));
                                    }
                                });
                            }
                            out
                        })
                        .collect();
                    Some(parts.concat())
                }
            })
            .collect();
        let writers: Vec<Option<DenseWriter>> = self
            .slots
            .iter_mut()
            .map(|s| match s {
                Slot::Dense(b) => Some(DenseWriter::new(b)),
                Slot::Sparse { .. } => None,
            })
            .collect();
        let this = &*self;
        let tasks: Vec<(usize, std::ops::Range<usize>)> = (0..this.slots.len())
            .filter(|&s| writers[s].is_some())
            .flat_map(|s| {
                let chunks = if chunked { n.div_ceil(PUSH_CHUNK) } else { 1 };
                (0..chunks).map(move |c| {
                    if chunked { (s, c * PUSH_CHUNK..((c + 1) * PUSH_CHUNK).min(n)) } else { (s, 0..n) }
                })
            })
            .collect();
        tasks.into_par_iter().for_each(|(slot, rows)| {
            let w = writers[slot].as_ref().expect("dense slot");
            for &f in &this.slot_features[slot] {
                column(f, rows.clone(), &mut |r, v| {
                    if let Some(raw) = to_raw(this, f, v) {
                        // SAFETY: tasks of one slot cover disjoint, even-aligned row ranges.
                        unsafe { w.set(r, raw) };
                    }
                });
            }
        });
        drop(writers);
        for (slot, p) in filled.into_iter().enumerate() {
            if let (Some(p), Slot::Sparse { pairs, .. }) = (p, &mut self.slots[slot]) {
                *pairs = p;
            }
        }
    }

    /// Row-wise pushes (CSR and text input, upstream `PushOneRow` with
    /// pairs): rows `row0..row0 + nrows`, where `row(k, sink)` calls
    /// `sink(column, value)` for the stored entries of row `row0 + k` in
    /// order. Columns at or past `real_to_inner.len()` and unused columns are
    /// skipped; afterwards the features in `feature_need_push_zeros_` that
    /// the row did not set are pushed a 0 (`FinishOneRow`). `row0` must be even.
    pub fn push_rows(
        &mut self,
        row0: usize,
        nrows: usize,
        real_to_inner: &[Option<usize>],
        row: impl Fn(usize, &mut dyn FnMut(usize, f64)) + Sync,
    ) {
        assert!(row0 % 2 == 0 && row0 + nrows <= self.num_data);
        let writers: Vec<Option<DenseWriter>> = self
            .slots
            .iter_mut()
            .map(|s| match s {
                Slot::Dense(b) => Some(DenseWriter::new(b)),
                Slot::Sparse { .. } => None,
            })
            .collect();
        let this = &*self;
        let nf = this.feats.len();
        let parts: Vec<Vec<(u32, u32, u32)>> = (0..nrows.div_ceil(PUSH_CHUNK))
            .into_par_iter()
            .map(|c| {
                let mut sparse = Vec::new();
                let mut added = vec![false; nf];
                let mut touched: Vec<usize> = Vec::new();
                for k in c * PUSH_CHUNK..((c + 1) * PUSH_CHUNK).min(nrows) {
                    let r = row0 + k;
                    let mut push = |f: usize, v: f64| {
                        if let Some(raw) = this.push_value(f, v) {
                            let slot = this.feats[f].slot as usize;
                            match &writers[slot] {
                                // SAFETY: tasks cover disjoint, even-aligned row ranges.
                                Some(w) => unsafe { w.set(r, raw) },
                                None => sparse.push((slot as u32, r as u32, raw)),
                            }
                        }
                    };
                    row(k, &mut |col, v| {
                        if let Some(&Some(f)) = real_to_inner.get(col) {
                            if !added[f] {
                                added[f] = true;
                                touched.push(f);
                            }
                            push(f, v);
                        }
                    });
                    for &f in &this.need_push_zeros {
                        if !added[f] {
                            push(f, 0.0);
                        }
                    }
                    for f in touched.drain(..) {
                        added[f] = false;
                    }
                }
                sparse
            })
            .collect();
        drop(writers);
        let mut counts = vec![0usize; self.slots.len()];
        for p in &parts {
            for &(s, _, _) in p {
                counts[s as usize] += 1;
            }
        }
        for (s, slot) in self.slots.iter_mut().enumerate() {
            if let Slot::Sparse { pairs, .. } = slot {
                pairs.reserve(counts[s]);
            }
        }
        for p in parts {
            for (s, r, v) in p {
                if let Slot::Sparse { pairs, .. } = &mut self.slots[s as usize] {
                    pairs.push((r, v));
                }
            }
        }
    }

    /// upstream `FeatureGroup::FinishLoad`: the filled groups.
    pub fn finish(self) -> Vec<FeatureGroup> {
        let n = self.num_data;
        let mut bins: Vec<Bin> = self
            .slots
            .into_par_iter()
            .map(|s| match s {
                Slot::Dense(b) => b,
                Slot::Sparse { kind, pairs } => Bin::sparse_from_pairs(kind, n, pairs),
            })
            .collect();
        let mut groups = self.groups;
        let mut rest = bins.drain(..);
        for g in &mut groups {
            let k = if g.is_multi_val { g.subs.len() } else { 1 };
            g.bins = rest.by_ref().take(k).collect();
        }
        groups
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binning::{BinParams, BinType};

    fn mapper(vals: &[f64], n: usize) -> BinMapper {
        let params = BinParams {
            max_bin: 255,
            min_data_in_bin: 1,
            min_split_data: 1,
            pre_filter: false,
            use_missing: true,
            zero_as_missing: false,
        };
        BinMapper::find_bin(vals, n, BinType::Numerical, &params).unwrap()
    }

    #[test]
    fn group_offsets_follow_upstream() {
        let sub = |inner, num_bin, mfb, sparse_rate| SubFeature {
            inner,
            num_bin,
            most_freq_bin: mfb,
            default_bin: 0,
            sparse_rate,
        };
        // bundled: slot 0 shared, mfb == 0 features drop their bin 0
        let g = FeatureGroup::new(vec![sub(0, 4, 0, 0.9), sub(1, 3, 1, 0.9)], false, 3);
        assert_eq!(g.bin_offsets, vec![1, 4, 7]);
        assert_eq!(g.push_value(0, 2), Some(2));
        assert_eq!(g.push_value(1, 0), Some(4));
        assert_eq!(g.push_value(1, 1), None);
        let fb_kinds = g.bin_kinds();
        assert_eq!(fb_kinds, vec![BinKind::Dense4]);
        // dense multi-value group 0 whose first feature has mfb > 0 keeps a leading bin
        let g = FeatureGroup::new(vec![sub(0, 4, 2, 0.1), sub(1, 3, 0, 0.1)], true, 0);
        assert!(g.is_dense_multi_val);
        assert_eq!(g.bin_offsets, vec![1, 5, 8]);
        let g = FeatureGroup::new(vec![sub(0, 4, 2, 0.1), sub(1, 3, 0, 0.1)], true, 1);
        assert_eq!(g.bin_offsets, vec![0, 4, 7]);
        let g = FeatureGroup::new_single(sub(0, 300, 0, 0.8));
        assert_eq!(g.bin_kinds(), vec![BinKind::Sparse16]);
    }

    #[test]
    fn conflicts_keep_the_last_push() {
        // two features with mfb = bin of 0.0; rows 2 and 3 conflict
        let a = mapper(&[1.0, 2.0], 10);
        let b = mapper(&[5.0, 6.0], 10);
        let ga = SubFeature::of(0, &a);
        let gb = SubFeature::of(1, &b);
        let group = FeatureGroup::new(vec![ga, gb], false, 0);
        let mut bld = GroupsBuilder::new(vec![group], vec![&a, &b], 4);
        let x = [[1.0, 0.0], [0.0, 6.0], [2.0, 5.0], [1.0, 6.0]];
        bld.push_columns(true, |f, rows, sink| {
            for r in rows {
                sink(r, x[r][f]);
            }
        });
        let g = &bld.finish()[0];
        let fa = g.feature_bins(0);
        let fb = g.feature_bins(1);
        let mut ca = fa.cursor(0);
        let mut cb = fb.cursor(0);
        let got: Vec<(u32, u32)> = (0..4).map(|i| (ca.get(i), cb.get(i))).collect();
        let (za, zb) = (a.value_to_bin(0.0), b.value_to_bin(0.0));
        assert_eq!(
            got,
            vec![
                (a.value_to_bin(1.0), zb),
                (za, b.value_to_bin(6.0)),
                (za, b.value_to_bin(5.0)),
                (za, b.value_to_bin(6.0)),
            ]
        );
    }
}
