//! Row indices grouped by leaf.
//!
//! upstream: src/treelearner/data_partition.hpp. The split is stable: rows
//! keep their relative order within each child, which keeps histogram
//! summation order identical to upstream. Large leaves are split in parallel
//! blocks whose left/right parts are concatenated in block order (upstream
//! `ParallelPartitionRunner`), which yields the same stable order.

use crate::dataset::BinColumn;
use crate::threading::{SharedMut, ThreadTeam};

/// Leaves smaller than this are split on the calling thread.
const PAR_MIN_ROWS: usize = 1 << 14;
const PAR_MIN_BLOCK: usize = 1 << 12;

#[derive(Debug, Clone)]
pub struct DataPartition {
    num_data: usize,
    indices: Vec<u32>,
    leaf_begin: Vec<usize>,
    leaf_count: Vec<usize>,
    num_leaves: usize,
    left_buf: Vec<u32>,
    right_buf: Vec<u32>,
    /// In-bag rows (ascending) when bagging; `None` uses every row.
    used: Option<Vec<u32>>,
}

#[derive(Clone, Copy)]
struct SyncPtr(*mut f64);
// SAFETY: only used to write disjoint rows (each row belongs to one leaf).
unsafe impl Send for SyncPtr {}
unsafe impl Sync for SyncPtr {}

impl DataPartition {
    pub fn new(num_data: usize, max_leaves: usize) -> Self {
        Self {
            num_data,
            indices: (0..num_data as u32).collect(),
            leaf_begin: vec![0; max_leaves],
            leaf_count: vec![0; max_leaves],
            num_leaves: 1,
            left_buf: Vec::new(),
            right_buf: Vec::new(),
            used: None,
        }
    }

    /// upstream: `DataPartition::SetUsedDataIndices`; takes effect at the next [`init`](Self::init).
    pub fn set_used_data_indices(&mut self, used: Option<&[u32]>) {
        match used {
            Some(u) => {
                let v = self.used.get_or_insert_with(Vec::new);
                v.clear();
                v.extend_from_slice(u);
            }
            None => self.used = None,
        }
    }

    pub fn init(&mut self) {
        self.leaf_begin.iter_mut().for_each(|x| *x = 0);
        self.leaf_count.iter_mut().for_each(|x| *x = 0);
        match &self.used {
            None => {
                for (i, v) in self.indices.iter_mut().enumerate() {
                    *v = i as u32;
                }
                self.leaf_count[0] = self.num_data;
            }
            Some(u) => {
                self.indices[..u.len()].copy_from_slice(u);
                self.leaf_count[0] = u.len();
            }
        }
        self.num_leaves = 1;
    }

    pub fn leaf_count(&self, leaf: usize) -> usize {
        self.leaf_count[leaf]
    }

    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    pub fn indices_on_leaf(&self, leaf: usize) -> &[u32] {
        let b = self.leaf_begin[leaf];
        &self.indices[b..b + self.leaf_count[leaf]]
    }

    /// Split `leaf` on a binned feature; rows whose bin `b` has `lut[b]`
    /// keep `leaf`, the others become `right_leaf`.
    pub fn split(&mut self, team: &ThreadTeam, leaf: usize, col: &BinColumn, lut: &[bool], right_leaf: usize) {
        let begin = self.leaf_begin[leaf];
        let cnt = self.leaf_count[leaf];
        let left_cnt = match col {
            BinColumn::U8(v) => self.split_typed(team, v, begin, cnt, lut),
            BinColumn::U16(v) => self.split_typed(team, v, begin, cnt, lut),
            BinColumn::U32(v) => self.split_typed(team, v, begin, cnt, lut),
        };
        self.leaf_count[leaf] = left_cnt;
        self.leaf_begin[right_leaf] = begin + left_cnt;
        self.leaf_count[right_leaf] = cnt - left_cnt;
        self.num_leaves += 1;
    }

    fn split_typed<T: Copy + Into<u32> + Sync>(
        &mut self,
        team: &ThreadTeam,
        bins: &[T],
        begin: usize,
        cnt: usize,
        lut: &[bool],
    ) -> usize {
        let slice = &mut self.indices[begin..begin + cnt];
        let threads = team.num_threads();
        self.left_buf.resize(cnt.max(self.left_buf.len()), 0);
        self.right_buf.resize(cnt.max(self.right_buf.len()), 0);
        if cnt < PAR_MIN_ROWS || threads <= 1 {
            let (nl, nr) = split_block(bins, lut, slice, &mut self.left_buf[..cnt], &mut self.right_buf[..cnt]);
            slice[..nl].copy_from_slice(&self.left_buf[..nl]);
            slice[nl..].copy_from_slice(&self.right_buf[..nr]);
            return nl;
        }
        let block = cnt.div_ceil(threads).max(PAR_MIN_BLOCK);
        let nblock = cnt.div_ceil(block);
        let lbuf = SharedMut::new(&mut self.left_buf[..cnt]);
        let rbuf = SharedMut::new(&mut self.right_buf[..cnt]);
        let counts: Vec<(usize, usize)> = {
            let src_all: &[u32] = slice;
            team.map(nblock, cnt, |b| {
                let start = b * block;
                let len = block.min(cnt - start);
                // SAFETY: block `b` writes only `[start, start + len)` of each buffer.
                let (l, r) = unsafe { (lbuf.slice(start, len), rbuf.slice(start, len)) };
                split_block(bins, lut, &src_all[start..start + len], l, r)
            })
        };
        // upstream: ParallelPartitionRunner::Run — concatenate the blocks'
        // left parts, then their right parts, both in block order.
        let left_cnt: usize = counts.iter().map(|c| c.0).sum();
        let mut lpos = Vec::with_capacity(nblock);
        let mut rpos = Vec::with_capacity(nblock);
        let (mut l, mut r) = (0usize, left_cnt);
        for &(nl, nr) in &counts {
            lpos.push(l);
            rpos.push(r);
            l += nl;
            r += nr;
        }
        let dst = SharedMut::new(slice);
        team.for_each(nblock, cnt, |b| {
            let (nl, nr) = counts[b];
            let start = b * block;
            // SAFETY: destination ranges are disjoint by construction; the
            // buffers are only read here.
            unsafe {
                dst.slice(lpos[b], nl).copy_from_slice(lbuf.slice(start, nl));
                dst.slice(rpos[b], nr).copy_from_slice(rbuf.slice(start, nr));
            }
        });
        left_cnt
    }

    /// Add each leaf's output to the scores of its rows
    /// (upstream `SerialTreeLearner::AddPredictionToScore`).
    pub fn add_leaf_outputs(&self, team: &ThreadTeam, leaf_values: &[f64], score: &mut [f64]) {
        assert!(score.len() >= self.num_data);
        let ptr = SyncPtr(score.as_mut_ptr());
        let leaves = self.num_leaves.min(leaf_values.len());
        team.for_each(leaves, self.num_data, |leaf| {
            let p = ptr;
            let v = leaf_values[leaf];
            for &i in self.indices_on_leaf(leaf) {
                // SAFETY: i < num_data <= score.len(), and every row index
                // appears in exactly one leaf, so writes never alias.
                unsafe { *p.0.add(i as usize) += v };
            }
        });
    }
}

/// Stable branchless split of `src` into `l` (rows going left) and `r`.
#[inline(always)]
fn split_block<T: Copy + Into<u32>>(bins: &[T], lut: &[bool], src: &[u32], l: &mut [u32], r: &mut [u32]) -> (usize, usize) {
    let (mut nl, mut nr) = (0usize, 0usize);
    assert!(l.len() >= src.len() && r.len() >= src.len());
    for &i in src {
        let go_left = lut[bins[i as usize].into() as usize];
        // SAFETY: nl, nr <= number of rows seen so far < src.len() <= l.len(), r.len().
        unsafe {
            *l.get_unchecked_mut(nl) = i;
            *r.get_unchecked_mut(nr) = i;
        }
        nl += go_left as usize;
        nr += !go_left as usize;
    }
    (nl, nr)
}
