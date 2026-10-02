//! Stored bins of one feature group (or of one feature of a multi-value
//! group): dense with 4, 8, 16 or 32 bits per row, or sparse (delta-coded
//! non-zero entries).
//!
//! upstream: src/io/dense_bin.hpp (`DenseBin`, including the 4-bit packing),
//! src/io/sparse_bin.hpp (`SparseBin`, `SparseBinIterator`, `LoadFromPair`,
//! `GetFastIndex`, `CopySubrow`), src/io/bin.cpp (`Bin::CreateDenseBin`,
//! `Bin::CreateSparseBin`).
//!
//! Stored value 0 is the group's most-frequent-bin slot: dense bins hold it
//! explicitly, sparse bins leave it out.

use crate::stdsort;

/// The integer type of stored values.
pub trait BinVal: Copy + Default + Send + Sync + Into<u32> + 'static {
    const BYTES: usize;
    fn from_u32(v: u32) -> Self;
    fn put_le(self, out: &mut Vec<u8>);
    fn get_le(b: &[u8]) -> Self;
}

macro_rules! bin_val {
    ($t:ty, $n:expr) => {
        impl BinVal for $t {
            const BYTES: usize = $n;
            #[inline(always)]
            fn from_u32(v: u32) -> Self {
                v as $t
            }
            fn put_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn get_le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().expect("width checked"))
            }
        }
    };
}
bin_val!(u8, 1);
bin_val!(u16, 2);
bin_val!(u32, 4);

/// Forward-only reader of stored values (upstream `BinIterator::RawGet`):
/// after `RawBins::cursor(start)`, rows must be read in non-decreasing order
/// from `start` on.
pub trait RawCursor {
    fn get(&mut self, i: usize) -> u32;
}

/// Bin storage that can be read row by row.
pub trait RawBins: Sync {
    type Cursor<'a>: RawCursor
    where
        Self: 'a;
    fn cursor(&self, start: usize) -> Self::Cursor<'_>;
    /// `f(row, value)` for rows `0..n` of a dense bin, or for the stored
    /// non-zero entries of a sparse bin, in row order.
    fn for_each_row(&self, n: usize, f: impl FnMut(usize, u32));
    /// Whether every row is stored (rows outside a sparse bin read as 0).
    const DENSE: bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dense<T> {
    pub(crate) data: Vec<T>,
}

impl<T: BinVal> RawCursor for &[T] {
    #[inline(always)]
    fn get(&mut self, i: usize) -> u32 {
        self[i].into()
    }
}

impl<T: BinVal> RawBins for Dense<T> {
    type Cursor<'a> = &'a [T];
    const DENSE: bool = true;
    #[inline(always)]
    fn cursor(&self, _start: usize) -> &[T] {
        &self.data
    }
    #[inline(always)]
    fn for_each_row(&self, n: usize, mut f: impl FnMut(usize, u32)) {
        for (i, &v) in self.data[..n].iter().enumerate() {
            f(i, v.into());
        }
    }
}

/// upstream `DenseBin<uint8_t, true>`: two rows per byte, the even row in the low nibble.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Dense4 {
    pub(crate) data: Vec<u8>,
}

#[derive(Clone, Copy)]
pub struct Dense4Cursor<'a>(&'a [u8]);

impl RawCursor for Dense4Cursor<'_> {
    #[inline(always)]
    fn get(&mut self, i: usize) -> u32 {
        ((self.0[i >> 1] >> ((i & 1) << 2)) & 0xf) as u32
    }
}

impl RawBins for Dense4 {
    type Cursor<'a> = Dense4Cursor<'a>;
    const DENSE: bool = true;
    #[inline(always)]
    fn cursor(&self, _start: usize) -> Dense4Cursor<'_> {
        Dense4Cursor(&self.data)
    }
    #[inline(always)]
    fn for_each_row(&self, n: usize, mut f: impl FnMut(usize, u32)) {
        let mut c = self.cursor(0);
        for i in 0..n {
            f(i, c.get(i));
        }
    }
}

const K_NUM_FAST_INDEX: usize = 64;

/// upstream `SparseBin`: `deltas[k]` is the row gap from entry `k - 1` to
/// entry `k` (gaps of 256 or more are split into runs of 255 with value 0),
/// followed by one trailing 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sparse<T> {
    num_data: usize,
    deltas: Vec<u8>,
    vals: Vec<T>,
    /// `(entry, row)` of the first entry at or after each block of `2^shift` rows.
    fast_index: Vec<(isize, usize)>,
    fast_index_shift: u32,
}

impl<T: BinVal> Sparse<T> {
    #[inline(always)]
    fn num_vals(&self) -> isize {
        self.vals.len() as isize
    }

    /// upstream `NextNonzero` / `NextNonzeroFast`.
    #[inline(always)]
    fn next_nonzero(&self, i_delta: &mut isize, cur_pos: &mut usize) -> bool {
        *i_delta += 1;
        *cur_pos += self.deltas[*i_delta as usize] as usize;
        if *i_delta < self.num_vals() {
            true
        } else {
            *cur_pos = self.num_data;
            false
        }
    }

    /// upstream `LoadFromPair` (pairs sorted by row; later duplicates of a row are dropped).
    fn from_sorted_pairs(num_data: usize, pairs: &[(u32, T)]) -> Self {
        let mut deltas = Vec::with_capacity(pairs.len() + 1);
        let mut vals = Vec::with_capacity(pairs.len());
        let mut last_idx = 0usize;
        for (i, &(cur_idx, bin)) in pairs.iter().enumerate() {
            let cur_idx = cur_idx as usize;
            let mut cur_delta = cur_idx - last_idx;
            if i > 0 && cur_delta == 0 {
                continue;
            }
            while cur_delta >= 256 {
                deltas.push(255);
                vals.push(T::default());
                cur_delta -= 255;
            }
            deltas.push(cur_delta as u8);
            vals.push(bin);
            last_idx = cur_idx;
        }
        Self::from_parts(num_data, deltas, vals)
    }

    /// Entries as stored: `deltas` without the trailing 0.
    fn from_parts(num_data: usize, mut deltas: Vec<u8>, vals: Vec<T>) -> Self {
        deltas.push(0);
        deltas.shrink_to_fit();
        let mut vals = vals;
        vals.shrink_to_fit();
        let mut s = Self { num_data, deltas, vals, fast_index: Vec::new(), fast_index_shift: 0 };
        s.build_fast_index();
        s
    }

    /// upstream `GetFastIndex`.
    fn build_fast_index(&mut self) {
        let mod_size = self.num_data.div_ceil(K_NUM_FAST_INDEX);
        let mut pow2 = 1usize;
        let mut shift = 0u32;
        while pow2 < mod_size {
            pow2 <<= 1;
            shift += 1;
        }
        let mut fast = Vec::new();
        let (mut i_delta, mut cur_pos) = (-1isize, 0usize);
        let mut next_threshold = 0usize;
        while self.next_nonzero(&mut i_delta, &mut cur_pos) {
            while next_threshold <= cur_pos {
                fast.push((i_delta, cur_pos));
                next_threshold += pow2;
            }
        }
        while next_threshold < self.num_data {
            fast.push((self.num_vals() - 1, cur_pos));
            next_threshold += pow2;
        }
        fast.shrink_to_fit();
        self.fast_index = fast;
        self.fast_index_shift = shift;
    }

    /// upstream `InitIndex` (past the end beyond the fast index).
    #[inline]
    fn init_index(&self, start: usize) -> (isize, usize) {
        match self.fast_index.get(start >> self.fast_index_shift) {
            Some(&p) => p,
            None => (self.num_vals() - 1, self.num_data),
        }
    }

    fn num_stored(&self) -> usize {
        self.vals.len()
    }
}

pub struct SparseCursor<'a, T> {
    bin: &'a Sparse<T>,
    i_delta: isize,
    cur_pos: usize,
}

impl<T: BinVal> RawCursor for SparseCursor<'_, T> {
    /// upstream `SparseBinIterator::InnerRawGet`.
    #[inline(always)]
    fn get(&mut self, i: usize) -> u32 {
        while self.cur_pos < i {
            self.bin.next_nonzero(&mut self.i_delta, &mut self.cur_pos);
        }
        if self.cur_pos == i { self.bin.vals[self.i_delta as usize].into() } else { 0 }
    }
}

impl<T: BinVal> RawBins for Sparse<T> {
    type Cursor<'a> = SparseCursor<'a, T>;
    const DENSE: bool = false;
    #[inline]
    fn cursor(&self, start: usize) -> SparseCursor<'_, T> {
        let (i_delta, cur_pos) = self.init_index(start);
        SparseCursor { bin: self, i_delta, cur_pos }
    }
    #[inline]
    fn for_each_row(&self, n: usize, mut f: impl FnMut(usize, u32)) {
        let mut pos = 0usize;
        for (k, &v) in self.vals.iter().enumerate() {
            pos += self.deltas[k] as usize;
            if pos >= n {
                break;
            }
            if v.into() != 0 {
                f(pos, v.into());
            }
        }
    }
}

/// The stored bins of a group or of one feature of a multi-value group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Bin {
    Dense4(Dense4),
    Dense8(Dense<u8>),
    Dense16(Dense<u16>),
    Dense32(Dense<u32>),
    Sparse8(Sparse<u8>),
    Sparse16(Sparse<u16>),
    Sparse32(Sparse<u32>),
}

/// Calls `$body` with `$b` bound to the concrete storage of `$bin`.
macro_rules! with_bin {
    ($bin:expr, $b:ident => $body:expr) => {
        match $bin {
            $crate::bin::Bin::Dense4($b) => $body,
            $crate::bin::Bin::Dense8($b) => $body,
            $crate::bin::Bin::Dense16($b) => $body,
            $crate::bin::Bin::Dense32($b) => $body,
            $crate::bin::Bin::Sparse8($b) => $body,
            $crate::bin::Bin::Sparse16($b) => $body,
            $crate::bin::Bin::Sparse32($b) => $body,
        }
    };
}
pub(crate) use with_bin;

/// Storage kind of a [`Bin`] (also its tag in binary dataset files).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinKind {
    Dense4 = 0,
    Dense8 = 1,
    Dense16 = 2,
    Dense32 = 3,
    Sparse8 = 4,
    Sparse16 = 5,
    Sparse32 = 6,
}

impl BinKind {
    /// upstream `Bin::CreateDenseBin`.
    pub fn dense(num_bin: u32) -> Self {
        if num_bin <= 16 {
            BinKind::Dense4
        } else if num_bin <= 256 {
            BinKind::Dense8
        } else if num_bin <= 65536 {
            BinKind::Dense16
        } else {
            BinKind::Dense32
        }
    }

    /// upstream `Bin::CreateSparseBin`.
    pub fn sparse(num_bin: u32) -> Self {
        if num_bin <= 256 {
            BinKind::Sparse8
        } else if num_bin <= 65536 {
            BinKind::Sparse16
        } else {
            BinKind::Sparse32
        }
    }

    pub fn is_sparse(self) -> bool {
        matches!(self, BinKind::Sparse8 | BinKind::Sparse16 | BinKind::Sparse32)
    }

    pub fn from_tag(t: u8) -> Option<Self> {
        Some(match t {
            0 => BinKind::Dense4,
            1 => BinKind::Dense8,
            2 => BinKind::Dense16,
            3 => BinKind::Dense32,
            4 => BinKind::Sparse8,
            5 => BinKind::Sparse16,
            6 => BinKind::Sparse32,
            _ => return None,
        })
    }
}

/// A forward cursor over any [`Bin`] (enum-dispatched, for non-hot paths).
pub enum AnyCursor<'a> {
    Dense4(Dense4Cursor<'a>),
    Dense8(&'a [u8]),
    Dense16(&'a [u16]),
    Dense32(&'a [u32]),
    Sparse8(SparseCursor<'a, u8>),
    Sparse16(SparseCursor<'a, u16>),
    Sparse32(SparseCursor<'a, u32>),
}

impl RawCursor for AnyCursor<'_> {
    #[inline]
    fn get(&mut self, i: usize) -> u32 {
        match self {
            AnyCursor::Dense4(c) => c.get(i),
            AnyCursor::Dense8(c) => c.get(i),
            AnyCursor::Dense16(c) => c.get(i),
            AnyCursor::Dense32(c) => c.get(i),
            AnyCursor::Sparse8(c) => c.get(i),
            AnyCursor::Sparse16(c) => c.get(i),
            AnyCursor::Sparse32(c) => c.get(i),
        }
    }
}

impl Bin {
    pub fn kind(&self) -> BinKind {
        match self {
            Bin::Dense4(_) => BinKind::Dense4,
            Bin::Dense8(_) => BinKind::Dense8,
            Bin::Dense16(_) => BinKind::Dense16,
            Bin::Dense32(_) => BinKind::Dense32,
            Bin::Sparse8(_) => BinKind::Sparse8,
            Bin::Sparse16(_) => BinKind::Sparse16,
            Bin::Sparse32(_) => BinKind::Sparse32,
        }
    }

    pub fn is_sparse(&self) -> bool {
        self.kind().is_sparse()
    }

    pub fn any_cursor(&self, start: usize) -> AnyCursor<'_> {
        match self {
            Bin::Dense4(b) => AnyCursor::Dense4(b.cursor(start)),
            Bin::Dense8(b) => AnyCursor::Dense8(b.cursor(start)),
            Bin::Dense16(b) => AnyCursor::Dense16(b.cursor(start)),
            Bin::Dense32(b) => AnyCursor::Dense32(b.cursor(start)),
            Bin::Sparse8(b) => AnyCursor::Sparse8(b.cursor(start)),
            Bin::Sparse16(b) => AnyCursor::Sparse16(b.cursor(start)),
            Bin::Sparse32(b) => AnyCursor::Sparse32(b.cursor(start)),
        }
    }

    /// Every stored value, row by row.
    pub fn values(&self, num_data: usize) -> Vec<u32> {
        let mut c = self.any_cursor(0);
        (0..num_data).map(|i| c.get(i)).collect()
    }

    /// Bytes held by the stored values.
    pub fn heap_bytes(&self) -> usize {
        match self {
            Bin::Dense4(b) => b.data.len(),
            Bin::Dense8(b) => b.data.len(),
            Bin::Dense16(b) => 2 * b.data.len(),
            Bin::Dense32(b) => 4 * b.data.len(),
            Bin::Sparse8(b) => b.deltas.len() + b.vals.len() + 16 * b.fast_index.len(),
            Bin::Sparse16(b) => b.deltas.len() + 2 * b.vals.len() + 16 * b.fast_index.len(),
            Bin::Sparse32(b) => b.deltas.len() + 4 * b.vals.len() + 16 * b.fast_index.len(),
        }
    }

    /// A sparse bin from `(row, value)` pairs in push order (upstream
    /// `SparseBin::FinishLoad`: `std::sort` by row, then `LoadFromPair`,
    /// which keeps the first pair of each row).
    pub fn sparse_from_pairs(kind: BinKind, num_data: usize, mut pairs: Vec<(u32, u32)>) -> Bin {
        if pairs.windows(2).any(|w| w[0].0 >= w[1].0) {
            stdsort::sort_by(&mut pairs, |a, b| a.0 < b.0);
        }
        fn mk<T: BinVal>(n: usize, pairs: &[(u32, u32)]) -> Sparse<T> {
            let typed: Vec<(u32, T)> = pairs.iter().map(|&(r, v)| (r, T::from_u32(v))).collect();
            Sparse::from_sorted_pairs(n, &typed)
        }
        match kind {
            BinKind::Sparse8 => Bin::Sparse8(mk(num_data, &pairs)),
            BinKind::Sparse16 => Bin::Sparse16(mk(num_data, &pairs)),
            BinKind::Sparse32 => Bin::Sparse32(mk(num_data, &pairs)),
            _ => unreachable!("dense kind {kind:?}"),
        }
    }

    /// A dense bin of zeros (upstream `DenseBin` before any push).
    pub fn dense_zeros(kind: BinKind, num_data: usize) -> Bin {
        match kind {
            BinKind::Dense4 => Bin::Dense4(Dense4 { data: vec![0; num_data.div_ceil(2)] }),
            BinKind::Dense8 => Bin::Dense8(Dense { data: vec![0; num_data] }),
            BinKind::Dense16 => Bin::Dense16(Dense { data: vec![0; num_data] }),
            BinKind::Dense32 => Bin::Dense32(Dense { data: vec![0; num_data] }),
            _ => unreachable!("sparse kind {kind:?}"),
        }
    }

    /// upstream `Bin::CopySubrow`: rows `used` (ascending) of `self`.
    pub fn copy_subrow(&self, used: &[u32]) -> Bin {
        fn pick<T: Copy>(v: &[T], used: &[u32]) -> Vec<T> {
            used.iter().map(|&i| v[i as usize]).collect()
        }
        fn sub<T: BinVal>(b: &Sparse<T>, used: &[u32]) -> Sparse<T> {
            let mut c = b.cursor(used.first().map_or(0, |&i| i as usize));
            let mut deltas = Vec::new();
            let mut vals = Vec::new();
            let mut last_idx = 0usize;
            for (i, &r) in used.iter().enumerate() {
                let v = c.get(r as usize);
                if v > 0 {
                    let mut cur_delta = i - last_idx;
                    while cur_delta >= 256 {
                        deltas.push(255);
                        vals.push(T::default());
                        cur_delta -= 255;
                    }
                    deltas.push(cur_delta as u8);
                    vals.push(T::from_u32(v));
                    last_idx = i;
                }
            }
            Sparse::from_parts(used.len(), deltas, vals)
        }
        match self {
            Bin::Dense4(b) => {
                let mut data = vec![0u8; used.len().div_ceil(2)];
                let mut c = b.cursor(0);
                for (i, &r) in used.iter().enumerate() {
                    data[i >> 1] |= (c.get(r as usize) as u8) << ((i & 1) << 2);
                }
                Bin::Dense4(Dense4 { data })
            }
            Bin::Dense8(b) => Bin::Dense8(Dense { data: pick(&b.data, used) }),
            Bin::Dense16(b) => Bin::Dense16(Dense { data: pick(&b.data, used) }),
            Bin::Dense32(b) => Bin::Dense32(Dense { data: pick(&b.data, used) }),
            Bin::Sparse8(b) => Bin::Sparse8(sub(b, used)),
            Bin::Sparse16(b) => Bin::Sparse16(sub(b, used)),
            Bin::Sparse32(b) => Bin::Sparse32(sub(b, used)),
        }
    }

    /// Serialized stored values (the kind is written by the caller).
    pub fn write(&self, out: &mut Vec<u8>) {
        fn sparse<T: BinVal>(b: &Sparse<T>, out: &mut Vec<u8>) {
            out.extend_from_slice(&(b.num_stored() as u64).to_le_bytes());
            out.extend_from_slice(&b.deltas[..b.num_stored()]);
            for &v in &b.vals {
                v.put_le(out);
            }
        }
        match self {
            Bin::Dense4(b) => out.extend_from_slice(&b.data),
            Bin::Dense8(b) => out.extend_from_slice(&b.data),
            Bin::Dense16(b) => b.data.iter().for_each(|&v| v.put_le(out)),
            Bin::Dense32(b) => b.data.iter().for_each(|&v| v.put_le(out)),
            Bin::Sparse8(b) => sparse(b, out),
            Bin::Sparse16(b) => sparse(b, out),
            Bin::Sparse32(b) => sparse(b, out),
        }
    }

    /// Inverse of [`Bin::write`]; `take(n)` returns the next `n` bytes
    /// (`None` when the input is too short). Values above `max_value` and
    /// sparse entries past `num_data` are rejected (`None`).
    pub fn read(
        kind: BinKind,
        num_data: usize,
        max_value: u32,
        take: &mut dyn FnMut(usize) -> Option<Vec<u8>>,
    ) -> Option<Bin> {
        fn dense<T: BinVal>(n: usize, max: u32, take: &mut dyn FnMut(usize) -> Option<Vec<u8>>) -> Option<Dense<T>> {
            let b = take(n.checked_mul(T::BYTES)?)?;
            let data: Vec<T> = b.chunks_exact(T::BYTES).map(T::get_le).collect();
            data.iter().all(|&v| v.into() <= max).then_some(Dense { data })
        }
        fn sparse<T: BinVal>(n: usize, max: u32, take: &mut dyn FnMut(usize) -> Option<Vec<u8>>) -> Option<Sparse<T>> {
            let cnt = usize::try_from(u64::from_le_bytes(take(8)?.try_into().ok()?)).ok()?;
            let deltas = take(cnt)?;
            let b = take(cnt.checked_mul(T::BYTES)?)?;
            let vals: Vec<T> = b.chunks_exact(T::BYTES).map(T::get_le).collect();
            let mut pos = 0usize;
            for (k, (&d, &v)) in deltas.iter().zip(&vals).enumerate() {
                if k > 0 && d == 0 {
                    return None;
                }
                pos += d as usize;
                if pos >= n || v.into() > max {
                    return None;
                }
            }
            Some(Sparse::from_parts(n, deltas, vals))
        }
        Some(match kind {
            BinKind::Dense4 => {
                let data = take(num_data.div_ceil(2))?;
                let b = Dense4 { data };
                let mut c = b.cursor(0);
                if (0..num_data).any(|i| c.get(i) > max_value) || (num_data % 2 == 1 && b.data[num_data / 2] >> 4 != 0) {
                    return None;
                }
                Bin::Dense4(b)
            }
            BinKind::Dense8 => Bin::Dense8(dense(num_data, max_value, take)?),
            BinKind::Dense16 => Bin::Dense16(dense(num_data, max_value, take)?),
            BinKind::Dense32 => Bin::Dense32(dense(num_data, max_value, take)?),
            BinKind::Sparse8 => Bin::Sparse8(sparse(num_data, max_value, take)?),
            BinKind::Sparse16 => Bin::Sparse16(sparse(num_data, max_value, take)?),
            BinKind::Sparse32 => Bin::Sparse32(sparse(num_data, max_value, take)?),
        })
    }
}

/// Writes stored values of a dense [`Bin`] from several threads at once.
///
/// Callers must write disjoint rows from different threads, and for a
/// 4-bit bin disjoint row pairs (`2k`, `2k + 1`), which share a byte.
pub(crate) struct DenseWriter {
    ptr: *mut u8,
    kind: BinKind,
}

// SAFETY: see the type's contract; each write touches only its row's bytes.
unsafe impl Send for DenseWriter {}
unsafe impl Sync for DenseWriter {}

impl DenseWriter {
    pub(crate) fn new(bin: &mut Bin) -> Self {
        let (ptr, kind) = match bin {
            Bin::Dense4(b) => (b.data.as_mut_ptr(), BinKind::Dense4),
            Bin::Dense8(b) => (b.data.as_mut_ptr(), BinKind::Dense8),
            Bin::Dense16(b) => (b.data.as_mut_ptr() as *mut u8, BinKind::Dense16),
            Bin::Dense32(b) => (b.data.as_mut_ptr() as *mut u8, BinKind::Dense32),
            _ => unreachable!("not a dense bin"),
        };
        Self { ptr, kind }
    }

    /// upstream `DenseBin::Push`: the last value pushed for a row wins.
    ///
    /// # Safety
    /// `i` is a row of the bin this writer was made from, which is still
    /// alive and not otherwise accessed, and the type's threading contract holds.
    #[inline(always)]
    pub(crate) unsafe fn set(&self, i: usize, v: u32) {
        unsafe {
            match self.kind {
                BinKind::Dense4 => {
                    let p = self.ptr.add(i >> 1);
                    let s = (i & 1) << 2;
                    *p = (*p & !(0xf << s)) | ((v as u8) << s);
                }
                BinKind::Dense8 => *self.ptr.add(i) = v as u8,
                BinKind::Dense16 => *(self.ptr as *mut u16).add(i) = v as u16,
                BinKind::Dense32 => *(self.ptr as *mut u32).add(i) = v,
                _ => unreachable!(),
            }
        }
    }
}

/// Accumulate `[g, h]` of rows into `hist[2 * raw]` (upstream
/// `Bin::ConstructHistogram`): rows `idx` with gradients `og[k]`/`oh[k]` of
/// `idx[k]` (ordered), or all rows with `og[i]`/`oh[i]`. Dense bins add every
/// row, sparse bins only their stored entries.
pub fn construct_histogram<B: RawBins>(bin: &B, indices: Option<&[u32]>, og: &[f32], oh: &[f32], hist: &mut [f64]) {
    let mut add = |raw: u32, k: usize| {
        let t = (raw as usize) << 1;
        hist[t] += og[k] as f64;
        hist[t + 1] += oh[k] as f64;
    };
    match indices {
        Some(idx) => {
            let Some(&first) = idx.first() else { return };
            let mut c = bin.cursor(first as usize);
            for (k, &i) in idx.iter().enumerate() {
                let raw = c.get(i as usize);
                if B::DENSE || raw != 0 {
                    add(raw, k);
                }
            }
        }
        None => bin.for_each_row(og.len(), |i, raw| add(raw, i)),
    }
}

/// Accumulate the packed integer gradients and hessians `gh[row]` (by global
/// row) into `hist[raw]` (upstream `Bin::ConstructHistogramInt32`): rows
/// `idx`, or rows `0..n`. Integer sums do not depend on the order.
pub fn construct_histogram_int<B: RawBins>(bin: &B, indices: Option<&[u32]>, n: usize, gh: &[i64], hist: &mut [i64]) {
    match indices {
        Some(idx) => {
            let Some(&first) = idx.first() else { return };
            let mut c = bin.cursor(first as usize);
            for &i in idx {
                let raw = c.get(i as usize);
                if B::DENSE || raw != 0 {
                    let t = &mut hist[raw as usize];
                    *t = t.wrapping_add(gh[i as usize]);
                }
            }
        }
        None => bin.for_each_row(n, |i, raw| {
            let t = &mut hist[raw as usize];
            *t = t.wrapping_add(gh[i]);
        }),
    }
}

/// Stable split of `src` (ascending rows) by `lut[raw]` into `l` and `r`;
/// returns the counts.
#[inline(always)]
pub fn split_rows<B: RawBins>(bin: &B, lut: &[bool], src: &[u32], l: &mut [u32], r: &mut [u32]) -> (usize, usize) {
    let (mut nl, mut nr) = (0usize, 0usize);
    assert!(l.len() >= src.len() && r.len() >= src.len());
    let Some(&first) = src.first() else { return (0, 0) };
    let mut c = bin.cursor(first as usize);
    for &i in src {
        let go_left = lut[c.get(i as usize) as usize];
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

#[cfg(test)]
mod tests {
    use super::*;

    fn values(b: &Bin, n: usize) -> Vec<u32> {
        b.values(n)
    }

    #[test]
    fn sparse_runs_fast_index_and_duplicates() {
        let n = 2000;
        // a gap of 600 rows needs runs of 255; row 7 is pushed twice
        let pairs = vec![(7, 3), (3, 1), (7, 9), (610, 2), (1999, 5)];
        let b = Bin::sparse_from_pairs(BinKind::Sparse8, n, pairs);
        let v = values(&b, n);
        assert_eq!(v[3], 1);
        assert_eq!(v[7], 3, "std::sort keeps the first of equal rows here");
        assert_eq!(v[610], 2);
        assert_eq!(v[1999], 5);
        assert_eq!(v.iter().filter(|&&x| x != 0).count(), 4);
        // every start point of a cursor reads the same values
        for start in [0usize, 5, 64, 600, 1024, 1998] {
            let mut c = b.any_cursor(start);
            for (i, &want) in v.iter().enumerate().skip(start) {
                assert_eq!(c.get(i), want, "start {start} row {i}");
            }
        }
        let sub = b.copy_subrow(&[3, 7, 8, 1999]);
        assert_eq!(values(&sub, 4), vec![1, 3, 0, 5]);
    }

    #[test]
    fn dense4_packs_two_rows_per_byte() {
        let mut b = Bin::dense_zeros(BinKind::Dense4, 5);
        let w = DenseWriter::new(&mut b);
        for (i, v) in [3u32, 15, 0, 7, 9].into_iter().enumerate() {
            unsafe { w.set(i, v) };
        }
        unsafe { w.set(1, 4) };
        assert_eq!(values(&b, 5), vec![3, 4, 0, 7, 9]);
        assert_eq!(values(&b.copy_subrow(&[1, 4]), 2), vec![4, 9]);
    }

    #[test]
    fn serialization_round_trips() {
        let n = 700;
        let bins = [
            Bin::sparse_from_pairs(BinKind::Sparse16, n, vec![(0, 300), (299, 1), (699, 2)]),
            Bin::sparse_from_pairs(BinKind::Sparse8, n, vec![]),
            {
                let mut b = Bin::dense_zeros(BinKind::Dense4, n);
                let w = DenseWriter::new(&mut b);
                (0..n).for_each(|i| unsafe { w.set(i, (i % 13) as u32) });
                b
            },
        ];
        for b in bins {
            let mut out = Vec::new();
            b.write(&mut out);
            let mut pos = 0;
            let mut take = |k: usize| {
                let s = out.get(pos..pos + k)?.to_vec();
                pos += k;
                Some(s)
            };
            let back = Bin::read(b.kind(), n, 300, &mut take).unwrap();
            assert_eq!(back, b);
            assert_eq!(pos, out.len());
        }
    }

    #[test]
    fn histogram_and_split_match_dense_reference() {
        let n = 300;
        let pairs: Vec<(u32, u32)> = (0..n as u32).filter(|i| i % 7 == 0).map(|i| (i, 1 + i % 3)).collect();
        let sp = Bin::sparse_from_pairs(BinKind::Sparse8, n, pairs);
        let vals = sp.values(n);
        let g: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let idx: Vec<u32> = (0..n as u32).filter(|i| i % 3 != 1).collect();
        let og: Vec<f32> = idx.iter().map(|&i| g[i as usize]).collect();
        let mut want = vec![0.0; 8];
        for (k, &i) in idx.iter().enumerate() {
            let v = vals[i as usize] as usize;
            if v != 0 {
                want[2 * v] += og[k] as f64;
                want[2 * v + 1] += og[k] as f64;
            }
        }
        let mut got = vec![0.0; 8];
        with_bin!(&sp, b => construct_histogram(b, Some(&idx), &og, &og, &mut got));
        assert_eq!(got, want);
        let lut = [true, false, true, false];
        let (mut l, mut r) = (vec![0; idx.len()], vec![0; idx.len()]);
        let (nl, nr) = with_bin!(&sp, b => split_rows(b, &lut, &idx, &mut l, &mut r));
        let wl: Vec<u32> = idx.iter().copied().filter(|&i| lut[vals[i as usize] as usize]).collect();
        assert_eq!(&l[..nl], &wl[..]);
        assert_eq!(nl + nr, idx.len());
    }
}
