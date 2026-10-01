//! Feature matrices accepted for training and prediction: dense, CSR and CSC.
//!
//! upstream: c_api.cpp `LGBM_DatasetCreateFromCSR`, `LGBM_DatasetCreateFromCSC`,
//! `RowFunctionFromCSR`, `CSC_RowIterator`, `LGBM_BoosterPredictForCSR`,
//! `LGBM_BoosterPredictForCSC`.
//!
//! Entries that are not stored are 0.0, so a sparse matrix bins and predicts
//! exactly like its dense equivalent. Inputs with duplicate entries for the
//! same cell are accepted, but the last stored entry wins, which only matches
//! upstream for CSR input.

use rayon::prelude::*;

use crate::binning::BinMapper;
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::{BinColumn, DenseMatrix, DenseValues};
use crate::error::{LgbmError, Result};
use crate::feature_groups::SampleColumn;

/// Offsets into `indices`/`values` (`indptr`), 32- or 64-bit like upstream.
#[derive(Debug, Clone, Copy)]
pub enum SparseIndptr<'a> {
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl SparseIndptr<'_> {
    fn len(&self) -> usize {
        match self {
            SparseIndptr::I32(v) => v.len(),
            SparseIndptr::I64(v) => v.len(),
        }
    }

    #[inline]
    fn get(&self, i: usize) -> i64 {
        match self {
            SparseIndptr::I32(v) => v[i] as i64,
            SparseIndptr::I64(v) => v[i],
        }
    }
}

/// A borrowed compressed sparse matrix: CSR (`row_major`) or CSC.
#[derive(Debug, Clone, Copy)]
pub struct SparseMatrix<'a> {
    indptr: SparseIndptr<'a>,
    indices: &'a [i32],
    values: DenseValues<'a>,
    nrows: usize,
    ncols: usize,
    row_major: bool,
}

impl<'a> SparseMatrix<'a> {
    pub fn new(
        indptr: SparseIndptr<'a>,
        indices: &'a [i32],
        values: DenseValues<'a>,
        nrows: usize,
        ncols: usize,
        row_major: bool,
    ) -> Result<Self> {
        let nnz = match values {
            DenseValues::F32(v) => v.len(),
            DenseValues::F64(v) => v.len(),
        };
        if indices.len() != nnz {
            return Err(LgbmError::InvalidData(format!("Length mismatch: {} vs {nnz}", indices.len())));
        }
        let (outer, inner) = if row_major { (nrows, ncols) } else { (ncols, nrows) };
        if indptr.len() != outer + 1 {
            return Err(LgbmError::InvalidData(format!(
                "sparse matrix indptr has {} entries, expected {}",
                indptr.len(),
                outer + 1
            )));
        }
        let mut prev = 0i64;
        for i in 0..indptr.len() {
            let p = indptr.get(i);
            if p < prev || p as u64 > nnz as u64 {
                return Err(LgbmError::InvalidData(
                    "sparse matrix indptr must be non-decreasing, start at 0 or more and not exceed the number of stored values"
                        .into(),
                ));
            }
            prev = p;
        }
        if let Some(&bad) = indices.iter().find(|&&j| j < 0 || j as usize >= inner) {
            return Err(LgbmError::InvalidData(format!("sparse matrix index {bad} out of range [0, {inner})")));
        }
        Ok(Self { indptr, indices, values, nrows, ncols, row_major })
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn is_csr(&self) -> bool {
        self.row_major
    }

    /// Stored entries of row (CSR) or column (CSC) `i`.
    #[inline]
    fn outer(&self, i: usize) -> std::ops::Range<usize> {
        self.indptr.get(i) as usize..self.indptr.get(i + 1) as usize
    }

    #[inline]
    fn value(&self, k: usize) -> f64 {
        match self.values {
            DenseValues::F32(v) => v[k] as f64,
            DenseValues::F64(v) => v[k],
        }
    }

    /// The same matrix in the other compression order. Entries of each
    /// output row/column keep their stored order.
    fn transpose(&self) -> Compressed {
        let (outer, inner) = if self.row_major { (self.nrows, self.ncols) } else { (self.ncols, self.nrows) };
        let mut ptr = vec![0usize; inner + 1];
        for &j in self.indices_used(outer) {
            ptr[j as usize + 1] += 1;
        }
        for j in 0..inner {
            ptr[j + 1] += ptr[j];
        }
        let nnz = ptr[inner];
        let mut idx = vec![0u32; nnz];
        let mut val = vec![0.0f64; nnz];
        let mut next = ptr.clone();
        for i in 0..outer {
            for k in self.outer(i) {
                let j = self.indices[k] as usize;
                idx[next[j]] = i as u32;
                val[next[j]] = self.value(k);
                next[j] += 1;
            }
        }
        Compressed { ptr, idx, val }
    }

    /// Indices referenced by `indptr` (stored values past `indptr[outer]` are ignored).
    fn indices_used(&self, outer: usize) -> &[i32] {
        &self.indices[self.indptr.get(0) as usize..self.indptr.get(outer) as usize]
    }
}

/// An owned compressed matrix (the transpose of a [`SparseMatrix`]).
pub(crate) struct Compressed {
    ptr: Vec<usize>,
    idx: Vec<u32>,
    val: Vec<f64>,
}

/// A borrowed feature matrix.
#[derive(Debug, Clone, Copy)]
pub enum Matrix<'a> {
    Dense(DenseMatrix<'a>),
    Sparse(SparseMatrix<'a>),
}

impl<'a> From<DenseMatrix<'a>> for Matrix<'a> {
    fn from(m: DenseMatrix<'a>) -> Self {
        Matrix::Dense(m)
    }
}

impl<'a> From<SparseMatrix<'a>> for Matrix<'a> {
    fn from(m: SparseMatrix<'a>) -> Self {
        Matrix::Sparse(m)
    }
}

impl Matrix<'_> {
    pub fn nrows(&self) -> usize {
        match self {
            Matrix::Dense(m) => m.nrows(),
            Matrix::Sparse(m) => m.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Matrix::Dense(m) => m.ncols(),
            Matrix::Sparse(m) => m.ncols(),
        }
    }

    /// Nonzero (or NaN) values of every column at the sampled rows.
    ///
    /// `sample_indices` must be sorted ascending (as `Random::sample` returns).
    pub(crate) fn sample_columns(&self, sample_indices: &[i32]) -> Vec<SampleColumn> {
        let keep = |v: f64| v.abs() > K_ZERO_THRESHOLD || v.is_nan();
        match self {
            Matrix::Dense(mat) => (0..mat.ncols())
                .into_par_iter()
                .map(|c| {
                    let mut col = SampleColumn { indices: Vec::new(), values: Vec::new() };
                    for (i, &r) in sample_indices.iter().enumerate() {
                        let v = mat.get(r as usize, c);
                        if keep(v) {
                            col.indices.push(i as i32);
                            col.values.push(v);
                        }
                    }
                    col
                })
                .collect(),
            Matrix::Sparse(mat) if mat.row_major => {
                // upstream: LGBM_DatasetCreateFromCSR sampling loop
                let mut cols: Vec<SampleColumn> = (0..mat.ncols)
                    .map(|_| SampleColumn { indices: Vec::new(), values: Vec::new() })
                    .collect();
                for (i, &r) in sample_indices.iter().enumerate() {
                    for k in mat.outer(r as usize) {
                        let v = mat.value(k);
                        if keep(v) {
                            let col = &mut cols[mat.indices[k] as usize];
                            col.indices.push(i as i32);
                            col.values.push(v);
                        }
                    }
                }
                cols
            }
            Matrix::Sparse(mat) => (0..mat.ncols)
                .into_par_iter()
                .map(|c| {
                    // upstream: LGBM_DatasetCreateFromCSC sampling via CSC_RowIterator::Get
                    let mut col = SampleColumn { indices: Vec::new(), values: Vec::new() };
                    let mut it = CscRowIterator::new(mat, c);
                    for (i, &r) in sample_indices.iter().enumerate() {
                        let v = it.get(r as i64);
                        if keep(v) {
                            col.indices.push(i as i32);
                            col.values.push(v);
                        }
                    }
                    col
                })
                .collect(),
        }
    }

    /// Bin indices of columns `used`.
    pub(crate) fn build_bins(&self, used: &[usize], mappers: &[BinMapper]) -> Vec<BinColumn> {
        let n = self.nrows();
        match self {
            Matrix::Dense(mat) => used
                .par_iter()
                .map(|&c| {
                    let m = &mappers[c];
                    BinColumn::build(m.num_bin, n, |r| m.value_to_bin(mat.get(r, c)))
                })
                .collect(),
            Matrix::Sparse(mat) => {
                let transposed;
                let (ptr, rows, vals): (Box<dyn Fn(usize) -> std::ops::Range<usize> + Sync>, _, _) =
                    if mat.row_major {
                        transposed = mat.transpose();
                        let p = &transposed.ptr;
                        (
                            Box::new(move |c: usize| p[c]..p[c + 1]),
                            Rows::Owned(&transposed.idx),
                            Vals::Owned(&transposed.val),
                        )
                    } else {
                        (Box::new(|c: usize| mat.outer(c)), Rows::Borrowed(mat.indices), Vals::Borrowed(mat))
                    };
                used.par_iter()
                    .map(|&c| {
                        let m = &mappers[c];
                        let mut col = BinColumn::filled(m.num_bin, n, m.value_to_bin(0.0));
                        for k in ptr(c) {
                            col.set(rows.get(k), m.value_to_bin(vals.get(k)));
                        }
                        col
                    })
                    .collect()
            }
        }
    }

    /// Row reader for prediction (CSC input is transposed to CSR once).
    pub(crate) fn rows(&self) -> RowReader<'_> {
        match self {
            Matrix::Dense(m) => RowReader::Dense(m),
            Matrix::Sparse(m) if m.row_major => RowReader::Csr(m),
            Matrix::Sparse(m) => RowReader::Owned(m.transpose()),
        }
    }
}

enum Rows<'a> {
    Owned(&'a [u32]),
    Borrowed(&'a [i32]),
}

impl Rows<'_> {
    #[inline]
    fn get(&self, k: usize) -> usize {
        match self {
            Rows::Owned(v) => v[k] as usize,
            Rows::Borrowed(v) => v[k] as usize,
        }
    }
}

enum Vals<'a> {
    Owned(&'a [f64]),
    Borrowed(&'a SparseMatrix<'a>),
}

impl Vals<'_> {
    #[inline]
    fn get(&self, k: usize) -> f64 {
        match self {
            Vals::Owned(v) => v[k],
            Vals::Borrowed(m) => m.value(k),
        }
    }
}

pub(crate) enum RowReader<'a> {
    Dense(&'a DenseMatrix<'a>),
    Csr(&'a SparseMatrix<'a>),
    Owned(Compressed),
}

impl RowReader<'_> {
    /// Copy row `r` into `out` (length `ncols`) as f64.
    #[inline]
    pub(crate) fn row_into(&self, r: usize, out: &mut [f64]) {
        match self {
            RowReader::Dense(m) => m.row_into(r, out),
            RowReader::Csr(m) => {
                out.fill(0.0);
                for k in m.outer(r) {
                    out[m.indices[k] as usize] = m.value(k);
                }
            }
            RowReader::Owned(c) => {
                out.fill(0.0);
                for k in c.ptr[r]..c.ptr[r + 1] {
                    out[c.idx[k] as usize] = c.val[k];
                }
            }
        }
    }
}

/// upstream: `CSC_RowIterator` (forward-only lookup within one column).
struct CscRowIterator<'a> {
    mat: &'a SparseMatrix<'a>,
    next: usize,
    end: usize,
    cur_idx: i64,
    cur_val: f64,
    is_end: bool,
}

impl<'a> CscRowIterator<'a> {
    fn new(mat: &'a SparseMatrix<'a>, col: usize) -> Self {
        let r = mat.outer(col);
        Self { mat, next: r.start, end: r.end, cur_idx: -1, cur_val: 0.0, is_end: false }
    }

    fn get(&mut self, idx: i64) -> f64 {
        while idx > self.cur_idx && !self.is_end {
            if self.next >= self.end {
                self.is_end = true;
                break;
            }
            self.cur_idx = self.mat.indices[self.next] as i64;
            self.cur_val = self.mat.value(self.next);
            self.next += 1;
        }
        if idx == self.cur_idx { self.cur_val } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_of(m: &SparseMatrix<'_>) -> Vec<f64> {
        let mat = Matrix::Sparse(*m);
        let rows = mat.rows();
        let mut out = vec![0.0; m.nrows() * m.ncols()];
        for r in 0..m.nrows() {
            rows.row_into(r, &mut out[r * m.ncols()..(r + 1) * m.ncols()]);
        }
        out
    }

    #[test]
    fn csr_and_csc_rows_match_dense() {
        // [[1, 0, 2], [0, 0, 3], [4, 5, 0]]
        let expected = [1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0];
        let (p, i, v) = ([0i32, 2, 3, 5], [0i32, 2, 2, 0, 1], [1.0f64, 2.0, 3.0, 4.0, 5.0]);
        let csr = SparseMatrix::new(SparseIndptr::I32(&p), &i, DenseValues::F64(&v), 3, 3, true).unwrap();
        assert_eq!(dense_of(&csr), expected);
        let (p, i, v) = ([0i64, 2, 3, 5], [0i32, 2, 2, 0, 1], [1.0f32, 4.0, 5.0, 2.0, 3.0]);
        let csc = SparseMatrix::new(SparseIndptr::I64(&p), &i, DenseValues::F32(&v), 3, 3, false).unwrap();
        assert_eq!(dense_of(&csc), expected);
    }

    #[test]
    fn csc_row_iterator_matches_upstream() {
        let (p, i, v) = ([0i32, 3], [1i32, 4, 6], [1.0f64, 2.0, 3.0]);
        let m = SparseMatrix::new(SparseIndptr::I32(&p), &i, DenseValues::F64(&v), 8, 1, false).unwrap();
        let mut it = CscRowIterator::new(&m, 0);
        let got: Vec<f64> = [0, 1, 2, 4, 7].iter().map(|&r| it.get(r)).collect();
        assert_eq!(got, [0.0, 1.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn invalid_inputs_rejected() {
        let v = [1.0f64];
        let e = SparseMatrix::new(SparseIndptr::I32(&[0, 1]), &[0, 1], DenseValues::F64(&v), 1, 2, true).unwrap_err();
        assert!(e.to_string().contains("Length mismatch: 2 vs 1"), "{e}");
        assert!(SparseMatrix::new(SparseIndptr::I32(&[0, 1]), &[2], DenseValues::F64(&v), 1, 2, true).is_err());
        assert!(SparseMatrix::new(SparseIndptr::I32(&[0, 2]), &[0], DenseValues::F64(&v), 1, 2, true).is_err());
        assert!(SparseMatrix::new(SparseIndptr::I32(&[0]), &[0], DenseValues::F64(&v), 1, 2, true).is_err());
    }
}
