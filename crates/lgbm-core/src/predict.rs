//! Prediction on raw (unbinned) feature values.
//!
//! upstream: src/application/predictor.hpp, src/boosting/gbdt_prediction.cpp.

use rayon::prelude::*;

use crate::binning::MissingType;
use crate::boosting::{Gbdt, PredictKind};
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::{DenseMatrix, DenseValues};
use crate::fmt::fmt_g17;
use crate::matrix::{Matrix, SparseIndptr, SparseMatrix};
use crate::error::{LgbmError, Result};
use crate::text_parser::{self, Parser};
use crate::tree::{Tree, find_in_bitset};

/// Rows walked through each tree together, so that their (independent)
/// root-to-leaf paths overlap in the pipeline.
const BLOCK: usize = 16;

#[derive(Clone, Copy)]
struct Node {
    threshold: f64,
    /// Feature index into the row buffer (`ncol` reads a constant 0.0).
    feature: u32,
    missing_type: i8,
    default_left: bool,
    /// Index into `FlatTree::cat_bits` for categorical nodes, else -1.
    cat: i32,
    /// `[left, right]`
    children: [i32; 2],
}

/// One tree in a compact layout for branchless traversal; decisions are
/// those of `Tree::decision`.
struct FlatTree {
    nodes: Vec<Node>,
    depth: usize,
    /// Category bitsets of the categorical nodes.
    cat_bits: Vec<Vec<u32>>,
}

impl FlatTree {
    fn new(t: &Tree, ncol: usize) -> Self {
        let ni = t.num_leaves.saturating_sub(1);
        let mut cat_bits = Vec::new();
        let nodes: Vec<Node> = (0..ni)
            .map(|n| Node {
                threshold: t.threshold[n],
                feature: (t.split_feature[n] as usize).min(ncol) as u32,
                missing_type: (t.decision_type[n] >> 2) & 3,
                default_left: t.decision_type[n] & 2 > 0,
                cat: if t.is_categorical(n) {
                    cat_bits.push(t.cat_bitset(n).to_vec());
                    cat_bits.len() as i32 - 1
                } else {
                    -1
                },
                children: [t.left_child[n], t.right_child[n]],
            })
            .collect();
        let mut depth = 0;
        let mut stack = if ni > 0 { vec![(0i32, 1usize)] } else { Vec::new() };
        while let Some((n, d)) = stack.pop() {
            depth = depth.max(d);
            let node = &nodes[n as usize];
            for c in node.children {
                if c >= 0 {
                    stack.push((c, d + 1));
                }
            }
        }
        Self { nodes, depth, cat_bits }
    }

    /// `step` for trees with categorical nodes (upstream `CategoricalDecision`).
    #[inline(always)]
    fn step_any(&self, node: i32, row: &[f64]) -> i32 {
        if node < 0 {
            return node;
        }
        let n = &self.nodes[node as usize];
        if n.cat < 0 {
            return self.step(node, row);
        }
        let v = row[n.feature as usize];
        let left = !v.is_nan() && {
            let c = v as i32;
            c >= 0 && find_in_bitset(&self.cat_bits[n.cat as usize], c)
        };
        n.children[!left as usize]
    }

    #[inline(always)]
    fn step(&self, node: i32, row: &[f64]) -> i32 {
        let n = unsafe { self.nodes.get_unchecked(node.max(0) as usize) };
        let v = unsafe { *row.get_unchecked(n.feature as usize) };
        let nan = v.is_nan() as u8;
        let nan_split = n.missing_type == MissingType::NaN as i8;
        let v = if nan == 1 && !nan_split { 0.0 } else { v };
        let zero = (v >= -K_ZERO_THRESHOLD) as u8 & (v <= K_ZERO_THRESHOLD) as u8;
        let is_default = (n.missing_type == MissingType::Zero as i8) as u8 & zero | nan_split as u8 & nan;
        let le = (v <= n.threshold) as u8;
        let left = (is_default & n.default_left as u8) | (!is_default & 1 & le);
        let next = n.children[(1 - left) as usize];
        // leaves (negative) stay put
        let keep = node >> 31;
        (next & !keep) | (node & keep)
    }

    /// Leaf indices of `rows.len() / stride` rows (row-major, `stride` wide).
    #[inline]
    fn leaves(&self, rows: &[f64], stride: usize, out: &mut [usize]) {
        let nb = out.len();
        if self.nodes.is_empty() {
            out.fill(0);
            return;
        }
        let mut node = [0i32; BLOCK];
        if self.cat_bits.is_empty() {
            for _ in 0..self.depth {
                for r in 0..nb {
                    node[r] = self.step(node[r], &rows[r * stride..(r + 1) * stride]);
                }
            }
        } else {
            for _ in 0..self.depth {
                for r in 0..nb {
                    node[r] = self.step_any(node[r], &rows[r * stride..(r + 1) * stride]);
                }
            }
        }
        for r in 0..nb {
            out[r] = !node[r] as usize;
        }
    }
}

/// Prediction early stopping: stop adding iterations to a row once its
/// margin exceeds `margin_threshold`, checked every `round_period`
/// iterations.
///
/// upstream: src/boosting/prediction_early_stop.cpp.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictEarlyStop {
    pub round_period: usize,
    pub margin_threshold: f64,
    /// Margin between the two largest raw scores (else `2 |score|`).
    pub multiclass: bool,
}

impl PredictEarlyStop {
    /// upstream: `CreateBinary` / `CreateMulticlass` callbacks.
    #[inline]
    fn should_stop(&self, raw: &[f64]) -> bool {
        let margin = if self.multiclass {
            let (mut a, mut b) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &v in raw {
                if v > a {
                    b = a;
                    a = v;
                } else if v > b {
                    b = v;
                }
            }
            a - b
        } else {
            2.0 * raw[0].abs()
        };
        margin > self.margin_threshold
    }
}

impl Gbdt {
    /// Early stopping for a prediction, as upstream's `Predictor` sets it up:
    /// only when requested and the objective does not need accurate
    /// predictions (binary, multiclass, ranking).
    pub fn prediction_early_stop(&self, enabled: bool, freq: i32, margin: f64) -> Result<Option<PredictEarlyStop>> {
        // upstream: RF::NeedAccuratePrediction is always true
        if !enabled || self.is_rf || self.objective.as_ref().is_none_or(|o| o.need_accurate_prediction()) {
            return Ok(None);
        }
        if freq <= 0 {
            return Err(LgbmError::InvalidParameter("Check failed: (early_stop_freq) > (0)".into()));
        }
        if !(margin >= 0.0) {
            return Err(LgbmError::InvalidParameter("Check failed: (early_stop_margin) >= (0)".into()));
        }
        Ok(Some(PredictEarlyStop { round_period: freq as usize, margin_threshold: margin, multiclass: self.num_class != 1 }))
    }

    /// upstream: `GBDT::InitPredict` iteration window.
    fn predict_window(&self, start_iteration: i32, num_iteration: i32) -> (usize, usize) {
        let total = self.models.len() / self.num_tree_per_iteration;
        let start = (start_iteration.max(0) as usize).min(total);
        let num = if num_iteration > 0 {
            (num_iteration as usize).min(total - start)
        } else {
            total - start
        };
        (start, num)
    }

    /// Predict for a dense matrix. Output layout is row-major:
    /// `nrows x num_tree_per_iteration` for Normal/Raw, and
    /// `nrows x (num_iteration * num_tree_per_iteration)` for LeafIndex.
    pub fn predict(
        &self,
        mat: &DenseMatrix<'_>,
        kind: PredictKind,
        start_iteration: i32,
        num_iteration: i32,
    ) -> Result<Vec<f64>> {
        self.predict_matrix(&Matrix::Dense(*mat), kind, start_iteration, num_iteration)
    }

    /// Like [`Gbdt::predict`], for dense or sparse (CSR/CSC) input.
    pub fn predict_matrix(
        &self,
        mat: &Matrix<'_>,
        kind: PredictKind,
        start_iteration: i32,
        num_iteration: i32,
    ) -> Result<Vec<f64>> {
        self.predict_matrix_early_stop(mat, kind, start_iteration, num_iteration, None)
    }

    /// [`Gbdt::predict_matrix`] with optional early stopping (see
    /// [`Gbdt::prediction_early_stop`]); it only affects Normal and Raw
    /// predictions.
    pub fn predict_matrix_early_stop(
        &self,
        mat: &Matrix<'_>,
        kind: PredictKind,
        start_iteration: i32,
        num_iteration: i32,
        early_stop: Option<PredictEarlyStop>,
    ) -> Result<Vec<f64>> {
        if kind == PredictKind::Contrib {
            return self.predict_contrib(mat, start_iteration, num_iteration);
        }
        self.check_num_features(mat)?;
        let ntpi = self.num_tree_per_iteration;
        let (start, num) = self.predict_window(start_iteration, num_iteration);
        let models = &self.models[start * ntpi..(start + num) * ntpi];
        let width = if kind == PredictKind::LeafIndex { models.len() } else { ntpi };
        let ncol = mat.ncols();
        let stride = ncol + 1;
        let flat: Vec<FlatTree> = models.iter().map(|t| FlatTree::new(t, ncol)).collect();
        let mut out = vec![0.0; mat.nrows() * width];
        let objective = self.objective.as_ref();
        let average_output = self.average_output;
        let reader = mat.rows();
        self.install(|| {
            out.par_chunks_mut((width * BLOCK).max(1)).enumerate().for_each_init(
                || (vec![0.0f64; stride * BLOCK], vec![0.0f64; ntpi * BLOCK], [0usize; BLOCK]),
                |(rows, raw, leaf), (b, o)| {
                    let r0 = b * BLOCK;
                    let nb = if width == 0 { (mat.nrows() - r0).min(BLOCK) } else { o.len() / width };
                    for r in 0..nb {
                        let row = &mut rows[r * stride..(r + 1) * stride];
                        reader.row_into(r0 + r, &mut row[..ncol]);
                        for v in row[..ncol].iter_mut() {
                            // upstream predictor drops |v| <= kZeroThreshold (sparse row pairs)
                            if !v.is_nan() && v.abs() <= K_ZERO_THRESHOLD {
                                *v = 0.0;
                            }
                        }
                        row[ncol] = 0.0;
                    }
                    let leaf = &mut leaf[..nb];
                    match kind {
                        PredictKind::LeafIndex => {
                            for (j, t) in flat.iter().enumerate() {
                                t.leaves(rows, stride, leaf);
                                for r in 0..nb {
                                    o[r * width + j] = leaf[r] as f64;
                                }
                            }
                        }
                        _ => {
                            let raw = &mut raw[..nb * ntpi];
                            raw.fill(0.0);
                            let mut stopped = [false; BLOCK];
                            let mut active = nb;
                            for it in 0..num {
                                for k in 0..ntpi {
                                    let m = it * ntpi + k;
                                    flat[m].leaves(rows, stride, leaf);
                                    let values = &models[m].leaf_value;
                                    for r in 0..nb {
                                        if !stopped[r] {
                                            raw[r * ntpi + k] += values[leaf[r]];
                                        }
                                    }
                                }
                                if let Some(es) = early_stop
                                    && (it + 1) % es.round_period == 0
                                {
                                    for r in 0..nb {
                                        if !stopped[r] && es.should_stop(&raw[r * ntpi..(r + 1) * ntpi]) {
                                            stopped[r] = true;
                                            active -= 1;
                                        }
                                    }
                                    if active == 0 {
                                        break;
                                    }
                                }
                            }
                            // upstream GBDT::Predict: average_output divides by the window size
                            if kind == PredictKind::Normal && average_output {
                                for v in raw.iter_mut() {
                                    *v /= num as f64;
                                }
                            }
                            for r in 0..nb {
                                let (src, dst) = (&raw[r * ntpi..(r + 1) * ntpi], &mut o[r * width..(r + 1) * width]);
                                match (kind, objective) {
                                    (PredictKind::Normal, Some(obj)) => obj.convert_output(src, dst),
                                    _ => dst.copy_from_slice(src),
                                }
                            }
                        }
                    }
                },
            );
        });
        Ok(out)
    }

    fn check_num_features(&self, mat: &Matrix<'_>) -> Result<()> {
        if mat.ncols() != self.num_feature() {
            return Err(LgbmError::InvalidData(format!(
                "The number of features in data ({}) is not the same as it was in training data ({}).",
                mat.ncols(),
                self.num_feature()
            )));
        }
        Ok(())
    }

    /// Per-row SHAP values, `nrows x (num_tree_per_iteration * (num_feature + 1))`
    /// row-major; the last entry of each class block is the expected value.
    ///
    /// upstream: `GBDT::PredictContrib`, `Predictor` (`predict_contrib`).
    pub fn predict_contrib(&self, mat: &Matrix<'_>, start_iteration: i32, num_iteration: i32) -> Result<Vec<f64>> {
        self.check_num_features(mat)?;
        let nf = self.num_feature();
        let ntpi = self.num_tree_per_iteration;
        let width = ntpi * (nf + 1);
        let mut out = vec![0.0; mat.nrows() * width];
        self.for_each_contrib_row(mat, start_iteration, num_iteration, &mut out, width, |_, o, phi| {
            o.copy_from_slice(phi);
        });
        Ok(out)
    }

    /// SHAP values as one sparse matrix per class, with `num_feature + 1`
    /// columns, in CSR (`csr`) or CSC layout.
    ///
    /// Like upstream, every row stores the columns of all features split on
    /// by that class's trees, plus the expected-value column, even where the
    /// value is 0. Column indices within a CSR row are sorted (upstream's
    /// order is that of an `unordered_map`).
    ///
    /// upstream: `Booster::PredictSparseCSR`, `Booster::PredictSparseCSC`,
    /// `GBDT::PredictContribByMap`.
    pub fn predict_contrib_sparse(
        &self,
        mat: &Matrix<'_>,
        start_iteration: i32,
        num_iteration: i32,
        csr: bool,
    ) -> Result<Vec<SparseContrib>> {
        self.check_num_features(mat)?;
        let nf = self.num_feature();
        let ntpi = self.num_tree_per_iteration;
        let (start, num) = self.predict_window(start_iteration, num_iteration);
        let keys: Vec<Vec<usize>> = (0..ntpi)
            .map(|k| {
                let mut used = vec![false; nf + 1];
                for it in start..start + num {
                    let t = &self.models[it * ntpi + k];
                    used[nf] = true;
                    if t.num_leaves > 1 {
                        for &f in &t.split_feature {
                            used[f as usize] = true;
                        }
                    }
                }
                (0..=nf).filter(|&c| used[c]).collect()
            })
            .collect();
        let offsets: Vec<usize> = std::iter::once(0).chain(keys.iter().scan(0, |s, k| { *s += k.len(); Some(*s) })).collect();
        let width = offsets[ntpi];
        let nrows = mat.nrows();
        let mut values = vec![0.0; nrows * width];
        self.for_each_contrib_row(mat, start_iteration, num_iteration, &mut values, width, |_, o, phi| {
            for (k, ks) in keys.iter().enumerate() {
                let block = &phi[k * (nf + 1)..(k + 1) * (nf + 1)];
                for (dst, &c) in o[offsets[k]..offsets[k + 1]].iter_mut().zip(ks) {
                    *dst = block[c];
                }
            }
        });
        if csr && ntpi == 1 {
            let nk = keys[0].len();
            return Ok(vec![SparseContrib {
                indptr: (0..=nrows).map(|r| (r * nk) as i64).collect(),
                indices: (0..nrows).flat_map(|_| keys[0].iter().map(|&c| c as i32)).collect(),
                values,
            }]);
        }
        Ok(keys
            .iter()
            .enumerate()
            .map(|(k, ks)| {
                let nk = ks.len();
                let cell = |r: usize, j: usize| values[r * width + offsets[k] + j];
                if csr {
                    SparseContrib {
                        indptr: (0..=nrows).map(|r| (r * nk) as i64).collect(),
                        indices: (0..nrows).flat_map(|_| ks.iter().map(|&c| c as i32)).collect(),
                        values: (0..nrows).flat_map(|r| (0..nk).map(move |j| (r, j))).map(|(r, j)| cell(r, j)).collect(),
                    }
                } else {
                    let mut indptr = Vec::with_capacity(nf + 2);
                    let mut stored = 0i64;
                    indptr.push(0);
                    let mut next_key = ks.iter().peekable();
                    for c in 0..=nf {
                        if next_key.peek() == Some(&&c) {
                            next_key.next();
                            stored += nrows as i64;
                        }
                        indptr.push(stored);
                    }
                    SparseContrib {
                        indptr,
                        indices: (0..nk).flat_map(|_| 0..nrows as i32).collect(),
                        values: (0..nk).flat_map(|j| (0..nrows).map(move |r| (r, j))).map(|(r, j)| cell(r, j)).collect(),
                    }
                }
            })
            .collect())
    }

    /// Run TreeSHAP for every row and hand `(row, out_row, phi)` to `emit`,
    /// where `phi` is `num_tree_per_iteration * (num_feature + 1)` wide.
    fn for_each_contrib_row(
        &self,
        mat: &Matrix<'_>,
        start_iteration: i32,
        num_iteration: i32,
        out: &mut [f64],
        width: usize,
        emit: impl Fn(usize, &mut [f64], &[f64]) + Sync + Send,
    ) {
        let nf = self.num_feature();
        let ntpi = self.num_tree_per_iteration;
        let (start, num) = self.predict_window(start_iteration, num_iteration);
        let models = &self.models[start * ntpi..(start + num) * ntpi];
        let expected: Vec<f64> = models.iter().map(|t| t.expected_value()).collect();
        let reader = mat.rows();
        if width == 0 {
            return;
        }
        self.install(|| {
            out.par_chunks_mut(width).enumerate().for_each_init(
                || (vec![0.0f64; nf], vec![0.0f64; ntpi * (nf + 1)], Vec::new()),
                |(row, phi, path), (r, o)| {
                    reader.row_into(r, row);
                    for v in row.iter_mut() {
                        // upstream predictor drops |v| <= kZeroThreshold (sparse row pairs)
                        if !v.is_nan() && v.abs() <= K_ZERO_THRESHOLD {
                            *v = 0.0;
                        }
                    }
                    phi.fill(0.0);
                    for it in 0..num {
                        for k in 0..ntpi {
                            let m = it * ntpi + k;
                            models[m].predict_contrib(row, nf, expected[m], &mut phi[k * (nf + 1)..(k + 1) * (nf + 1)], path);
                        }
                    }
                    emit(r, o, phi);
                },
            );
        });
    }
}

/// Options of [`Gbdt::predict_file`] besides the prediction kind and window.
#[derive(Debug, Clone, Copy, Default)]
pub struct PredictFileOptions {
    /// The data file has a header line; its names select the model's features.
    pub header: bool,
    pub disable_shape_check: bool,
    pub precise_float_parser: bool,
    pub early_stop: Option<PredictEarlyStop>,
}

/// Lines predicted per batch in [`Gbdt::predict_file`].
const FILE_BATCH: usize = 1 << 16;

impl Gbdt {
    /// Predict the rows of a CSV, TSV or LibSVM file and write one line per
    /// row to `result_filename`: the row's outputs, tab-separated, with 17
    /// significant digits. Returns upstream's warnings.
    ///
    /// upstream: `Predictor::Predict(data_filename, result_filename, ...)`.
    pub fn predict_file(
        &self,
        data_filename: &str,
        result_filename: &str,
        kind: PredictKind,
        start_iteration: i32,
        num_iteration: i32,
        opts: PredictFileOptions,
    ) -> Result<Vec<String>> {
        use std::io::Write;
        let file = std::fs::File::create(result_filename).map_err(|_| {
            LgbmError::InvalidParameter(format!("Prediction results file {result_filename} cannot be created"))
        })?;
        let mut writer = std::io::BufWriter::new(file);
        let mut warnings = Vec::new();
        if let Ok((_, crate::binary::DataFileKind::Binary)) = crate::binary::detect_data_file_exact(data_filename) {
            // upstream's format detection rejects its own binary files with this message
            return Err(LgbmError::InvalidData(
                "Unknown format of training data. Only CSV, TSV, and LibSVM (zero-based) formatted text files are supported."
                    .into(),
            ));
        }
        let nf = self.num_feature();
        let label_idx = if opts.header { -1 } else { self.label_index };
        let parser =
            Parser::create(data_filename, opts.header, nf as i32, label_idx, opts.precise_float_parser, &mut warnings)?;
        if !opts.header && !opts.disable_shape_check && parser.num_features() != nf as i32 {
            return Err(LgbmError::InvalidData(format!(
                "The number of features in data ({}) is not the same as it was in training data ({nf}).\n\
                 You can set ``predict_disable_shape_check=true`` to discard this error, but please be aware what you are doing.",
                parser.num_features()
            )));
        }
        let mut skip = 0;
        let mut remap: Option<Vec<i32>> = None;
        if opts.header {
            let (first_line, bytes) = text_parser::read_header(data_filename)?;
            skip = bytes;
            let words: Vec<&str> = first_line.split(['\t', ',']).filter(|t| !t.is_empty()).collect();
            let mut position = std::collections::HashMap::new();
            for (i, w) in words.iter().enumerate() {
                if position.insert(*w, i).is_some() {
                    return Err(LgbmError::InvalidData(format!("Feature ({w}) appears more than one time.")));
                }
            }
            let mut remapper = vec![-1i32; (parser.num_features().max(0) as usize).max(words.len())];
            for (i, name) in self.feature_names.iter().enumerate() {
                match position.get(name.as_str()) {
                    Some(&p) => remapper[p] = i as i32,
                    None => warnings.push(format!(
                        "Feature ({name}) is missed in data file. If it is weight/query/group/ignore_column, \
                         you can ignore this warning."
                    )),
                }
            }
            if remapper.iter().enumerate().any(|(i, &r)| r >= 0 && r != i as i32) {
                remap = Some(remapper);
            }
        }
        let predict_batch = |lines: &[Vec<u8>], writer: &mut std::io::BufWriter<std::fs::File>| -> Result<()> {
            let rows: Vec<Vec<(i32, f64)>> = lines
                .par_iter()
                .map(|l| {
                    let mut f = Vec::new();
                    parser.parse_line(l, &mut f)?;
                    if let Some(r) = &remap {
                        f = f
                            .into_iter()
                            .filter_map(|(i, v)| {
                                r.get(i as usize).copied().filter(|&j| i >= 0 && j >= 0).map(|j| (j, v))
                            })
                            .collect();
                    }
                    Ok(f)
                })
                .collect::<Result<_>>()?;
            let mut indptr = vec![0i64];
            let (mut indices, mut values) = (Vec::new(), Vec::new());
            for row in &rows {
                for &(i, v) in row {
                    if i >= 0 && (i as usize) < nf {
                        indices.push(i);
                        values.push(v);
                    }
                }
                indptr.push(indices.len() as i64);
            }
            let mat = Matrix::Sparse(SparseMatrix::new(
                SparseIndptr::I64(&indptr),
                &indices,
                DenseValues::F64(&values),
                rows.len(),
                nf,
                true,
            )?);
            let out = self.predict_matrix_early_stop(&mat, kind, start_iteration, num_iteration, opts.early_stop)?;
            let width = if rows.is_empty() { 0 } else { out.len() / rows.len() };
            let text: Vec<String> = out
                .par_chunks(width.max(1))
                .map(|r| r.iter().map(|&v| fmt_g17(v)).collect::<Vec<_>>().join("\t"))
                .collect();
            for t in text {
                writer.write_all(t.as_bytes()).and_then(|_| writer.write_all(b"\n")).map_err(write_err)?;
            }
            Ok(())
        };
        let mut batch: Vec<Vec<u8>> = Vec::new();
        text_parser::for_each_line(data_filename, skip, |l| {
            batch.push(l.to_vec());
            if batch.len() == FILE_BATCH {
                predict_batch(&batch, &mut writer)?;
                batch.clear();
            }
            Ok(())
        })?;
        if !batch.is_empty() {
            predict_batch(&batch, &mut writer)?;
        }
        writer.flush().map_err(write_err)?;
        Ok(warnings)
    }
}

fn write_err(e: std::io::Error) -> LgbmError {
    LgbmError::InvalidData(format!("Could not write prediction results: {e}"))
}

/// One class's SHAP values from [`Gbdt::predict_contrib_sparse`].
#[derive(Debug, Clone, PartialEq)]
pub struct SparseContrib {
    pub indptr: Vec<i64>,
    pub indices: Vec<i32>,
    pub values: Vec<f64>,
}
