//! Prediction on raw (unbinned) feature values.
//!
//! upstream: src/application/predictor.hpp, src/boosting/gbdt_prediction.cpp.

use rayon::prelude::*;

use crate::binning::MissingType;
use crate::boosting::{Gbdt, PredictKind};
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::DenseMatrix;
use crate::matrix::Matrix;
use crate::error::{LgbmError, Result};
use crate::tree::Tree;

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
    /// `[left, right]`
    children: [i32; 2],
}

/// One tree in a compact layout for branchless traversal; decisions are
/// those of `Tree::numerical_decision`.
struct FlatTree {
    nodes: Vec<Node>,
    depth: usize,
}

impl FlatTree {
    fn new(t: &Tree, ncol: usize) -> Self {
        let ni = t.num_leaves.saturating_sub(1);
        let nodes: Vec<Node> = (0..ni)
            .map(|n| Node {
                threshold: t.threshold[n],
                feature: (t.split_feature[n] as usize).min(ncol) as u32,
                missing_type: (t.decision_type[n] >> 2) & 3,
                default_left: t.decision_type[n] & 2 > 0,
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
        Self { nodes, depth }
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
        for _ in 0..self.depth {
            for r in 0..nb {
                node[r] = self.step(node[r], &rows[r * stride..(r + 1) * stride]);
            }
        }
        for r in 0..nb {
            out[r] = !node[r] as usize;
        }
    }
}

impl Gbdt {
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
        if mat.ncols() != self.num_feature() {
            return Err(LgbmError::InvalidData(format!(
                "The number of features in data ({}) is not the same as it was in training data ({}).",
                mat.ncols(),
                self.num_feature()
            )));
        }
        let ntpi = self.num_tree_per_iteration;
        let (start, num) = self.predict_window(start_iteration, num_iteration);
        let models = &self.models[start * ntpi..(start + num) * ntpi];
        let width = if kind == PredictKind::LeafIndex { models.len() } else { ntpi };
        let ncol = mat.ncols();
        let stride = ncol + 1;
        let flat: Vec<FlatTree> = models.iter().map(|t| FlatTree::new(t, ncol)).collect();
        let mut out = vec![0.0; mat.nrows() * width];
        let objective = self.objective.as_ref();
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
                            for it in 0..num {
                                for k in 0..ntpi {
                                    let m = it * ntpi + k;
                                    flat[m].leaves(rows, stride, leaf);
                                    let values = &models[m].leaf_value;
                                    for r in 0..nb {
                                        raw[r * ntpi + k] += values[leaf[r]];
                                    }
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
}
