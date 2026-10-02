//! Linear trees: a linear model of the raw feature values in every leaf.
//!
//! upstream: src/treelearner/linear_tree_learner.cpp and .h. The normal
//! equations are solved the way upstream's `-XTHX.fullPivLu().inverse() * XTg`
//! evaluates with Eigen 3.4.0 on x86-64 (SSE2 kernels without FMA), so the
//! coefficients match bit for bit.

use crate::binning::BinType;
use crate::consts::K_ZERO_THRESHOLD;
use crate::dataset::Dataset;
use crate::error::{LgbmError, Result};
use crate::threading::ThreadTeam;
use crate::tree::Tree;

/// Largest system (leaf features + 1) solved. Up to this size Eigen's
/// triangular solves keep the whole matrix in one depth block (`kc`) for any
/// L1 cache of at least 32 KiB, which the emulation below assumes.
const MAX_SYSTEM: usize = 120;

/// Rows above which upstream splits the accumulation over its threads.
const PARALLEL_ROWS: usize = 1024;

/// upstream `LinearTreeLearner` state.
#[derive(Debug, Clone)]
pub(crate) struct Linear {
    /// upstream `contains_nan_`, per inner feature of the data given to `new`.
    contains_nan: Vec<bool>,
    any_nan: bool,
    pub linear_lambda: f64,
}

/// The rows a linear fit visits, in upstream's iteration order: all rows, or
/// the bag (upstream's learner then iterates the subset dataset).
pub(crate) struct LinearRows<'a> {
    /// Leaf of each row (-1: not in the partition).
    pub leaf_map: &'a [i32],
    pub bag: Option<&'a [u32]>,
    pub num_data: usize,
    /// Rows per leaf (upstream `data_partition_->leaf_count`).
    pub leaf_count: &'a (dyn Fn(usize) -> usize + Sync),
}

impl LinearRows<'_> {
    #[inline]
    fn row(&self, p: usize) -> usize {
        match self.bag {
            Some(b) => b[p] as usize,
            None => p,
        }
    }
}

/// libgomp's `schedule(static)` share of `n` iterations for thread `tid` of `t`.
fn static_range(n: usize, t: usize, tid: usize) -> std::ops::Range<usize> {
    let (q, r) = (n / t, n % t);
    let start = tid * q + tid.min(r);
    start..start + q + usize::from(tid < r)
}

impl Linear {
    /// upstream: `LinearTreeLearner::InitLinear`. `data` must keep raw values.
    pub fn new(data: &Dataset, linear_lambda: f64) -> Self {
        let contains_nan: Vec<bool> = (0..data.num_features())
            .map(|f| data.feature_bin_mapper(f).bin_type == BinType::Numerical && data.raw(f).iter().any(|v| v.is_nan()))
            .collect();
        let any_nan = contains_nan.iter().any(|&c| c);
        Self { contains_nan, any_nan, linear_lambda }
    }

    /// Whether a split feature of `tree` contains NaN (upstream's `has_nan`).
    fn tree_has_nan(&self, data: &Dataset, tree: &Tree) -> bool {
        self.any_nan
            && tree.split_feature[..tree.num_leaves - 1].iter().any(|&f| {
                data.inner_feature_index(f as usize).and_then(|i| self.contains_nan.get(i).copied()).unwrap_or(false)
            })
    }

    /// Fit the linear model of every leaf of `tree`.
    ///
    /// upstream: `LinearTreeLearner::CalculateLinear`.
    #[allow(clippy::too_many_arguments)]
    pub fn calculate(
        &self,
        tree: &mut Tree,
        data: &Dataset,
        team: &ThreadTeam,
        rows: &LinearRows<'_>,
        grad: &[f32],
        hess: &[f32],
        decay_rate: f64,
        is_refit: bool,
        is_first_tree: bool,
    ) -> Result<()> {
        let num_leaves = tree.num_leaves;
        tree.make_linear();
        if is_first_tree {
            for leaf in 0..num_leaves {
                tree.set_leaf_const(leaf, tree.leaf_value[leaf]);
            }
            return Ok(());
        }
        let has_nan = self.tree_has_nan(data, tree);
        let branch = if is_refit { tree.leaf_features.clone() } else { branch_features(tree) };
        let mut leaf_features: Vec<Vec<usize>> = Vec::with_capacity(num_leaves);
        for mut raw in branch {
            raw.sort_unstable();
            raw.dedup();
            let mut feats = Vec::with_capacity(raw.len());
            for r in raw {
                let inner = (r >= 0).then(|| data.inner_feature_index(r as usize)).flatten().ok_or_else(|| {
                    LgbmError::InvalidData(format!("linear model feature {r} is not used by the training data"))
                })?;
                if data.feature_bin_mapper(inner).bin_type == BinType::Numerical {
                    feats.push(inner);
                }
            }
            if feats.len() + 1 > MAX_SYSTEM {
                return Err(LgbmError::Unsupported(format!(
                    "linear leaf models with more than {} features",
                    MAX_SYSTEM - 1
                )));
            }
            leaf_features.push(feats);
        }
        let raws: Vec<Vec<&[f32]>> = leaf_features.iter().map(|fs| fs.iter().map(|&f| data.raw(f)).collect()).collect();
        let tri = |k: usize| (k + 1) * (k + 2) / 2;
        let max_feat = leaf_features.iter().map(Vec::len).max().unwrap_or(0);
        let nthreads = if rows.num_data > PARALLEL_ROWS { team.num_threads() } else { 1 };
        // per-thread partial sums, combined in thread order as upstream does
        let partials = team.map(nthreads, rows.num_data * (max_feat + 2), |tid| {
            let mut xthx: Vec<Vec<f64>> = leaf_features.iter().map(|f| vec![0.0; tri(f.len())]).collect();
            let mut xtg: Vec<Vec<f64>> = leaf_features.iter().map(|f| vec![0.0; f.len() + 1]).collect();
            let mut nonzero = vec![0usize; num_leaves];
            let mut curr = vec![0f32; max_feat + 1];
            for p in static_range(rows.num_data, nthreads, tid) {
                let i = rows.row(p);
                let leaf = rows.leaf_map[i];
                if leaf < 0 {
                    continue;
                }
                let leaf = leaf as usize;
                let feats = &raws[leaf];
                let nf = feats.len();
                let mut nan_found = false;
                for (f, col) in feats.iter().enumerate() {
                    let v = col[i];
                    if has_nan {
                        if v.is_nan() {
                            nan_found = true;
                            break;
                        }
                        nonzero[leaf] += 1;
                    }
                    curr[f] = v;
                }
                if nan_found {
                    continue;
                }
                curr[nf] = 1.0;
                // upstream: float h, g; double f1_val
                let (h, g) = (hess[i] as f64, grad[i] as f64);
                let (a, b) = (&mut xthx[leaf], &mut xtg[leaf]);
                let mut j = 0;
                for f1 in 0..=nf {
                    let mut v1 = curr[f1] as f64;
                    b[f1] += v1 * g;
                    v1 *= h;
                    for &c in &curr[f1..=nf] {
                        a[j] += v1 * c as f64;
                        j += 1;
                    }
                }
            }
            (xthx, xtg, nonzero)
        });
        let mut xthx: Vec<Vec<f64>> = leaf_features.iter().map(|f| vec![0.0; tri(f.len())]).collect();
        let mut xtg: Vec<Vec<f64>> = leaf_features.iter().map(|f| vec![0.0; f.len() + 1]).collect();
        let mut total_nonzero = vec![0usize; num_leaves];
        for (a, b, nz) in &partials {
            for leaf in 0..num_leaves {
                for (t, &v) in xthx[leaf].iter_mut().zip(&a[leaf]) {
                    *t += v;
                }
                for (t, &v) in xtg[leaf].iter_mut().zip(&b[leaf]) {
                    *t += v;
                }
                total_nonzero[leaf] += nz[leaf];
            }
        }
        if !has_nan {
            for (leaf, t) in total_nonzero.iter_mut().enumerate() {
                *t = (rows.leaf_count)(leaf);
            }
        }
        let shrinkage = tree.shrinkage;
        for leaf in 0..num_leaves {
            let feats = &leaf_features[leaf];
            let nf = feats.len();
            if total_nonzero[leaf] < nf + 1 {
                if is_refit {
                    let old_const = tree.leaf_const[leaf];
                    let v = decay_rate * old_const + (1.0 - decay_rate) * tree.leaf_value[leaf] * shrinkage;
                    tree.set_leaf_const(leaf, v);
                    tree.set_leaf_coeffs(leaf, &vec![0.0; nf]);
                    tree.leaf_features_inner[leaf] = feats.iter().map(|&f| f as i32).collect();
                } else {
                    tree.set_leaf_const(leaf, tree.leaf_value[leaf]);
                }
                continue;
            }
            let n = nf + 1;
            let mut a = vec![0.0f64; n * n];
            let mut j = 0;
            for f1 in 0..n {
                for f2 in f1..n {
                    a[f1 + f2 * n] = xthx[leaf][j];
                    a[f2 + f1 * n] = a[f1 + f2 * n];
                    if f1 == f2 && f1 < nf {
                        a[f1 + f2 * n] += self.linear_lambda;
                    }
                    j += 1;
                }
            }
            let coeffs = neg_inverse_times(a, &xtg[leaf], n);
            let mut coeffs_vec = Vec::new();
            let mut features_new: Vec<usize> = Vec::new();
            for i in 0..nf {
                if is_refit {
                    features_new.push(feats[i]);
                    let old = tree.leaf_coeff[leaf].get(i).copied().unwrap_or(0.0);
                    coeffs_vec.push(decay_rate * old + (1.0 - decay_rate) * coeffs[i] * shrinkage);
                } else if coeffs[i] < -K_ZERO_THRESHOLD || coeffs[i] > K_ZERO_THRESHOLD {
                    coeffs_vec.push(coeffs[i]);
                    features_new.push(feats[i]);
                }
            }
            tree.leaf_features_inner[leaf] = features_new.iter().map(|&f| f as i32).collect();
            tree.leaf_features[leaf] = features_new.iter().map(|&f| data.real_feature_index(f) as i32).collect();
            tree.set_leaf_coeffs(leaf, &coeffs_vec);
            if is_refit {
                let old_const = tree.leaf_const[leaf];
                tree.set_leaf_const(leaf, decay_rate * old_const + (1.0 - decay_rate) * coeffs[nf] * shrinkage);
            } else {
                tree.set_leaf_const(leaf, coeffs[nf]);
            }
        }
        Ok(())
    }

    /// Add the linear outputs of `tree` to `score` for the rows mapped to a
    /// leaf by `leaf_map`.
    ///
    /// upstream: `LinearTreeLearner::AddPredictionToScore`.
    pub fn add_prediction_to_score(&self, tree: &Tree, data: &Dataset, leaf_map: &[i32], score: &mut [f64]) {
        let has_nan = self.tree_has_nan(data, tree);
        let feats: Vec<Vec<&[f32]>> = (0..tree.num_leaves)
            .map(|l| tree.leaf_features_inner[l].iter().map(|&f| data.raw(f as usize)).collect())
            .collect();
        for (i, (&leaf, s)) in leaf_map.iter().zip(score.iter_mut()).enumerate() {
            if leaf < 0 {
                continue;
            }
            let leaf = leaf as usize;
            let mut output = tree.leaf_const[leaf];
            let mut nan_found = false;
            for (col, &c) in feats[leaf].iter().zip(&tree.leaf_coeff[leaf]) {
                let v = col[i];
                if has_nan && v.is_nan() {
                    nan_found = true;
                    break;
                }
                output += v as f64 * c;
            }
            *s += if nan_found { tree.leaf_value[leaf] } else { output };
        }
    }
}

/// Real split features on the path to each leaf (upstream `branch_features_`).
fn branch_features(tree: &Tree) -> Vec<Vec<i32>> {
    let mut out = vec![Vec::new(); tree.num_leaves];
    if tree.num_leaves <= 1 {
        return out;
    }
    let mut stack: Vec<(i32, Vec<i32>)> = vec![(0, Vec::new())];
    while let Some((node, mut path)) = stack.pop() {
        if node < 0 {
            out[!node as usize] = path;
            continue;
        }
        let n = node as usize;
        path.push(tree.split_feature[n]);
        stack.push((tree.left_child[n], path.clone()));
        stack.push((tree.right_child[n], path));
    }
    out
}

/// `-(A^-1 b)` for the column-major `n x n` matrix `a`, as Eigen evaluates
/// `-a.fullPivLu().inverse() * b`: a coefficient-based product for `n <= 9`
/// (`EIGEN_GEMM_TO_COEFFBASED_THRESHOLD`), else the column-by-column GEMV of
/// an expression without direct access.
fn neg_inverse_times(a: Vec<f64>, b: &[f64], n: usize) -> Vec<f64> {
    let inv = full_piv_lu_inverse(a, n);
    (0..n)
        .map(|i| {
            if 2 * n + 1 < 20 {
                let mut s = inv[i] * b[0];
                for k in 1..n {
                    s += inv[i + k * n] * b[k];
                }
                -s
            } else {
                let mut d = 0.0;
                for k in 0..n {
                    d += b[k] * -inv[i + k * n];
                }
                d
            }
        })
        .collect()
}

/// Eigen 3.4.0 `FullPivLU<MatrixXd>(a).inverse()` (column-major).
fn full_piv_lu_inverse(mut lu: Vec<f64>, n: usize) -> Vec<f64> {
    let at = |i: usize, j: usize| i + j * n;
    // FullPivLU::computeInPlace
    let mut rows_t: Vec<usize> = (0..n).collect();
    let mut cols_t: Vec<usize> = (0..n).collect();
    let mut nonzero_pivots = n;
    let mut max_pivot = 0.0f64;
    for k in 0..n {
        // maxCoeff of |corner| visits column by column; ties keep the first
        let (mut best, mut br, mut bc) = (lu[at(k, k)].abs(), k, k);
        for j in k..n {
            for i in k..n {
                if (i, j) != (k, k) {
                    let v = lu[at(i, j)].abs();
                    if v > best {
                        (best, br, bc) = (v, i, j);
                    }
                }
            }
        }
        if best == 0.0 {
            nonzero_pivots = k;
            for i in k..n {
                rows_t[i] = i;
                cols_t[i] = i;
            }
            break;
        }
        if best > max_pivot {
            max_pivot = best;
        }
        rows_t[k] = br;
        cols_t[k] = bc;
        if k != br {
            for j in 0..n {
                lu.swap(at(k, j), at(br, j));
            }
        }
        if k != bc {
            for i in 0..n {
                lu.swap(at(i, k), at(i, bc));
            }
        }
        let p = lu[at(k, k)];
        for i in k + 1..n {
            lu[at(i, k)] /= p;
        }
        for j in k + 1..n {
            let r = lu[at(k, j)];
            for i in k + 1..n {
                lu[at(i, j)] -= r * lu[at(i, k)];
            }
        }
    }
    let mut perm_p: Vec<usize> = (0..n).collect();
    for k in (0..n).rev() {
        perm_p.swap(k, rows_t[k]);
    }
    let mut perm_q: Vec<usize> = (0..n).collect();
    for (k, &c) in cols_t.iter().enumerate() {
        perm_q.swap(k, c);
    }
    // FullPivLU::rank with the default threshold (epsilon * diagonal size)
    let threshold = max_pivot.abs() * (f64::EPSILON * n as f64);
    let rank = (0..nonzero_pivots).filter(|&i| lu[at(i, i)].abs() > threshold).count();
    let mut dst = vec![0.0f64; n * n];
    if rank == 0 {
        return dst;
    }
    // FullPivLU::_solve_impl with the identity as right-hand side
    let mut c = vec![0.0f64; n * n];
    for i in 0..n {
        c[at(perm_p[i], i)] = 1.0;
    }
    trsm_unit_lower(&lu, n, &mut c, n);
    trsm_upper(&lu, n, rank, &mut c, n);
    for i in 0..rank {
        for j in 0..n {
            dst[at(perm_q[i], j)] = c[at(i, j)];
        }
    }
    dst
}

/// Eigen's `SmallPanelWidth` (`max(mr, nr)`) for doubles with SSE2.
const PANEL: usize = 4;

/// `triangular_solve_matrix<OnTheLeft, UnitLower, ColMajor tri, ColMajor rhs>`
/// on the `size x size` corner of `tri` (stride `ld`) and the `size x cols`
/// top of `c` (stride `ld`), with one depth block.
fn trsm_unit_lower(tri: &[f64], ld: usize, c: &mut [f64], cols: usize) {
    let size = ld;
    let at = |i: usize, j: usize| i + j * ld;
    let mut k1 = 0;
    while k1 < size {
        let w = (size - k1).min(PANEL);
        for k in 0..w {
            let i = k1 + k;
            let rs = w - k - 1;
            let s = i + 1;
            for j in 0..cols {
                let b = c[at(i, j)];
                for i3 in 0..rs {
                    c[at(s + i3, j)] -= b * tri[at(s + i3, i)];
                }
            }
        }
        let length_target = size - k1 - w;
        if length_target > 0 {
            gebp_sub(tri, c, ld, k1 + w, length_target, k1, w, cols);
        }
        k1 += PANEL;
    }
}

/// `triangular_solve_matrix<OnTheLeft, Upper, ...>` on the `size x size`
/// corner of `tri` and the top `size` rows of `c` (both of stride `ld`).
fn trsm_upper(tri: &[f64], ld: usize, size: usize, c: &mut [f64], cols: usize) {
    let at = |i: usize, j: usize| i + j * ld;
    let mut k1 = 0;
    while k1 < size {
        let w = (size - k1).min(PANEL);
        for k in 0..w {
            let i = size - k1 - k - 1;
            let rs = w - k - 1;
            let s = i - rs;
            let a = 1.0 / tri[at(i, i)];
            for j in 0..cols {
                c[at(i, j)] *= a;
                let b = c[at(i, j)];
                for i3 in 0..rs {
                    c[at(s + i3, j)] -= b * tri[at(s + i3, i)];
                }
            }
        }
        let length_target = size - k1 - w;
        if length_target > 0 {
            gebp_sub(tri, c, ld, 0, length_target, size - k1 - w, w, cols);
        }
        k1 += PANEL;
    }
}

/// The GEBP update `C[rows] += -1 * (A[rows, block] * C[block])`: each entry
/// accumulates its `depth` products from zero, in order (`depth < 8`, so no
/// peeled accumulators), and is then added to the target.
#[allow(clippy::too_many_arguments)]
fn gebp_sub(
    tri: &[f64],
    c: &mut [f64],
    ld: usize,
    start_target: usize,
    length_target: usize,
    start_block: usize,
    depth: usize,
    cols: usize,
) {
    let at = |i: usize, j: usize| i + j * ld;
    for j in 0..cols {
        for r in start_target..start_target + length_target {
            let mut acc = 0.0f64;
            for kk in 0..depth {
                acc += tri[at(r, start_block + kk)] * c[at(start_block + kk, j)];
            }
            c[at(r, j)] += -acc;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverse_solves_small_systems() {
        // 3x3 SPD system, column-major
        let a = vec![4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
        let b = [1.0, 2.0, 3.0];
        let x = neg_inverse_times(a.clone(), &b, 3);
        for i in 0..3 {
            let ax: f64 = (0..3).map(|k| a[i + k * 3] * x[k]).sum();
            assert!((ax + b[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn large_systems_solve() {
        let n = 12;
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i + j * n] = 1.0 / (1.0 + (i as f64 - j as f64).abs()) + if i == j { n as f64 } else { 0.0 };
            }
        }
        let b: Vec<f64> = (0..n).map(|i| i as f64 - 3.0).collect();
        let x = neg_inverse_times(a.clone(), &b, n);
        for i in 0..n {
            let ax: f64 = (0..n).map(|k| a[i + k * n] * x[k]).sum();
            assert!((ax + b[i]).abs() < 1e-10, "{i}: {}", ax + b[i]);
        }
    }

    #[test]
    fn singular_systems_use_the_rank() {
        // rank 1: the inverse keeps one pivot
        let a = vec![2.0, 2.0, 2.0, 2.0];
        let inv = full_piv_lu_inverse(a, 2);
        assert_eq!(inv, vec![0.5, 0.0, 0.0, 0.0]);
        assert_eq!(full_piv_lu_inverse(vec![0.0; 4], 2), vec![0.0; 4]);
    }

    #[test]
    fn static_schedule_matches_libgomp() {
        let parts: Vec<_> = (0..4).map(|t| static_range(10, 4, t)).collect();
        assert_eq!(parts, vec![0..3, 3..6, 6..8, 8..10]);
    }
}
