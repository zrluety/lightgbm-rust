//! Objective tests: analytic gradients/Hessians vs finite differences of the
//! loss, plus the grouped-objective plumbing.

use super::*;
use crate::dataset::Metadata;

/// Per-row L2 loss 0.5*(s-y)^2 (weighted).
fn l2_loss(s: f64, y: f64, w: f64) -> f64 {
    0.5 * w * (s - y) * (s - y)
}

/// Per-row binary log loss with labels in {0,1}, sigmoid scale `a`.
fn logloss(s: f64, y: f64, w: f64, a: f64) -> f64 {
    let p = 1.0 / (1.0 + (-a * s).exp());
    -w * (y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

fn fd(f: impl Fn(f64) -> f64, x: f64) -> (f64, f64) {
    let h = 1e-4;
    let g = (f(x + h) - f(x - h)) / (2.0 * h);
    let hh = (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
    (g, hh)
}

fn check_row_objective(
    obj: &mut dyn RowObjective,
    meta: &Metadata,
    scores: &[f64],
    loss: impl Fn(usize, f64) -> f64,
) {
    let n = scores.len();
    obj.init(meta, n).unwrap();
    let mut g = vec![0.0f32; n];
    let mut h = vec![0.0f32; n];
    obj.gradients(ScoreView { scores, num_data: n, num_outputs: 1 }, &mut g, &mut h);
    for i in 0..n {
        let (fg, fh) = fd(|s| loss(i, s), scores[i]);
        // f32 gradient storage limits agreement to ~1e-6 relative.
        assert!((g[i] as f64 - fg).abs() <= 1e-5 * (1.0 + fg.abs()), "grad row {i}: {} vs {fg}", g[i]);
        assert!((h[i] as f64 - fh).abs() <= 1e-3 * (1.0 + fh.abs()), "hess row {i}: {} vs {fh}", h[i]);
    }
}

#[test]
fn l2_matches_finite_differences() {
    let label = vec![0.5f32, -1.0, 2.0, 3.25];
    let weight = vec![1.0f32, 0.5, 2.0, 1.5];
    let scores = [0.1, 0.2, -0.3, 4.0];
    let meta = Metadata { label: label.clone(), weight: None, init_score: None };
    let mut o = regression::RegressionL2::with_sqrt(false);
    check_row_objective(&mut o, &meta, &scores, |i, s| l2_loss(s, label[i] as f64, 1.0));
    let meta_w = Metadata { label: label.clone(), weight: Some(weight.clone()), init_score: None };
    let mut o = regression::RegressionL2::with_sqrt(false);
    check_row_objective(&mut o, &meta_w, &scores, |i, s| l2_loss(s, label[i] as f64, weight[i] as f64));
}

#[test]
fn binary_matches_finite_differences() {
    let label = vec![0.0f32, 1.0, 1.0, 0.0, 1.0];
    let weight = vec![1.0f32, 0.5, 2.0, 1.5, 0.25];
    let scores = [0.1, -2.0, 0.7, 3.0, 0.0];
    for sigmoid in [1.0, 0.5, 2.0] {
        let cfg = Config::from_pairs([("objective", "binary"), ("sigmoid", &sigmoid.to_string())]).unwrap();
        let meta = Metadata { label: label.clone(), weight: Some(weight.clone()), init_score: None };
        let mut o = binary::BinaryLogloss::new(&cfg).unwrap();
        check_row_objective(&mut o, &meta, &scores, |i, s| {
            logloss(s, label[i] as f64, weight[i] as f64, sigmoid)
        });
    }
}

#[test]
fn binary_boost_from_score_is_logit_of_mean() {
    let cfg = Config::from_pairs([("objective", "binary")]).unwrap();
    let meta = Metadata { label: vec![1.0, 0.0, 0.0, 0.0], weight: None, init_score: None };
    let mut o = binary::BinaryLogloss::new(&cfg).unwrap();
    o.init(&meta, 4).unwrap();
    assert!((o.boost_from_score(0) - (0.25f64 / 0.75).ln()).abs() < 1e-15);
}

#[test]
fn binary_single_class_needs_no_training() {
    let cfg = Config::from_pairs([("objective", "binary")]).unwrap();
    let meta = Metadata { label: vec![1.0; 3], weight: None, init_score: None };
    let mut o = binary::BinaryLogloss::new(&cfg).unwrap();
    o.init(&meta, 3).unwrap();
    assert!(!o.class_need_train(0));
}

/// Test-only grouped objective: per group, squared error of each row plus a
/// penalty `c/2 * (sum_t s_t - sum_t y_t)^2` coupling all rows in the group.
/// Hessian is dense per group: `I + c * 11^T`.
struct CoupledSquares {
    c: f64,
    label: Vec<f64>,
    how: DiagonalReduction,
}

impl GroupedObjective for CoupledSquares {
    fn name(&self) -> &str {
        "test_coupled_squares"
    }
    fn num_outputs(&self) -> usize {
        1
    }
    fn hessian_mode(&self) -> HessianMode {
        HessianMode::BlockPerGroup
    }
    fn diagonal_reduction(&self) -> DiagonalReduction {
        self.how
    }
    fn init(&mut self, meta: &Metadata, _groups: &GroupIndex) -> Result<()> {
        self.label = meta.label.iter().map(|&x| x as f64).collect();
        Ok(())
    }
    fn gradients(&self, scores: ScoreView<'_>, groups: &GroupIndex, out: &mut GradHessBlock) {
        for g in 0..groups.num_groups() {
            let r = groups.range(g);
            let m = r.len();
            let resid: f64 = r.clone().map(|i| scores.get(0, i) - self.label[i]).sum();
            for (t, i) in r.clone().enumerate() {
                out.grad[i] = (scores.get(0, i) - self.label[i]) + self.c * resid;
                out.hess_diag[i] = 1.0 + self.c;
                for u in 0..m {
                    out.group_blocks[g][t * m + u] = if t == u { 1.0 + self.c } else { self.c };
                }
            }
        }
    }
    fn loss(&self, scores: ScoreView<'_>, groups: &GroupIndex) -> f64 {
        let mut l = 0.0;
        for g in 0..groups.num_groups() {
            let r = groups.range(g);
            let mut resid = 0.0;
            for i in r {
                let d = scores.get(0, i) - self.label[i];
                l += 0.5 * d * d;
                resid += d;
            }
            l += 0.5 * self.c * resid * resid;
        }
        l
    }
    fn to_model_string(&self) -> String {
        "test_coupled_squares".into()
    }
}

#[test]
fn grouped_objective_block_hessian_matches_finite_differences() {
    let groups = GroupIndex::from_sizes(&[3, 2]).unwrap();
    let meta = Metadata { label: vec![1.0, 2.0, 0.0, -1.0, 0.5], weight: None, init_score: None };
    let mut o = CoupledSquares { c: 0.7, label: vec![], how: DiagonalReduction::GershgorinBound };
    o.init(&meta, &groups).unwrap();
    let scores = vec![0.3, -0.2, 1.0, 0.0, 2.0];
    let n = scores.len();
    let mut block = GradHessBlock::new(n, 1, HessianMode::BlockPerGroup, &groups);
    o.gradients(ScoreView { scores: &scores, num_data: n, num_outputs: 1 }, &groups, &mut block);
    let h = 1e-4;
    let loss_at = |s: &[f64]| o.loss(ScoreView { scores: s, num_data: n, num_outputs: 1 }, &groups);
    for i in 0..n {
        let mut p = scores.clone();
        let mut m = scores.clone();
        p[i] += h;
        m[i] -= h;
        let fg = (loss_at(&p) - loss_at(&m)) / (2.0 * h);
        assert!((block.grad[i] - fg).abs() < 1e-6, "grad {i}");
    }
    // cross second derivative inside a group and between groups
    let mixed = |i: usize, j: usize| {
        let mut s = scores.clone();
        let f = |di: f64, dj: f64, s: &mut Vec<f64>| {
            s[i] += di;
            s[j] += dj;
            let v = loss_at(s);
            s[i] -= di;
            s[j] -= dj;
            v
        };
        (f(h, h, &mut s) - f(h, -h, &mut s) - f(-h, h, &mut s) + f(-h, -h, &mut s)) / (4.0 * h * h)
    };
    assert!((mixed(0, 1) - block.group_blocks[0][1]).abs() < 1e-5);
    assert!((mixed(3, 4) - block.group_blocks[1][1]).abs() < 1e-5);
    assert!(mixed(0, 3).abs() < 1e-5, "groups are independent");

    let diag = block.reduce_to_diagonal(DiagonalReduction::GershgorinBound, &groups);
    // row 0 of group 0: |1+c| + 2|c|
    assert!((diag[0] - (1.0 + 0.7 + 2.0 * 0.7)).abs() < 1e-12);
    let drop = block.reduce_to_diagonal(DiagonalReduction::DropOffDiagonal, &groups);
    assert!((drop[0] - 1.7).abs() < 1e-12);
}

#[test]
fn grouped_objective_drives_objective_enum() {
    let groups = GroupIndex::from_sizes(&[2, 2]).unwrap();
    let meta = Metadata { label: vec![1.0, 2.0, 3.0, 4.0], weight: None, init_score: None };
    let mut obj = Objective::Grouped {
        objective: Box::new(CoupledSquares { c: 1.0, label: vec![], how: DiagonalReduction::GershgorinBound }),
        groups,
    };
    obj.init(&meta, 4).unwrap();
    let scores = vec![0.0; 4];
    let mut g = vec![0.0f32; 4];
    let mut h = vec![0.0f32; 4];
    obj.gradients(ScoreView { scores: &scores, num_data: 4, num_outputs: 1 }, &mut g, &mut h);
    assert_eq!(g, vec![-4.0, -5.0, -10.0, -11.0]);
    assert_eq!(h, vec![3.0; 4]);
    assert_eq!(obj.to_model_string(), "test_coupled_squares hessian_reduction:gershgorin_bound");
    assert!(obj.init(&meta, 5).is_err(), "group/row mismatch is rejected");
}

#[test]
fn group_index_validation() {
    assert!(GroupIndex::from_sizes(&[2, 0]).is_err());
    let g = GroupIndex::from_row_ids(&[7, 7, 3, 3, 3]).unwrap();
    assert_eq!(g.num_groups(), 2);
    assert_eq!(g.range(1), 2..5);
    assert!(GroupIndex::from_row_ids(&[1, 2, 1]).is_err());
}
