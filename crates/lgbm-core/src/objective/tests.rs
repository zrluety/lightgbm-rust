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
    check_row_objective_with(obj, meta, scores, loss, true);
}

/// `check_hess = false` for losses whose upstream Hessian is a deliberate
/// constant rather than the second derivative (L1, Huber, quantile, MAPE).
fn check_row_objective_with(
    obj: &mut dyn RowObjective,
    meta: &Metadata,
    scores: &[f64],
    loss: impl Fn(usize, f64) -> f64,
    check_hess: bool,
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
        if check_hess {
            assert!((h[i] as f64 - fh).abs() <= 1e-3 * (1.0 + fh.abs()), "hess row {i}: {} vs {fh}", h[i]);
        }
    }
}

fn regression_obj(pairs: &[(&str, &str)]) -> regression::Regression {
    regression::Regression::new(&Config::from_pairs(pairs.iter().copied()).unwrap()).unwrap()
}

#[test]
fn l2_matches_finite_differences() {
    let label = vec![0.5f32, -1.0, 2.0, 3.25];
    let weight = vec![1.0f32, 0.5, 2.0, 1.5];
    let scores = [0.1, 0.2, -0.3, 4.0];
    let meta = Metadata { label: label.clone(), weight: None, init_score: None, ..Default::default() };
    let mut o = regression_obj(&[("objective", "regression")]);
    check_row_objective(&mut o, &meta, &scores, |i, s| l2_loss(s, label[i] as f64, 1.0));
    let meta_w = Metadata { label: label.clone(), weight: Some(weight.clone()), init_score: None, ..Default::default() };
    let mut o = regression_obj(&[("objective", "regression")]);
    check_row_objective(&mut o, &meta_w, &scores, |i, s| l2_loss(s, label[i] as f64, weight[i] as f64));
}

#[test]
fn regression_losses_match_finite_differences() {
    let label = vec![0.5f32, 1.0, 2.0, 3.25, 0.75];
    let weight = vec![1.0f32, 0.5, 2.0, 1.5, 0.25];
    // keep |score - label| away from the kinks of L1/Huber/quantile
    let scores = [0.1, 1.7, -0.3, 4.0, 0.2];
    let (alpha, c, rho) = (0.8f64, 1.3f64, 1.4f64);
    let a32 = alpha as f32 as f64;
    type Loss = Box<dyn Fn(f64, f64) -> f64>;
    type Case<'a> = (&'a str, Vec<(&'a str, String)>, Loss, bool);
    let cases: Vec<Case> = vec![
        ("fair", vec![("fair_c", c.to_string())], Box::new(move |s, y| {
            let x = (s - y).abs();
            c * x - c * c * (1.0 + x / c).ln()
        }), true),
        ("poisson", vec![("poisson_max_delta_step", "1e-300".into())], Box::new(|s, y| s.exp() - y * s), true),
        ("gamma", vec![], Box::new(|s, y| y * (-s).exp() + s), true),
        ("tweedie", vec![("tweedie_variance_power", rho.to_string())], Box::new(move |s, y| {
            -y * ((1.0 - rho) * s).exp() / (1.0 - rho) + ((2.0 - rho) * s).exp() / (2.0 - rho)
        }), true),
        ("huber", vec![("alpha", alpha.to_string())], Box::new(move |s, y| {
            let d = s - y;
            if d.abs() <= alpha { 0.5 * d * d } else { alpha * (d.abs() - 0.5 * alpha) }
        }), false),
        ("regression_l1", vec![], Box::new(|s, y| (s - y).abs()), false),
        ("quantile", vec![("alpha", alpha.to_string())], Box::new(move |s, y| {
            let d = y - s;
            if d < 0.0 { (a32 - 1.0) * d } else { a32 * d }
        }), false),
        ("mape", vec![], Box::new(|s, y| (s - y).abs() / y.abs().max(1.0)), false),
    ];
    for (name, extra, loss, check_hess) in cases {
        for weighted in [false, true] {
            let mut pairs = vec![("objective", name.to_string())];
            pairs.extend(extra.iter().map(|(k, v)| (*k, v.clone())));
            let mut o = regression::Regression::new(&Config::from_pairs(pairs).unwrap()).unwrap();
            let meta = Metadata {
                label: label.clone(),
                weight: weighted.then(|| weight.clone()),
                init_score: None,
                ..Default::default()
            };
            let w = |i: usize| if weighted { weight[i] as f64 } else { 1.0 };
            check_row_objective_with(&mut o, &meta, &scores, |i, s| w(i) * loss(s, label[i] as f64), check_hess);
        }
    }
}

#[test]
fn regression_boost_from_score_and_renew() {
    let meta = Metadata { label: vec![1.0, 5.0, 2.0, 9.0, 3.0], weight: None, init_score: None, ..Default::default() };
    let mut l1 = regression_obj(&[("objective", "l1")]);
    l1.init(&meta, 5).unwrap();
    assert_eq!(l1.boost_from_score(0), 3.0);
    assert!(l1.is_renew_tree_output());
    // residuals label - score over rows {0, 1, 3}: 1, 5, 9 -> median 5
    assert_eq!(l1.renew_leaf_output(&[0, 1, 3], &[0.0; 5]), 5.0);
    let mut p = regression_obj(&[("objective", "poisson")]);
    p.init(&meta, 5).unwrap();
    assert_eq!(p.boost_from_score(0), 4.0f64.ln());
    assert!(!p.is_renew_tree_output());
    let mut out = [0.0];
    p.convert_output(&[1.0], &mut out);
    assert_eq!(out[0], 1.0f64.exp());
    let bad = Metadata { label: vec![1.0, -1.0], weight: None, init_score: None, ..Default::default() };
    let e = regression_obj(&[("objective", "gamma")]).init(&bad, 2).unwrap_err();
    assert!(e.to_string().contains("[gamma]: at least one target label is negative"), "{e}");
    let zero = Metadata { label: vec![0.0, 0.0], weight: None, init_score: None, ..Default::default() };
    assert!(regression_obj(&[("objective", "tweedie")]).init(&zero, 2).is_err());
    let cfg = Config::from_pairs([("objective", "quantile"), ("alpha", "1.5")]);
    assert!(cfg.is_err() || regression::Regression::new(&cfg.unwrap()).is_err());
}

#[test]
fn binary_matches_finite_differences() {
    let label = vec![0.0f32, 1.0, 1.0, 0.0, 1.0];
    let weight = vec![1.0f32, 0.5, 2.0, 1.5, 0.25];
    let scores = [0.1, -2.0, 0.7, 3.0, 0.0];
    for sigmoid in [1.0, 0.5, 2.0] {
        let cfg = Config::from_pairs([("objective", "binary"), ("sigmoid", &sigmoid.to_string())]).unwrap();
        let meta = Metadata { label: label.clone(), weight: Some(weight.clone()), init_score: None, ..Default::default() };
        let mut o = binary::BinaryLogloss::new(&cfg).unwrap();
        check_row_objective(&mut o, &meta, &scores, |i, s| {
            logloss(s, label[i] as f64, weight[i] as f64, sigmoid)
        });
    }
}

#[test]
fn binary_boost_from_score_is_logit_of_mean() {
    let cfg = Config::from_pairs([("objective", "binary")]).unwrap();
    let meta = Metadata { label: vec![1.0, 0.0, 0.0, 0.0], weight: None, init_score: None, ..Default::default() };
    let mut o = binary::BinaryLogloss::new(&cfg).unwrap();
    o.init(&meta, 4).unwrap();
    assert!((o.boost_from_score(0) - (0.25f64 / 0.75).ln()).abs() < 1e-15);
}

#[test]
fn binary_single_class_needs_no_training() {
    let cfg = Config::from_pairs([("objective", "binary")]).unwrap();
    let meta = Metadata { label: vec![1.0; 3], weight: None, init_score: None, ..Default::default() };
    let mut o = binary::BinaryLogloss::new(&cfg).unwrap();
    o.init(&meta, 3).unwrap();
    assert!(!o.class_need_train(0));
}

#[test]
fn multiclass_softmax_matches_finite_differences() {
    let label = vec![0.0f32, 2.0, 1.0, 2.0];
    let weight = vec![1.0f32, 0.5, 2.0, 1.5];
    let k = 3;
    let n = label.len();
    // class-major scores
    let scores = vec![0.1, -1.0, 0.5, 2.0, 0.3, 0.0, -0.2, 1.0, -0.4, 0.8, 0.9, -1.5];
    let cfg = Config::from_pairs([("objective", "multiclass"), ("num_class", "3")]).unwrap();
    let factor = 3.0 / 2.0;
    for w in [None, Some(weight.clone())] {
        let meta = Metadata { label: label.clone(), weight: w.clone(), init_score: None, ..Default::default() };
        let mut o = multiclass::MulticlassSoftmax::new(&cfg);
        o.init(&meta, n).unwrap();
        let mut g = vec![0.0f32; n * k];
        let mut h = vec![0.0f32; n * k];
        o.gradients(ScoreView { scores: &scores, num_data: n, num_outputs: k }, &mut g, &mut h);
        for i in 0..n {
            let wi = w.as_ref().map_or(1.0, |w| w[i] as f64);
            let loss = |c: usize, x: f64| {
                let s: Vec<f64> = (0..k).map(|j| if j == c { x } else { scores[j * n + i] }).collect();
                let lse = s.iter().map(|v| v.exp()).sum::<f64>().ln();
                wi * (lse - s[label[i] as usize])
            };
            for c in 0..k {
                let (fg, fh) = fd(|x| loss(c, x), scores[c * n + i]);
                let (ag, ah) = (g[c * n + i] as f64, h[c * n + i] as f64);
                assert!((ag - fg).abs() <= 1e-5 * (1.0 + fg.abs()), "grad {c},{i}: {ag} vs {fg}");
                // upstream scales the diagonal Hessian by K / (K - 1)
                assert!((ah - factor * fh).abs() <= 1e-3 * (1.0 + fh.abs()), "hess {c},{i}: {ah} vs {fh}");
            }
        }
    }
    // init scores are log class priors; a class absent from the labels is not trained
    let meta = Metadata { label: vec![0.0, 0.0, 2.0, 0.0], weight: None, init_score: None, ..Default::default() };
    let mut o = multiclass::MulticlassSoftmax::new(&cfg);
    o.init(&meta, 4).unwrap();
    assert_eq!(o.boost_from_score(0), 0.75f64.ln());
    assert!(!o.class_need_train(1) && o.class_need_train(0));
    let bad = Metadata { label: vec![0.0, 3.0], weight: None, init_score: None, ..Default::default() };
    let err = multiclass::MulticlassSoftmax::new(&cfg).init(&bad, 2).unwrap_err();
    assert!(err.to_string().contains("Label must be in [0, 3), but found 3 in label"));
}

#[test]
fn multiclass_ova_is_per_class_binary() {
    let label = vec![0.0f32, 2.0, 1.0, 2.0];
    let n = label.len();
    let scores = vec![0.1, -1.0, 0.5, 2.0, 0.3, 0.0, -0.2, 1.0, -0.4, 0.8, 0.9, -1.5];
    let cfg = Config::from_pairs([("objective", "multiclassova"), ("num_class", "3")]).unwrap();
    let meta = Metadata { label: label.clone(), weight: None, init_score: None, ..Default::default() };
    let mut o = multiclass::MulticlassOva::new(&cfg).unwrap();
    o.init(&meta, n).unwrap();
    let mut g = vec![0.0f32; 3 * n];
    let mut h = vec![0.0f32; 3 * n];
    o.gradients(ScoreView { scores: &scores, num_data: n, num_outputs: 3 }, &mut g, &mut h);
    for c in 0..3 {
        let y: Vec<f32> = label.iter().map(|&l| (l as usize == c) as i32 as f32).collect();
        let mut b = binary::BinaryLogloss::new(&Config::from_pairs([("objective", "binary")]).unwrap()).unwrap();
        b.init(&Metadata { label: y, weight: None, init_score: None, ..Default::default() }, n).unwrap();
        let (mut bg, mut bh) = (vec![0.0f32; n], vec![0.0f32; n]);
        b.gradients(ScoreView { scores: &scores[c * n..(c + 1) * n], num_data: n, num_outputs: 1 }, &mut bg, &mut bh);
        assert_eq!(&g[c * n..(c + 1) * n], &bg[..]);
        assert_eq!(&h[c * n..(c + 1) * n], &bh[..]);
        assert_eq!(o.boost_from_score(c), b.boost_from_score(0));
    }
    assert_eq!(o.to_model_string(), "multiclassova num_class:3 sigmoid:1");
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
    let meta = Metadata { label: vec![1.0, 2.0, 0.0, -1.0, 0.5], weight: None, init_score: None, ..Default::default() };
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
    let meta = Metadata { label: vec![1.0, 2.0, 3.0, 4.0], weight: None, init_score: None, ..Default::default() };
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
