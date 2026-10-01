use std::sync::Arc;

use lgbm_core::{Config, Dataset, DatasetFields, DenseMatrix, Gbdt, PredictKind};

fn synth(n: usize, p: usize, seed: u64, binary: bool) -> (Vec<f64>, Vec<f32>) {
    let mut s = seed | 1;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut x = vec![0.0; n * p];
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..p {
            let v = next() * 4.0 - 2.0;
            // sprinkle zeros and NaNs to exercise missing handling
            x[i * p + j] = if j == 2 && i % 7 == 0 { f64::NAN } else if j == 3 && i % 3 == 0 { 0.0 } else { v };
        }
        let r = &x[i * p..(i + 1) * p];
        let z = r[0] - 0.5 * r[1] * r[1] + if r[2].is_nan() { 1.0 } else { r[2] } + 0.3 * (next() - 0.5);
        y[i] = if binary { (z > 0.0) as i32 as f32 } else { z as f32 };
    }
    (x, y)
}

fn fit(cfg: &Config, x: &[f64], y: &[f32], n: usize, p: usize) -> Gbdt {
    let mat = DenseMatrix::from_f64_row_major(x, n, p).unwrap();
    let ds = Arc::new(Dataset::from_dense(&mat, DatasetFields { label: y, ..Default::default() }, cfg).unwrap());
    let mut b = Gbdt::new(cfg.clone(), ds, None).unwrap();
    b.train().unwrap();
    b
}

#[test]
fn binary_training_reduces_logloss_and_roundtrips() {
    let (n, p) = (2000, 5);
    let (x, y) = synth(n, p, 42, true);
    let cfg = Config::from_pairs([("objective", "binary"), ("num_iterations", "30"), ("is_provide_training_metric", "true")]).unwrap();
    let b = fit(&cfg, &x, &y, n, p);
    assert_eq!(b.current_iteration(), 30);
    let ev = b.eval_train();
    assert_eq!(ev[0].1, "binary_logloss");
    assert!(ev[0].2 < 0.4, "training logloss {}", ev[0].2);

    let mat = DenseMatrix::from_f64_row_major(&x, n, p).unwrap();
    let p1 = b.predict(&mat, PredictKind::Normal, 0, -1).unwrap();
    let raw = b.predict(&mat, PredictKind::Raw, 0, -1).unwrap();
    for (pr, r) in p1.iter().zip(&raw) {
        assert!((pr - 1.0 / (1.0 + (-r).exp())).abs() < 1e-15);
    }
    // Raw predictions on training rows equal the incrementally updated scores.
    let scores = b.train_scores().unwrap();
    for (a, s) in raw.iter().zip(scores) {
        assert!((a - s).abs() <= 1e-9 * (1.0 + s.abs()), "{a} vs {s}");
    }

    let text = b.save_model_to_string(0, -1, 0).unwrap();
    let c = Gbdt::load_model_from_string(&text).unwrap();
    assert_eq!(c.save_model_to_string(0, -1, 0).unwrap(), text, "save(load(save)) is stable");
    let p2 = c.predict(&mat, PredictKind::Normal, 0, -1).unwrap();
    assert_eq!(p1, p2, "loaded model predicts identically");

    // iteration window and leaf indices
    let first10 = b.predict(&mat, PredictKind::Raw, 0, 10).unwrap();
    let rest = b.predict(&mat, PredictKind::Raw, 10, -1).unwrap();
    for i in 0..n {
        assert!((first10[i] + rest[i] - raw[i]).abs() < 1e-12);
    }
    let leaves = b.predict(&mat, PredictKind::LeafIndex, 0, -1).unwrap();
    assert_eq!(leaves.len(), n * 30);
}

#[test]
fn regression_with_validation_and_early_stopping() {
    let (n, p) = (1500, 4);
    let (x, y) = synth(n, p, 7, false);
    let (xv, yv) = synth(500, p, 99, false);
    let cfg = Config::from_pairs([
        ("objective", "regression"),
        ("num_iterations", "500"),
        ("learning_rate", "0.3"),
        ("early_stopping_round", "5"),
        ("metric", "l2,l1"),
    ])
    .unwrap();
    let mat = DenseMatrix::from_f64_row_major(&x, n, p).unwrap();
    let ds = Arc::new(Dataset::from_dense(&mat, DatasetFields { label: &y, ..Default::default() }, &cfg).unwrap());
    let vmat = DenseMatrix::from_f64_row_major(&xv, 500, p).unwrap();
    let vds = Arc::new(
        Dataset::from_dense_with_reference(&vmat, DatasetFields { label: &yv, ..Default::default() }, &ds, 0).unwrap(),
    );
    let mut b = Gbdt::new(cfg, ds, None).unwrap();
    b.add_valid(vds, "valid_0").unwrap();
    let best = b.train().unwrap();
    assert!(best > 0 && best < 500, "early stopping triggered at {best}");
    assert_eq!(b.current_iteration(), best);
    let ev = b.eval_valid();
    assert_eq!(ev.len(), 2);
    assert_eq!((ev[0].1.as_str(), ev[1].1.as_str()), ("l2", "l1"));
    // validation scores maintained incrementally match fresh prediction
    let pred = b.predict(&vmat, PredictKind::Raw, 0, -1).unwrap();
    for (a, s) in pred.iter().zip(b.valid_scores(0).unwrap()) {
        assert!((a - s).abs() <= 1e-9 * (1.0 + s.abs()));
    }
}

#[test]
fn single_class_binary_gives_constant_model() {
    let (n, p) = (100, 3);
    let (x, _) = synth(n, p, 3, true);
    let y = vec![1.0f32; n];
    let cfg = Config::from_pairs([("objective", "binary"), ("num_iterations", "5")]).unwrap();
    let b = fit(&cfg, &x, &y, n, p);
    assert_eq!(b.num_trees(), 1);
    assert_eq!(b.trees()[0].num_leaves, 1);
}

#[test]
fn custom_gradients_path() {
    let (n, p) = (500, 3);
    let (x, y) = synth(n, p, 11, false);
    let cfg = Config::from_pairs([("objective", "none"), ("num_iterations", "3")]).unwrap();
    let mat = DenseMatrix::from_f64_row_major(&x, n, p).unwrap();
    let ds = Arc::new(Dataset::from_dense(&mat, DatasetFields { label: &y, ..Default::default() }, &cfg).unwrap());
    let mut b = Gbdt::new(cfg, ds, None).unwrap();
    assert!(b.train_one_iter(None).is_err(), "no objective requires custom gradients");
    for _ in 0..3 {
        let s = b.train_scores().unwrap().to_vec();
        let g: Vec<f32> = s.iter().zip(&y).map(|(s, y)| (*s - *y as f64) as f32).collect();
        let h = vec![1.0f32; n];
        b.train_one_iter(Some((&g, &h))).unwrap();
    }
    assert_eq!(b.current_iteration(), 3);
    assert!(!b.save_model_to_string(0, -1, 0).unwrap().contains("objective="));
}

#[test]
fn histogram_layouts_and_thread_counts_agree() {
    // large enough for the parallel row-wise blocks and partition paths
    let (n, p) = (40_000, 6);
    let (x, y) = synth(n, p, 5, true);
    let trees = |pairs: &[(&str, &str)]| {
        let mut all = vec![("objective", "binary"), ("num_iterations", "8"), ("num_leaves", "63")];
        all.extend_from_slice(pairs);
        fit(&Config::from_pairs(all).unwrap(), &x, &y, n, p).trees().to_vec()
    };
    let row1 = trees(&[("num_threads", "1"), ("force_row_wise", "true")]);
    let col1 = trees(&[("num_threads", "1"), ("force_col_wise", "true")]);
    let col4 = trees(&[("num_threads", "4"), ("force_col_wise", "true")]);
    assert!(row1 == col1, "row-wise and col-wise histograms agree at 1 thread");
    assert!(col1 == col4, "col-wise training does not depend on the thread count");
    let row4 = trees(&[("num_threads", "4"), ("force_row_wise", "true")]);
    assert!(row4 == trees(&[("num_threads", "4"), ("force_row_wise", "true")]), "row-wise is reproducible");
    let both = Config::from_pairs([("force_row_wise", "true"), ("force_col_wise", "true")]).unwrap();
    let mat = DenseMatrix::from_f64_row_major(&x, n, p).unwrap();
    let ds = Arc::new(Dataset::from_dense(&mat, DatasetFields { label: &y, ..Default::default() }, &both).unwrap());
    assert!(Gbdt::new(both, ds, None).is_err());
}

#[test]
fn block_prediction_matches_per_row_traversal() {
    let (n, p) = (1003, 5);
    let (x, y) = synth(n, p, 21, true);
    let cfg = Config::from_pairs([("objective", "binary"), ("num_iterations", "20"), ("num_leaves", "40")]).unwrap();
    let b = fit(&cfg, &x, &y, n, p);
    let mat = DenseMatrix::from_f64_row_major(&x, n, p).unwrap();
    let raw = b.predict(&mat, PredictKind::Raw, 0, -1).unwrap();
    let leaves = b.predict(&mat, PredictKind::LeafIndex, 0, -1).unwrap();
    let ntrees = b.trees().len();
    for i in 0..n {
        let row: Vec<f64> = x[i * p..(i + 1) * p]
            .iter()
            .map(|&v| if !v.is_nan() && v.abs() <= 1e-35_f32 as f64 { 0.0 } else { v })
            .collect();
        let mut s = 0.0;
        for (j, t) in b.trees().iter().enumerate() {
            s += t.predict(&row);
            assert_eq!(leaves[i * ntrees + j], t.get_leaf(&row) as f64);
        }
        assert_eq!(raw[i], s, "row {i}");
    }
}

#[test]
fn unsupported_parameters_are_rejected() {
    for (k, v) in [("cegb_tradeoff", "0.5"), ("objective", "multiclass"), ("boosting", "dart"), ("linear_tree", "true")] {
        let e = Config::from_pairs([(k, v)]);
        assert!(e.is_err(), "{k}={v} should be rejected");
    }
}
