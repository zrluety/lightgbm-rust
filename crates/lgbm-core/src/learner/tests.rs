use std::sync::Arc;

use super::*;
use crate::dataset::{DatasetFields, DenseMatrix};

fn step_data(n: usize) -> (Vec<f64>, Vec<f32>) {
    // x0 decides the label; x1 is noise-free but uninformative.
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64 / n as f64;
        x.push(a);
        x.push(((i * 37) % 11) as f64);
        y.push(if a < 0.5 { -1.0 } else { 1.0 });
    }
    (x, y)
}

fn dataset(x: &[f64], y: &[f32], n: usize, cfg: &Config) -> Arc<Dataset> {
    let m = DenseMatrix::from_f64_row_major(x, n, 2).unwrap();
    Arc::new(Dataset::from_dense(&m, DatasetFields { label: y, ..Default::default() }, cfg).unwrap())
}

#[test]
fn first_split_separates_step() {
    let n = 200;
    let (x, y) = step_data(n);
    let cfg = Config::from_pairs([("num_leaves", "2"), ("min_data_in_leaf", "5")]).unwrap();
    let ds = dataset(&x, &y, n, &cfg);
    let mut l = SerialTreeLearner::new(ds.clone(), &cfg);
    let grad: Vec<f32> = y.iter().map(|v| -v).collect();
    let hess = vec![1.0f32; n];
    let tree = l.train(&grad, &hess, false).unwrap();
    assert_eq!(tree.num_leaves, 2);
    assert_eq!(tree.split_feature[0], 0);
    assert_eq!(tree.leaf_count, vec![100, 100]);
    assert!((tree.leaf_value[0] + 1.0).abs() < 1e-9);
    assert!((tree.leaf_value[1] - 1.0).abs() < 1e-9);
    // binned and raw routing agree
    for i in 0..n {
        assert_eq!(tree.get_leaf_binned(&ds, i), tree.get_leaf(&x[2 * i..2 * i + 2]));
    }
}

#[test]
fn partition_is_stable() {
    let n = 100;
    let (x, y) = step_data(n);
    let cfg = Config::from_pairs([("num_leaves", "4"), ("min_data_in_leaf", "3")]).unwrap();
    let ds = dataset(&x, &y, n, &cfg);
    let mut l = SerialTreeLearner::new(ds, &cfg);
    let grad: Vec<f32> = (0..n).map(|i| ((i * 7919) % 13) as f32 - 6.0).collect();
    let hess = vec![1.0f32; n];
    let tree = l.train(&grad, &hess, false).unwrap();
    let p = l.partition();
    let mut seen = 0;
    for leaf in 0..tree.num_leaves {
        let idx = p.indices_on_leaf(leaf);
        assert!(idx.windows(2).all(|w| w[0] < w[1]), "leaf {leaf} indices not increasing");
        seen += idx.len();
    }
    assert_eq!(seen, n);
}

#[test]
fn max_depth_and_min_data_respected() {
    let n = 300;
    let (x, y) = step_data(n);
    let cfg = Config::from_pairs([("num_leaves", "31"), ("max_depth", "2"), ("min_data_in_leaf", "40")]).unwrap();
    let ds = dataset(&x, &y, n, &cfg);
    let mut l = SerialTreeLearner::new(ds, &cfg);
    let grad: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
    let hess = vec![1.0f32; n];
    let tree = l.train(&grad, &hess, false).unwrap();
    assert!(tree.num_leaves <= 4);
    assert!(tree.max_depth() <= 2);
    assert!(tree.leaf_count.iter().all(|&c| c >= 40));
}

#[test]
fn constant_gradient_gives_single_leaf() {
    let n = 100;
    let (x, y) = step_data(n);
    let cfg = Config::default();
    let ds = dataset(&x, &y, n, &cfg);
    let mut l = SerialTreeLearner::new(ds, &cfg);
    let tree = l.train(&vec![0.5f32; n], &vec![1.0f32; n], false).unwrap();
    assert_eq!(tree.num_leaves, 1);
}
