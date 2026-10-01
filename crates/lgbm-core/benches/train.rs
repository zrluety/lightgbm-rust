//! Training/prediction throughput on synthetic dense data.
//!
//! Run with `cargo bench -p lgbm-core --bench train`. Environment variables:
//! `LGBM_BENCH_ROWS` (default 1_000_000), `LGBM_BENCH_COLS` (28),
//! `LGBM_BENCH_ITERS` (50), `LGBM_BENCH_THREADS` (comma list, default "1,0";
//! 0 = all cores). Prints one JSON line per configuration.

use std::sync::Arc;
use std::time::Instant;

use lgbm_core::{Config, Dataset, DatasetFields, DenseMatrix, Gbdt, PredictKind};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Deterministic xorshift data: label depends nonlinearly on a few columns.
fn make_data(n: usize, p: usize) -> (Vec<f64>, Vec<f32>) {
    let mut s: u64 = 0x9E3779B97F4A7C15;
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
            x[i * p + j] = next() * 2.0 - 1.0;
        }
        let r = &x[i * p..(i + 1) * p];
        let z = 1.5 * r[0] - r[1] * r[2] + (3.0 * r[3]).sin() + 0.5 * next();
        y[i] = (z > 0.0) as i32 as f32;
    }
    (x, y)
}

fn main() {
    let n = env_usize("LGBM_BENCH_ROWS", 1_000_000);
    let p = env_usize("LGBM_BENCH_COLS", 28);
    let iters = env_usize("LGBM_BENCH_ITERS", 50);
    let threads: Vec<i32> = std::env::var("LGBM_BENCH_THREADS")
        .unwrap_or_else(|_| "1,0".into())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let (x, y) = make_data(n, p);
    let mat = DenseMatrix::from_f64_row_major(&x, n, p).unwrap();
    for t in threads {
        let it = iters.to_string();
        let th = t.to_string();
        let cfg = Config::from_pairs([
            ("objective", "binary"),
            ("num_iterations", it.as_str()),
            ("num_threads", th.as_str()),
            ("verbosity", "-1"),
        ])
        .unwrap();
        let pool = (t > 0).then(|| rayon::ThreadPoolBuilder::new().num_threads(t as usize).build().unwrap());
        let t0 = Instant::now();
        let build = || Dataset::from_dense(&mat, DatasetFields { label: &y, ..Default::default() }, &cfg).unwrap();
        let ds = Arc::new(match &pool {
            Some(pl) => pl.install(build),
            None => build(),
        });
        let t_construct = t0.elapsed().as_secs_f64();
        let mut b = Gbdt::new(cfg.clone(), ds, None).unwrap();
        let t1 = Instant::now();
        for _ in 0..iters {
            if b.train_one_iter(None).unwrap() {
                break;
            }
        }
        let t_train = t1.elapsed().as_secs_f64();
        let t2 = Instant::now();
        let pred = b.predict(&mat, PredictKind::Normal, 0, -1).unwrap();
        let t_pred = t2.elapsed().as_secs_f64();
        std::hint::black_box(pred);
        println!(
            "{{\"rows\":{n},\"cols\":{p},\"iterations\":{},\"threads\":{t},\"construct_s\":{t_construct:.3},\"train_s\":{t_train:.3},\"predict_s\":{t_pred:.3}}}",
            b.current_iteration()
        );
    }
}
