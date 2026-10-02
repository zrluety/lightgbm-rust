//! Evaluation metrics.
//!
//! upstream: src/metric/regression_metric.hpp, src/metric/binary_metric.hpp,
//! src/metric/multiclass_metric.hpp, src/metric/rank_metric.hpp,
//! src/metric/map_metric.hpp.

use crate::config::Config;
use crate::consts::{K_EPSILON, neg_log_epsilon};
use crate::dataset::Metadata;
use crate::dcg::{self, DcgCalculator};
use crate::error::{LgbmError, Result};
use crate::objective::Objective;
use crate::objective::regression::safe_log;
use crate::threading::resolve_num_threads;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    L2,
    Rmse,
    L1,
    Quantile,
    Huber,
    Fair,
    Poisson,
    Mape,
    Gamma,
    GammaDeviance,
    Tweedie,
    BinaryLogloss,
    BinaryError,
    Auc,
    MultiLogloss,
    MultiError,
    Ndcg,
    Map,
}

impl MetricKind {
    pub fn from_name(name: &str) -> Result<Self> {
        Ok(match name {
            "l2" => MetricKind::L2,
            "rmse" => MetricKind::Rmse,
            "l1" => MetricKind::L1,
            "quantile" => MetricKind::Quantile,
            "huber" => MetricKind::Huber,
            "fair" => MetricKind::Fair,
            "poisson" => MetricKind::Poisson,
            "mape" => MetricKind::Mape,
            "gamma" => MetricKind::Gamma,
            "gamma_deviance" => MetricKind::GammaDeviance,
            "tweedie" => MetricKind::Tweedie,
            "binary_logloss" => MetricKind::BinaryLogloss,
            "binary_error" => MetricKind::BinaryError,
            "auc" => MetricKind::Auc,
            "multi_logloss" => MetricKind::MultiLogloss,
            "multi_error" => MetricKind::MultiError,
            "ndcg" => MetricKind::Ndcg,
            "map" => MetricKind::Map,
            other => return Err(LgbmError::Unsupported(format!("metric={other}"))),
        })
    }

    pub fn name(&self) -> &'static str {
        match self {
            MetricKind::L2 => "l2",
            MetricKind::Rmse => "rmse",
            MetricKind::L1 => "l1",
            MetricKind::Quantile => "quantile",
            MetricKind::Huber => "huber",
            MetricKind::Fair => "fair",
            MetricKind::Poisson => "poisson",
            MetricKind::Mape => "mape",
            MetricKind::Gamma => "gamma",
            MetricKind::GammaDeviance => "gamma_deviance",
            MetricKind::Tweedie => "tweedie",
            MetricKind::BinaryLogloss => "binary_logloss",
            MetricKind::BinaryError => "binary_error",
            MetricKind::Auc => "auc",
            MetricKind::MultiLogloss => "multi_logloss",
            MetricKind::MultiError => "multi_error",
            MetricKind::Ndcg => "ndcg",
            MetricKind::Map => "map",
        }
    }

    pub fn higher_better(&self) -> bool {
        matches!(self, MetricKind::Auc | MetricKind::Ndcg | MetricKind::Map)
    }
}

/// Query data of the NDCG / MAP metrics.
#[derive(Debug, Clone)]
struct RankEval {
    eval_at: Vec<i32>,
    boundaries: Vec<i32>,
    query_weights: Option<Vec<f32>>,
    sum_query_weights: f64,
    /// NDCG: inverse max DCG per query and `eval_at` (-1 when the query has
    /// no relevant document).
    inverse_max_dcgs: Vec<Vec<f64>>,
    dcg: Option<DcgCalculator>,
    /// MAP: rows with label > 0.5 per query.
    npos_per_query: Vec<i32>,
    num_threads: usize,
}

/// upstream: libgomp's `schedule(static)` partition of `n` iterations.
fn omp_static_chunks(n: usize, threads: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    let (q, r) = (n / threads, n % threads);
    (0..threads).map(move |t| {
        let start = t * q + t.min(r);
        start..start + q + (t < r) as usize
    })
}

/// Loss parameters read by the regression metrics.
#[derive(Debug, Clone, Copy)]
struct LossParams {
    alpha: f64,
    fair_c: f64,
    rho: f64,
    multi_error_top_k: i32,
}

/// A metric bound to one dataset's labels and weights.
#[derive(Debug, Clone)]
pub struct Metric {
    pub kind: MetricKind,
    names: Vec<String>,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    sum_weights: f64,
    params: LossParams,
    rank: Option<RankEval>,
}

/// upstream `MultiErrorMetric::LossOnPoint` / `MultiSoftmaxLoglossMetric::LossOnPoint`.
fn multiclass_loss(kind: MetricKind, p: LossParams, label: f32, rec: &[f64]) -> f64 {
    let k = label as usize;
    match kind {
        MetricKind::MultiError => {
            let mut num_larger = 0;
            for &s in rec {
                if s >= rec[k] {
                    num_larger += 1;
                }
                if num_larger > p.multi_error_top_k {
                    return 1.0;
                }
            }
            0.0
        }
        MetricKind::MultiLogloss => {
            if rec[k] > K_EPSILON { -rec[k].ln() } else { neg_log_epsilon() }
        }
        _ => unreachable!("not a multiclass metric"),
    }
}

/// upstream `MapMetric::CalMapAtK`.
fn map_at_k(ks: &[i32], npos: i32, label: &[f32], score: &[f64], out: &mut [f64]) {
    let sorted = dcg::sort_by_score_desc(score);
    let mut num_hit = 0i32;
    let mut sum_ap = 0.0f64;
    let mut left = 0usize;
    for (o, &k) in out.iter_mut().zip(ks) {
        let k = (k as usize).min(label.len());
        for (j, &idx) in sorted.iter().enumerate().take(k).skip(left) {
            if label[idx] > 0.5 {
                num_hit += 1;
                // (j + 1.0f) is evaluated in float
                sum_ap += num_hit as f64 / (j as f32 + 1.0f32) as f64;
            }
        }
        *o = if npos > 0 { sum_ap / npos.min(k as i32) as f64 } else { 1.0 };
        left = k;
    }
}

/// upstream `PointWiseLossCalculator::LossOnPoint` of the regression metrics.
fn regression_loss(kind: MetricKind, p: LossParams, label: f32, score: f64) -> f64 {
    match kind {
        MetricKind::L2 | MetricKind::Rmse => (score - label as f64) * (score - label as f64),
        MetricKind::L1 => (score - label as f64).abs(),
        MetricKind::Quantile => {
            let delta = label as f64 - score;
            if delta < 0.0 { (p.alpha - 1.0) * delta } else { p.alpha * delta }
        }
        MetricKind::Huber => {
            let diff = score - label as f64;
            if diff.abs() <= p.alpha { 0.5 * diff * diff } else { p.alpha * (diff.abs() - 0.5 * p.alpha) }
        }
        MetricKind::Fair => {
            let x = (score - label as f64).abs();
            let c = p.fair_c;
            c * x - c * c * (x / c).ln_1p()
        }
        MetricKind::Poisson => {
            let eps = 1e-10f32 as f64;
            let score = if score < eps { eps } else { score };
            score - label as f64 * score.ln()
        }
        MetricKind::Mape => (label as f64 - score).abs() / 1.0f32.max(label.abs()) as f64,
        MetricKind::Gamma => {
            let psi = 1.0f64;
            let theta = -1.0 / score;
            let a = psi;
            let b = -safe_log(-theta);
            // SafeLog(label) is evaluated in float (label_t) upstream
            let log_label = if label > 0.0 { label.ln() } else { f32::NEG_INFINITY };
            let c = 1. / psi * safe_log(label as f64 / psi) - log_label as f64 - 0.0;
            -((label as f64 * theta - b) / a + c)
        }
        MetricKind::GammaDeviance => {
            let epsilon = 1.0e-9;
            let tmp = label as f64 / (score + epsilon);
            tmp - safe_log(tmp) - 1.0
        }
        MetricKind::Tweedie => {
            let rho = p.rho;
            let eps = 1e-10f32 as f64;
            let score = if score < eps { eps } else { score };
            let a = label as f64 * ((1.0 - rho) * score.ln()).exp() / (1.0 - rho);
            let b = ((2.0 - rho) * score.ln()).exp() / (2.0 - rho);
            -a + b
        }
        _ => unreachable!("not a regression metric"),
    }
}

impl Metric {
    pub fn new(kind: MetricKind, meta: &Metadata, cfg: &Config) -> Result<Self> {
        if matches!(kind, MetricKind::Gamma | MetricKind::GammaDeviance) {
            // upstream: CheckLabel -> CHECK_GT(label, 0)
            if meta.label.iter().any(|&l| !(l > 0.0)) {
                return Err(LgbmError::InvalidData("Check failed: (label) > (0)".into()));
            }
        }
        let sum_weights = match &meta.weight {
            None => meta.label.len() as f64,
            Some(w) => {
                let mut s = 0.0f64;
                for x in w {
                    s += *x as f64;
                }
                s
            }
        };
        let params = LossParams {
            alpha: cfg.alpha,
            fair_c: cfg.fair_c,
            rho: cfg.tweedie_variance_power,
            multi_error_top_k: cfg.multi_error_top_k,
        };
        let (names, rank) = match kind {
            MetricKind::Ndcg | MetricKind::Map => {
                let (names, rank) = Self::rank_init(kind, meta, cfg)?;
                (names, Some(rank))
            }
            MetricKind::MultiError if cfg.multi_error_top_k != 1 => {
                (vec![format!("multi_error@{}", cfg.multi_error_top_k)], None)
            }
            _ => (vec![kind.name().to_string()], None),
        };
        Ok(Self { kind, names, label: meta.label.clone(), weight: meta.weight.clone(), sum_weights, params, rank })
    }

    /// upstream `NDCGMetric` / `MapMetric` constructor and `Init`.
    fn rank_init(kind: MetricKind, meta: &Metadata, cfg: &Config) -> Result<(Vec<String>, RankEval)> {
        let eval_at = dcg::default_eval_at(&cfg.eval_at)?;
        let prefix = if kind == MetricKind::Ndcg { "ndcg@" } else { "map@" };
        let names = eval_at.iter().map(|k| format!("{prefix}{k}")).collect();
        let mut dcg_calc = None;
        if kind == MetricKind::Ndcg {
            let c = DcgCalculator::new(dcg::default_label_gain(&cfg.label_gain));
            dcg::check_metadata(meta)?;
            c.check_label(&meta.label)?;
            dcg_calc = Some(c);
        }
        let boundaries = meta.query_boundaries.clone().ok_or_else(|| {
            LgbmError::InvalidData(
                if kind == MetricKind::Ndcg {
                    "The NDCG metric requires query information"
                } else {
                    "For MAP metric, there should be query information"
                }
                .into(),
            )
        })?;
        let nq = boundaries.len() - 1;
        let query_weights = meta.query_weights.clone();
        let sum_query_weights = match &query_weights {
            None => nq as f64,
            Some(w) => {
                let mut s = 0.0f64;
                for &x in w.iter().take(nq) {
                    s += x as f64;
                }
                s
            }
        };
        let rows = |q: usize| &meta.label[boundaries[q] as usize..boundaries[q + 1] as usize];
        let mut inverse_max_dcgs = Vec::new();
        let mut npos_per_query = Vec::new();
        match &dcg_calc {
            Some(c) => {
                inverse_max_dcgs = (0..nq)
                    .map(|q| {
                        let mut v = vec![0.0f64; eval_at.len()];
                        c.max_dcg(&eval_at, rows(q), &mut v);
                        for x in v.iter_mut() {
                            *x = if *x > 0.0 { 1.0 / *x } else { -1.0 };
                        }
                        v
                    })
                    .collect();
            }
            None => {
                npos_per_query = (0..nq).map(|q| rows(q).iter().filter(|&&l| l > 0.5).count() as i32).collect();
            }
        }
        let rank = RankEval {
            eval_at,
            boundaries,
            query_weights,
            sum_query_weights,
            inverse_max_dcgs,
            dcg: dcg_calc,
            npos_per_query,
            num_threads: resolve_num_threads(cfg.num_threads),
        };
        Ok((names, rank))
    }

    /// Names reported with the values (e.g. `multi_error@2`, `ndcg@1`...).
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Evaluate on raw scores (class-major, `num_data * num_outputs` entries);
    /// one value per [`names`](Self::names) entry.
    pub fn eval(&self, score: &[f64], objective: Option<&Objective>) -> Vec<f64> {
        let v = match self.kind {
            MetricKind::Ndcg | MetricKind::Map => return self.rank_eval(score),
            MetricKind::Auc => self.auc(score),
            MetricKind::BinaryLogloss | MetricKind::BinaryError => self.binary(score, objective),
            MetricKind::MultiLogloss | MetricKind::MultiError => self.multiclass(score, objective),
            kind => {
                let p = self.params;
                let s = self.weighted_sum(score, objective, |label, s| regression_loss(kind, p, label, s));
                match kind {
                    MetricKind::Rmse => (s / self.sum_weights).sqrt(),
                    MetricKind::GammaDeviance => s * 2.0,
                    _ => s / self.sum_weights,
                }
            }
        };
        vec![v]
    }

    /// upstream `NDCGMetric::Eval` / `MapMetric::Eval`. NDCG reproduces the
    /// per-thread partial sums of OpenMP `schedule(static)`; MAP uses
    /// `schedule(guided)`, whose partition is only reproducible with one
    /// thread, so it is summed in query order.
    fn rank_eval(&self, score: &[f64]) -> Vec<f64> {
        let r = self.rank.as_ref().expect("ranking metric state");
        let k = r.eval_at.len();
        let nq = r.boundaries.len() - 1;
        let threads = if self.kind == MetricKind::Ndcg { r.num_threads } else { 1 };
        let mut buffers = vec![vec![0.0f64; k]; threads];
        let mut tmp = vec![0.0f64; k];
        for (buf, range) in buffers.iter_mut().zip(omp_static_chunks(nq, threads)) {
            for q in range {
                let (s, e) = (r.boundaries[q] as usize, r.boundaries[q + 1] as usize);
                // stale query weights (see `Metadata::query_weights`) may be short
                let qw = r.query_weights.as_ref().map(|w| w.get(q).map_or(0.0, |&x| x as f64));
                match &r.dcg {
                    Some(c) => {
                        if r.inverse_max_dcgs[q][0] <= 0.0 {
                            for b in buf.iter_mut() {
                                *b += 1.0;
                            }
                            continue;
                        }
                        c.dcg(&r.eval_at, &self.label[s..e], &score[s..e], &mut tmp);
                        for j in 0..k {
                            buf[j] += match qw {
                                None => tmp[j] * r.inverse_max_dcgs[q][j],
                                Some(w) => tmp[j] * r.inverse_max_dcgs[q][j] * w,
                            };
                        }
                    }
                    None => {
                        map_at_k(&r.eval_at, r.npos_per_query[q], &self.label[s..e], &score[s..e], &mut tmp);
                        for j in 0..k {
                            buf[j] += match qw {
                                None => tmp[j],
                                Some(w) => tmp[j] * w,
                            };
                        }
                    }
                }
            }
        }
        (0..k)
            .map(|j| {
                let mut v = 0.0f64;
                for b in &buffers {
                    v += b[j];
                }
                v / r.sum_query_weights
            })
            .collect()
    }

    fn binary(&self, score: &[f64], objective: Option<&Objective>) -> f64 {
        match self.kind {
            MetricKind::BinaryLogloss | MetricKind::BinaryError => {
                let kind = self.kind;
                let loss = move |label: f32, prob: f64| -> f64 {
                    if kind == MetricKind::BinaryLogloss {
                        if label <= 0.0 {
                            if 1.0 - prob > K_EPSILON { -(1.0 - prob).ln() } else { neg_log_epsilon() }
                        } else if prob > K_EPSILON {
                            -prob.ln()
                        } else {
                            neg_log_epsilon()
                        }
                    } else if prob <= 0.5 {
                        (label > 0.0) as i32 as f64
                    } else {
                        (label <= 0.0) as i32 as f64
                    }
                };
                self.weighted_sum(score, objective, loss) / self.sum_weights
            }
            _ => unreachable!("not a binary metric"),
        }
    }

    /// upstream `MulticlassMetric::Eval`: the objective's transform when there
    /// is one, raw scores otherwise.
    fn multiclass(&self, score: &[f64], objective: Option<&Objective>) -> f64 {
        let n = self.label.len();
        let k = score.len().checked_div(n).unwrap_or(0);
        let mut raw = vec![0.0f64; k];
        let mut rec = vec![0.0f64; k];
        let mut sum = 0.0f64;
        for i in 0..n {
            for c in 0..k {
                raw[c] = score[c * n + i];
            }
            let r = match objective {
                Some(o) => {
                    o.convert_output(&raw, &mut rec);
                    &rec
                }
                None => &raw,
            };
            let l = multiclass_loss(self.kind, self.params, self.label[i], r);
            sum += match &self.weight {
                Some(w) => l * w[i] as f64,
                None => l,
            };
        }
        sum / self.sum_weights
    }

    fn weighted_sum(&self, score: &[f64], objective: Option<&Objective>, loss: impl Fn(f32, f64) -> f64) -> f64 {
        let mut sum = 0.0f64;
        let mut t = [0.0f64];
        for i in 0..self.label.len() {
            let s = match objective {
                Some(o) => {
                    o.convert_output(&score[i..i + 1], &mut t);
                    t[0]
                }
                None => score[i],
            };
            let l = loss(self.label[i], s);
            sum += match &self.weight {
                Some(w) => l * w[i] as f64,
                None => l,
            };
        }
        sum
    }

    fn auc(&self, score: &[f64]) -> f64 {
        let n = self.label.len();
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cur_pos = 0.0f64;
        let mut sum_pos = 0.0f64;
        let mut accum = 0.0f64;
        let mut cur_neg = 0.0f64;
        let mut threshold = score[idx[0]];
        for &i in &idx {
            let lbl = self.label[i];
            let s = score[i];
            if s != threshold {
                threshold = s;
                accum += cur_neg * (cur_pos * 0.5 + sum_pos);
                sum_pos += cur_pos;
                cur_neg = 0.0;
                cur_pos = 0.0;
            }
            let w = self.weight.as_ref().map_or(1.0, |w| w[i] as f64);
            cur_neg += (lbl <= 0.0) as i32 as f64 * w;
            cur_pos += (lbl > 0.0) as i32 as f64 * w;
        }
        accum += cur_neg * (cur_pos * 0.5 + sum_pos);
        sum_pos += cur_pos;
        if sum_pos > 0.0 && sum_pos != self.sum_weights {
            accum / (sum_pos * (self.sum_weights - sum_pos))
        } else {
            1.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(label: Vec<f32>) -> Metadata {
        Metadata { label, weight: None, init_score: None, ..Default::default() }
    }

    fn metric(kind: MetricKind, md: &Metadata) -> Metric {
        Metric::new(kind, md, &Config::default()).unwrap()
    }

    #[test]
    fn auc_perfect_and_ties() {
        let m = metric(MetricKind::Auc, &meta(vec![0.0, 0.0, 1.0, 1.0]));
        assert_eq!(m.eval(&[0.1, 0.2, 0.8, 0.9], None)[0], 1.0);
        assert_eq!(m.eval(&[0.5, 0.5, 0.5, 0.5], None)[0], 0.5);
    }

    #[test]
    fn l2_and_rmse() {
        let md = meta(vec![1.0, 2.0]);
        let l2 = metric(MetricKind::L2, &md);
        let rmse = metric(MetricKind::Rmse, &md);
        assert_eq!(l2.eval(&[2.0, 4.0], None)[0], 2.5);
        assert_eq!(rmse.eval(&[2.0, 4.0], None)[0], 2.5f64.sqrt());
    }

    #[test]
    fn logloss_clamps() {
        let m = metric(MetricKind::BinaryLogloss, &meta(vec![1.0]));
        assert_eq!(m.eval(&[0.0], None)[0], -(1e-15_f32.ln()) as f64);
    }

    #[test]
    fn regression_metric_values() {
        let md = meta(vec![1.0, 3.0]);
        let s = [2.0, 2.0];
        // alpha = 0.9 by default
        assert!((metric(MetricKind::Quantile, &md).eval(&s, None)[0] - (0.1 * 1.0 + 0.9 * 1.0) / 2.0).abs() < 1e-15);
        // |diff| = 1 > alpha: alpha * (1 - alpha / 2)
        assert!((metric(MetricKind::Huber, &md).eval(&s, None)[0] - 0.9 * (1.0 - 0.45)).abs() < 1e-15);
        assert_eq!(metric(MetricKind::Mape, &md).eval(&s, None)[0], (1.0 + 1.0 / 3.0) / 2.0);
        // gamma deviance sums (no averaging): 2 * sum(y/s - ln(y/s) - 1)
        let gd = metric(MetricKind::GammaDeviance, &md).eval(&[1.0, 3.0], None)[0];
        assert!(gd.abs() < 1e-8);
        assert!(Metric::new(MetricKind::Gamma, &meta(vec![1.0, 0.0]), &Config::default()).is_err());
    }

    fn grouped(label: Vec<f32>, counts: &[i32]) -> Metadata {
        let n = label.len();
        let mut m = meta(label);
        m.set_query(n, Some(counts)).unwrap();
        m
    }

    #[test]
    fn ndcg_and_map_values() {
        let cfg = Config::from_pairs([("eval_at", "2,1"), ("num_threads", "3")]).unwrap();
        // query 0 ranks the relevant row second; query 1 has no relevant rows
        let md = grouped(vec![0.0, 1.0, 0.0, 0.0], &[2, 2]);
        let ndcg = Metric::new(MetricKind::Ndcg, &md, &cfg).unwrap();
        assert_eq!(ndcg.names(), &["ndcg@1".to_string(), "ndcg@2".to_string()]);
        let s = [1.0, 0.0, 0.0, 0.0];
        let v = ndcg.eval(&s, None);
        assert_eq!(v, vec![(0.0 + 1.0) / 2.0, (1.0 / 3f64.log2() + 1.0) / 2.0]);
        let map = Metric::new(MetricKind::Map, &md, &cfg).unwrap();
        assert_eq!(map.names(), &["map@1".to_string(), "map@2".to_string()]);
        assert_eq!(map.eval(&s, None), vec![(0.0 + 1.0) / 2.0, (0.5 + 1.0) / 2.0]);
        let e = Metric::new(MetricKind::Ndcg, &meta(vec![0.0]), &cfg).unwrap_err();
        assert!(e.to_string().contains("The NDCG metric requires query information"));
        let e = Metric::new(MetricKind::Map, &meta(vec![0.0]), &cfg).unwrap_err();
        assert!(e.to_string().contains("For MAP metric, there should be query information"));
    }

    #[test]
    fn omp_static_partition() {
        let parts: Vec<_> = omp_static_chunks(7, 3).collect();
        assert_eq!(parts, vec![0..3, 3..5, 5..7]);
        let parts: Vec<_> = omp_static_chunks(2, 4).collect();
        assert_eq!(parts, vec![0..1, 1..2, 2..2, 2..2]);
    }
}
