//! Evaluation metrics.
//!
//! upstream: src/metric/regression_metric.hpp, src/metric/binary_metric.hpp.

use crate::config::Config;
use crate::consts::K_EPSILON;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};
use crate::objective::Objective;
use crate::objective::regression::safe_log;

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
        }
    }

    pub fn higher_better(&self) -> bool {
        matches!(self, MetricKind::Auc)
    }
}

/// Loss parameters read by the regression metrics.
#[derive(Debug, Clone, Copy)]
struct LossParams {
    alpha: f64,
    fair_c: f64,
    rho: f64,
}

/// A metric bound to one dataset's labels and weights.
#[derive(Debug, Clone)]
pub struct Metric {
    pub kind: MetricKind,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    sum_weights: f64,
    params: LossParams,
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
        let params = LossParams { alpha: cfg.alpha, fair_c: cfg.fair_c, rho: cfg.tweedie_variance_power };
        Ok(Self { kind, label: meta.label.clone(), weight: meta.weight.clone(), sum_weights, params })
    }

    /// Evaluate on raw scores (`num_data` entries for single-output models).
    pub fn eval(&self, score: &[f64], objective: Option<&Objective>) -> f64 {
        match self.kind {
            MetricKind::Auc => self.auc(score),
            MetricKind::BinaryLogloss | MetricKind::BinaryError => self.binary(score, objective),
            kind => {
                let p = self.params;
                let s = self.weighted_sum(score, objective, |label, s| regression_loss(kind, p, label, s));
                match kind {
                    MetricKind::Rmse => (s / self.sum_weights).sqrt(),
                    MetricKind::GammaDeviance => s * 2.0,
                    _ => s / self.sum_weights,
                }
            }
        }
    }

    fn binary(&self, score: &[f64], objective: Option<&Objective>) -> f64 {
        match self.kind {
            MetricKind::BinaryLogloss | MetricKind::BinaryError => {
                let kind = self.kind;
                let loss = move |label: f32, prob: f64| -> f64 {
                    if kind == MetricKind::BinaryLogloss {
                        if label <= 0.0 {
                            if 1.0 - prob > K_EPSILON { -(1.0 - prob).ln() } else { -K_EPSILON.ln() }
                        } else if prob > K_EPSILON {
                            -prob.ln()
                        } else {
                            -K_EPSILON.ln()
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
        Metadata { label, weight: None, init_score: None }
    }

    fn metric(kind: MetricKind, md: &Metadata) -> Metric {
        Metric::new(kind, md, &Config::default()).unwrap()
    }

    #[test]
    fn auc_perfect_and_ties() {
        let m = metric(MetricKind::Auc, &meta(vec![0.0, 0.0, 1.0, 1.0]));
        assert_eq!(m.eval(&[0.1, 0.2, 0.8, 0.9], None), 1.0);
        assert_eq!(m.eval(&[0.5, 0.5, 0.5, 0.5], None), 0.5);
    }

    #[test]
    fn l2_and_rmse() {
        let md = meta(vec![1.0, 2.0]);
        let l2 = metric(MetricKind::L2, &md);
        let rmse = metric(MetricKind::Rmse, &md);
        assert_eq!(l2.eval(&[2.0, 4.0], None), 2.5);
        assert_eq!(rmse.eval(&[2.0, 4.0], None), 2.5f64.sqrt());
    }

    #[test]
    fn logloss_clamps() {
        let m = metric(MetricKind::BinaryLogloss, &meta(vec![1.0]));
        assert_eq!(m.eval(&[0.0], None), -K_EPSILON.ln());
    }

    #[test]
    fn regression_metric_values() {
        let md = meta(vec![1.0, 3.0]);
        let s = [2.0, 2.0];
        // alpha = 0.9 by default
        assert!((metric(MetricKind::Quantile, &md).eval(&s, None) - (0.1 * 1.0 + 0.9 * 1.0) / 2.0).abs() < 1e-15);
        // |diff| = 1 > alpha: alpha * (1 - alpha / 2)
        assert!((metric(MetricKind::Huber, &md).eval(&s, None) - 0.9 * (1.0 - 0.45)).abs() < 1e-15);
        assert_eq!(metric(MetricKind::Mape, &md).eval(&s, None), (1.0 + 1.0 / 3.0) / 2.0);
        // gamma deviance sums (no averaging): 2 * sum(y/s - ln(y/s) - 1)
        let gd = metric(MetricKind::GammaDeviance, &md).eval(&[1.0, 3.0], None);
        assert!(gd.abs() < 1e-8);
        assert!(Metric::new(MetricKind::Gamma, &meta(vec![1.0, 0.0]), &Config::default()).is_err());
    }
}
