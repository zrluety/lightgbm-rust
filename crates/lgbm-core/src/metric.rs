//! Evaluation metrics.
//!
//! upstream: src/metric/regression_metric.hpp, src/metric/binary_metric.hpp.

use crate::consts::K_EPSILON;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};
use crate::objective::Objective;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricKind {
    L2,
    Rmse,
    L1,
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
            MetricKind::BinaryLogloss => "binary_logloss",
            MetricKind::BinaryError => "binary_error",
            MetricKind::Auc => "auc",
        }
    }

    pub fn higher_better(&self) -> bool {
        matches!(self, MetricKind::Auc)
    }
}

/// A metric bound to one dataset's labels and weights.
#[derive(Debug, Clone)]
pub struct Metric {
    pub kind: MetricKind,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    sum_weights: f64,
}

impl Metric {
    pub fn new(kind: MetricKind, meta: &Metadata) -> Self {
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
        Self { kind, label: meta.label.clone(), weight: meta.weight.clone(), sum_weights }
    }

    /// Evaluate on raw scores (`num_data` entries for single-output models).
    pub fn eval(&self, score: &[f64], objective: Option<&Objective>) -> f64 {
        match self.kind {
            MetricKind::Auc => self.auc(score),
            MetricKind::L2 | MetricKind::Rmse | MetricKind::L1 => {
                let loss = |label: f32, s: f64| -> f64 {
                    let d = s - label as f64;
                    match self.kind {
                        MetricKind::L1 => d.abs(),
                        _ => d * d,
                    }
                };
                let s = self.weighted_sum(score, objective, loss);
                let avg = s / self.sum_weights;
                if self.kind == MetricKind::Rmse { avg.sqrt() } else { avg }
            }
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

    #[test]
    fn auc_perfect_and_ties() {
        let m = Metric::new(MetricKind::Auc, &meta(vec![0.0, 0.0, 1.0, 1.0]));
        assert_eq!(m.eval(&[0.1, 0.2, 0.8, 0.9], None), 1.0);
        assert_eq!(m.eval(&[0.5, 0.5, 0.5, 0.5], None), 0.5);
    }

    #[test]
    fn l2_and_rmse() {
        let md = meta(vec![1.0, 2.0]);
        let l2 = Metric::new(MetricKind::L2, &md);
        let rmse = Metric::new(MetricKind::Rmse, &md);
        assert_eq!(l2.eval(&[2.0, 4.0], None), 2.5);
        assert_eq!(rmse.eval(&[2.0, 4.0], None), 2.5f64.sqrt());
    }

    #[test]
    fn logloss_clamps() {
        let m = Metric::new(MetricKind::BinaryLogloss, &meta(vec![1.0]));
        assert_eq!(m.eval(&[0.0], None), -K_EPSILON.ln());
    }
}
