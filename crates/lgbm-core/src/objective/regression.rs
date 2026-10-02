//! upstream: src/objective/regression_objective.hpp — `RegressionL2loss` and
//! the losses derived from it (L1, Huber, Fair, Poisson, Quantile, MAPE,
//! Gamma, Tweedie).
//!
//! Gradient arithmetic follows the C++ expression by expression, including
//! where it runs in `float` (`score_t`, `label_t`) versus `double`.

use rayon::prelude::*;

use super::percentile::{percentile, weighted_percentile};
use super::{RowObjective, ScoreView};
use crate::config::Config;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegressionKind {
    L2,
    L1,
    Huber { alpha: f64 },
    Fair { c: f64 },
    Poisson { max_delta_step: f64 },
    /// `alpha` is `score_t` (f32) upstream.
    Quantile { alpha: f32 },
    Mape,
    Gamma,
    Tweedie { rho: f64 },
}

impl RegressionKind {
    pub fn name(&self) -> &'static str {
        match self {
            RegressionKind::L2 => "regression",
            RegressionKind::L1 => "regression_l1",
            RegressionKind::Huber { .. } => "huber",
            RegressionKind::Fair { .. } => "fair",
            RegressionKind::Poisson { .. } => "poisson",
            RegressionKind::Quantile { .. } => "quantile",
            RegressionKind::Mape => "mape",
            RegressionKind::Gamma => "gamma",
            RegressionKind::Tweedie { .. } => "tweedie",
        }
    }

    /// Poisson, Gamma and Tweedie use a log link (`RegressionPoissonLoss`).
    fn log_link(&self) -> bool {
        matches!(self, RegressionKind::Poisson { .. } | RegressionKind::Gamma | RegressionKind::Tweedie { .. })
    }

    /// Huber and the log-link losses disable `reg_sqrt` (with a warning upstream).
    fn allows_sqrt(&self) -> bool {
        !matches!(self, RegressionKind::Huber { .. }) && !self.log_link()
    }

    fn from_config(name: &str, cfg: &Config) -> Option<Self> {
        Some(match name {
            "regression" => RegressionKind::L2,
            "regression_l1" => RegressionKind::L1,
            "huber" => RegressionKind::Huber { alpha: cfg.alpha },
            "fair" => RegressionKind::Fair { c: cfg.fair_c },
            "poisson" => RegressionKind::Poisson { max_delta_step: cfg.poisson_max_delta_step },
            "quantile" => RegressionKind::Quantile { alpha: cfg.alpha as f32 },
            "mape" => RegressionKind::Mape,
            "gamma" => RegressionKind::Gamma,
            "tweedie" => RegressionKind::Tweedie { rho: cfg.tweedie_variance_power },
            _ => return None,
        })
    }
}

pub struct Regression {
    kind: RegressionKind,
    sqrt: bool,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    /// MAPE: `1 / max(1, |label|)` (times the weight, if any).
    label_weight: Vec<f32>,
}

/// Names accepted by [`Regression::new`].
pub const REGRESSION_OBJECTIVES: &[&str] =
    &["regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie"];

#[inline]
fn sign(x: f64) -> f64 {
    // upstream Common::Sign: (x > 0) - (x < 0)
    ((x > 0.0) as i32 - (x < 0.0) as i32) as f64
}

/// upstream Common::SafeLog
#[inline]
pub(crate) fn safe_log(x: f64) -> f64 {
    if x > 0.0 { x.ln() } else { f64::NEG_INFINITY }
}

impl Regression {
    pub fn new(cfg: &Config) -> Result<Self> {
        let kind = RegressionKind::from_config(&cfg.objective, cfg)
            .ok_or_else(|| LgbmError::Unsupported(format!("objective={}", cfg.objective)))?;
        if let RegressionKind::Quantile { alpha } = kind {
            if !(alpha > 0.0 && alpha < 1.0) {
                return Err(LgbmError::InvalidParameter("Check failed: alpha_ > 0 && alpha_ < 1".into()));
            }
        }
        let obj = Self::with_kind(kind, cfg.reg_sqrt && kind.allows_sqrt());
        if cfg.reg_sqrt && !kind.allows_sqrt() {
            crate::log::warning(&format!("Cannot use sqrt transform in {} Regression, will auto disable it", kind.name()));
        }
        Ok(obj)
    }

    pub fn with_kind(kind: RegressionKind, sqrt: bool) -> Self {
        Self { kind, sqrt, label: Vec::new(), weight: None, label_weight: Vec::new() }
    }

    /// Prediction-time objective from a model file `objective=` line; only
    /// the output transform matters, so loss parameters keep their defaults.
    pub fn for_prediction(name: &str, tokens: &[&str]) -> Option<Self> {
        let kind = RegressionKind::from_config(name, &Config::default())?;
        Some(Self::with_kind(kind, tokens.contains(&"sqrt") && kind.allows_sqrt()))
    }

    fn mean_label(&self) -> f64 {
        let mut suml = 0.0f64;
        match &self.weight {
            Some(w) => {
                let mut sumw = 0.0f64;
                for (l, wi) in self.label.iter().zip(w) {
                    suml += *l as f64 * *wi as f64;
                    sumw += *wi as f64;
                }
                suml / sumw
            }
            None => {
                for l in &self.label {
                    suml += *l as f64;
                }
                suml / self.label.len() as f64
            }
        }
    }

    /// Percentile of the labels (L1 / Quantile `BoostFromScore`).
    fn label_percentile(&self, alpha: f64) -> f64 {
        let n = self.label.len();
        let l = &self.label;
        match &self.weight {
            Some(w) => weighted_percentile(n, |i| l[i], |i| w[i] as f64, alpha) as f64,
            None => percentile(n, |i| l[i], alpha) as f64,
        }
    }
}

impl RowObjective for Regression {
    fn name(&self) -> &str {
        self.kind.name()
    }

    fn is_constant_hessian(&self) -> bool {
        match self.kind {
            RegressionKind::L2 | RegressionKind::L1 | RegressionKind::Huber { .. } | RegressionKind::Quantile { .. } => {
                self.weight.is_none()
            }
            RegressionKind::Mape => true,
            RegressionKind::Fair { .. }
            | RegressionKind::Poisson { .. }
            | RegressionKind::Gamma
            | RegressionKind::Tweedie { .. } => false,
        }
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        self.label = if self.sqrt {
            meta.label.iter().map(|&l| (sign(l as f64) as f32) * (l.abs()).sqrt()).collect()
        } else {
            meta.label.clone()
        };
        self.weight = meta.weight.clone();
        if self.kind.log_link() {
            let name = self.kind.name();
            let mut min = f32::INFINITY;
            let mut sum = 0.0f32;
            for &l in &self.label {
                min = min.min(l);
                sum += l;
            }
            if min < 0.0 {
                return Err(LgbmError::InvalidData(format!("[{name}]: at least one target label is negative")));
            }
            if sum == 0.0 {
                return Err(LgbmError::InvalidData(format!("[{name}]: sum of labels is zero")));
            }
        }
        if self.kind == RegressionKind::Mape {
            if self.label.iter().any(|l| l.abs() < 1.0) {
                crate::log::warning(
                    "Some label values are < 1 in absolute value. MAPE is unstable with such values, \
                     so LightGBM rounds them to 1.0 when calculating MAPE.",
                );
            }
            self.label_weight = match &self.weight {
                None => self.label.iter().map(|&l| 1.0f32 / 1.0f32.max(l.abs())).collect(),
                Some(w) => self.label.iter().zip(w).map(|(&l, &wi)| 1.0f32 / 1.0f32.max(l.abs()) * wi).collect(),
            };
        }
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        let s = scores.scores;
        let n = scores.num_data;
        let label = &self.label;
        let w = self.weight.as_deref();
        let kind = self.kind;
        let lw = &self.label_weight;
        let exp_max_delta_step = match kind {
            RegressionKind::Poisson { max_delta_step } => max_delta_step.exp(),
            _ => 0.0,
        };
        grad[..n].par_iter_mut().zip(&mut hess[..n]).with_min_len(4096).enumerate().for_each(|(i, (g, h))| {
            let l = label[i];
            let (gv, hv): (f32, f32) = match (kind, w) {
                (RegressionKind::L2, None) => ((s[i] - l as f64) as f32, 1.0),
                (RegressionKind::L2, Some(w)) => (((s[i] - l as f64) as f32) * w[i], w[i]),
                (RegressionKind::L1, None) => (sign(s[i] - l as f64) as f32, 1.0),
                (RegressionKind::L1, Some(w)) => ((sign(s[i] - l as f64) * w[i] as f64) as f32, w[i]),
                (RegressionKind::Huber { alpha }, w) => {
                    let diff = s[i] - l as f64;
                    match w {
                        None => {
                            let g = if diff.abs() <= alpha { diff as f32 } else { (sign(diff) * alpha) as f32 };
                            (g, 1.0)
                        }
                        Some(w) => {
                            let g = if diff.abs() <= alpha {
                                (diff * w[i] as f64) as f32
                            } else {
                                (sign(diff) * w[i] as f64 * alpha) as f32
                            };
                            (g, w[i])
                        }
                    }
                }
                (RegressionKind::Fair { c }, w) => {
                    let x = s[i] - l as f64;
                    let g = c * x / (x.abs() + c);
                    let hh = c * c / ((x.abs() + c) * (x.abs() + c));
                    match w {
                        None => (g as f32, hh as f32),
                        Some(w) => ((g * w[i] as f64) as f32, (hh * w[i] as f64) as f32),
                    }
                }
                (RegressionKind::Poisson { .. }, w) => {
                    let e = s[i].exp();
                    match w {
                        None => ((e - l as f64) as f32, (e * exp_max_delta_step) as f32),
                        Some(w) => {
                            (((e - l as f64) * w[i] as f64) as f32, (e * exp_max_delta_step * w[i] as f64) as f32)
                        }
                    }
                }
                (RegressionKind::Quantile { alpha }, w) => {
                    let delta = (s[i] - l as f64) as f32;
                    match w {
                        None => (if delta >= 0.0 { 1.0f32 - alpha } else { -alpha }, 1.0),
                        Some(w) => (if delta >= 0.0 { (1.0f32 - alpha) * w[i] } else { -alpha * w[i] }, w[i]),
                    }
                }
                (RegressionKind::Mape, w) => {
                    let g = (sign(s[i] - l as f64) * lw[i] as f64) as f32;
                    (g, w.map_or(1.0, |w| w[i]))
                }
                (RegressionKind::Gamma, w) => {
                    let e = (-s[i]).exp();
                    let l = l as f64;
                    match w {
                        None => ((1.0 - l * e) as f32, (l * e) as f32),
                        Some(w) => (((1.0 - l * e) * w[i] as f64) as f32, (l * e * w[i] as f64) as f32),
                    }
                }
                (RegressionKind::Tweedie { rho }, w) => {
                    let e1 = ((1.0 - rho) * s[i]).exp();
                    let e2 = ((2.0 - rho) * s[i]).exp();
                    let nl = -l as f64;
                    let g = nl * e1 + e2;
                    let hh = nl * (1.0 - rho) * e1 + (2.0 - rho) * e2;
                    match w {
                        None => (g as f32, hh as f32),
                        Some(w) => ((g * w[i] as f64) as f32, (hh * w[i] as f64) as f32),
                    }
                }
            };
            *g = gv;
            *h = hv;
        });
    }

    fn boost_from_score(&self, _output: usize) -> f64 {
        match self.kind {
            RegressionKind::L2 | RegressionKind::Huber { .. } | RegressionKind::Fair { .. } => self.mean_label(),
            RegressionKind::L1 => self.label_percentile(0.5),
            RegressionKind::Quantile { alpha } => self.label_percentile(alpha as f64),
            RegressionKind::Mape => {
                let (l, lw) = (&self.label, &self.label_weight);
                weighted_percentile(l.len(), |i| l[i], |i| lw[i] as f64, 0.5) as f64
            }
            RegressionKind::Poisson { .. } | RegressionKind::Gamma | RegressionKind::Tweedie { .. } => {
                safe_log(self.mean_label())
            }
        }
    }

    fn is_renew_tree_output(&self) -> bool {
        matches!(self.kind, RegressionKind::L1 | RegressionKind::Quantile { .. } | RegressionKind::Mape)
    }

    fn renew_leaf_output(&self, indices: &[u32], scores: &[f64]) -> f64 {
        let residual = |i: usize| {
            let r = indices[i] as usize;
            self.label[r] as f64 - scores[r]
        };
        let n = indices.len();
        let alpha = match self.kind {
            RegressionKind::Quantile { alpha } => alpha as f64,
            _ => 0.5,
        };
        match (self.kind, &self.weight) {
            (RegressionKind::Mape, _) => {
                weighted_percentile(n, residual, |i| self.label_weight[indices[i] as usize] as f64, alpha)
            }
            (_, None) => percentile(n, residual, alpha),
            (_, Some(w)) => weighted_percentile(n, residual, |i| w[indices[i] as usize] as f64, alpha),
        }
    }

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out[0] = if self.kind.log_link() {
            raw[0].exp()
        } else if self.sqrt {
            sign(raw[0]) * raw[0] * raw[0]
        } else {
            raw[0]
        };
    }

    fn to_model_string(&self) -> String {
        if self.sqrt { format!("{} sqrt", self.kind.name()) } else { self.kind.name().into() }
    }
}
