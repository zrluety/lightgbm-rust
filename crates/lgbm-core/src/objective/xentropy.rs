//! upstream: src/objective/xentropy_objective.hpp `CrossEntropy` and
//! `CrossEntropyLambda` (labels in `[0, 1]`).

use super::{RowObjective, ScoreView};
use crate::consts::K_EPSILON;
use crate::dataset::{check_elements_interval_closed, Metadata};
use crate::error::{LgbmError, Result};
use crate::fmt::fmt_g6;

/// upstream: utils/common.h `ObtainMinMaxSum`: one pass in pairs, the sum
/// accumulated in the element type (`float` for weights).
pub(crate) fn obtain_min_max_sum(w: &[f32]) -> (f32, f32, f32) {
    let n = w.len();
    let (mut minw, mut maxw, mut sumw, mut i);
    if n & 1 == 1 {
        minw = w[0];
        maxw = w[0];
        sumw = w[0];
        i = 2;
    } else {
        if w[0] < w[1] {
            minw = w[0];
            maxw = w[1];
        } else {
            minw = w[1];
            maxw = w[0];
        }
        sumw = w[0] + w[1];
        i = 3;
    }
    // std::min / std::max comparison semantics (NaN handling included)
    let mn = |a: f32, b: f32| if b < a { b } else { a };
    let mx = |a: f32, b: f32| if a < b { b } else { a };
    while i < n {
        if w[i - 1] < w[i] {
            minw = mn(minw, w[i - 1]);
            maxw = mx(maxw, w[i]);
        } else {
            minw = mn(minw, w[i]);
            maxw = mx(maxw, w[i - 1]);
        }
        sumw += w[i - 1] + w[i];
        i += 2;
    }
    (minw, maxw, sumw)
}

/// The `[0, 1]` label check shared by the cross-entropy objectives and metrics.
pub(crate) fn check_unit_interval(label: &[f32], caller: &str) -> Result<()> {
    check_elements_interval_closed(label, 0.0f32, 1.0f32, caller, |v| fmt_g6(v as f64))
}

/// upstream: `BoostFromScore` of both objectives (label mean, weighted when
/// there are weights).
fn label_mean(label: &[f32], weight: Option<&[f32]>) -> f64 {
    let mut suml = 0.0f64;
    let sumw = match weight {
        Some(w) => {
            let mut sw = 0.0f64;
            for (&l, &wi) in label.iter().zip(w) {
                suml += l as f64 * wi as f64;
                sw += wi as f64;
            }
            sw
        }
        None => {
            for &l in label {
                suml += l as f64;
            }
            label.len() as f64
        }
    };
    suml / sumw
}

/// upstream `CrossEntropy`: `p = sigmoid(f)`, weights enter the loss linearly.
pub struct CrossEntropy {
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
}

impl CrossEntropy {
    pub fn new() -> Self {
        Self { label: Vec::new(), weight: None }
    }
}

impl Default for CrossEntropy {
    fn default() -> Self {
        Self::new()
    }
}

impl RowObjective for CrossEntropy {
    fn name(&self) -> &str {
        "cross_entropy"
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        check_unit_interval(&meta.label, "cross_entropy")?;
        if let Some(w) = &meta.weight {
            let (minw, _, sumw) = obtain_min_max_sum(w);
            if minw < 0.0 {
                return Err(LgbmError::InvalidData("[cross_entropy]: at least one weight is negative".into()));
            }
            if sumw as f64 == 0.0 {
                return Err(LgbmError::InvalidData("[cross_entropy]: sum of weights is zero".into()));
            }
        }
        self.label = meta.label.clone();
        self.weight = meta.weight.clone();
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        use rayon::prelude::*;
        let s = scores.scores;
        let n = scores.num_data;
        grad[..n].par_iter_mut().zip(&mut hess[..n]).with_min_len(4096).enumerate().for_each(|(i, (g, h))| {
            let l = self.label[i];
            let (gi, hi) = if s[i] > -37.0 {
                let e = (-s[i]).exp();
                (((1.0f32 - l) as f64 - l as f64 * e) / (1.0 + e), e / ((1.0 + e) * (1.0 + e)))
            } else {
                let e = s[i].exp();
                (e - l as f64, e)
            };
            match &self.weight {
                None => {
                    *g = gi as f32;
                    *h = hi as f32;
                }
                Some(w) => {
                    *g = (gi * w[i] as f64) as f32;
                    *h = (hi * w[i] as f64) as f32;
                }
            }
        });
    }

    fn boost_from_score(&self, _output: usize) -> f64 {
        // clamp keeps a NaN mean, as std::min / std::max do
        let pavg = label_mean(&self.label, self.weight.as_deref()).clamp(K_EPSILON, 1.0 - K_EPSILON);
        (pavg / (1.0f32 as f64 - pavg)).ln()
    }

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out[0] = 1.0 / (1.0 + (-raw[0]).exp());
    }

    fn to_model_string(&self) -> String {
        "cross_entropy".into()
    }
}

/// upstream `CrossEntropyLambda`: `p = 1 - exp(-lambda * w)` with
/// `lambda = log(1 + exp(f))`; predictions are `lambda`.
pub struct CrossEntropyLambda {
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
}

impl CrossEntropyLambda {
    pub fn new() -> Self {
        Self { label: Vec::new(), weight: None }
    }
}

impl Default for CrossEntropyLambda {
    fn default() -> Self {
        Self::new()
    }
}

impl RowObjective for CrossEntropyLambda {
    fn name(&self) -> &str {
        "cross_entropy_lambda"
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        check_unit_interval(&meta.label, "cross_entropy_lambda")?;
        if let Some(w) = &meta.weight {
            let (minw, _, _) = obtain_min_max_sum(w);
            if minw <= 0.0 {
                return Err(LgbmError::InvalidData(
                    "[cross_entropy_lambda]: at least one weight is non-positive".into(),
                ));
            }
        }
        self.label = meta.label.clone();
        self.weight = meta.weight.clone();
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        use rayon::prelude::*;
        let s = scores.scores;
        let n = scores.num_data;
        grad[..n].par_iter_mut().zip(&mut hess[..n]).with_min_len(4096).enumerate().for_each(|(i, (g, h))| {
            match &self.weight {
                None => {
                    let z = 1.0 / (1.0 + (-s[i]).exp());
                    *g = (z - self.label[i] as f64) as f32;
                    *h = (z * (1.0 - z)) as f32;
                }
                Some(wv) => {
                    let w = wv[i] as f64;
                    let y = self.label[i] as f64;
                    let epf = s[i].exp();
                    let hhat = epf.ln_1p();
                    let z = 1.0 - (-w * hhat).exp();
                    let enf = 1.0 / epf;
                    *g = ((1.0 - y / z) * w / (1.0 + enf)) as f32;
                    let c = 1.0 / (1.0 - z);
                    let mut d = 1.0 + epf;
                    let a = w * epf / (d * d);
                    d = c - 1.0;
                    let b = (c / (d * d)) * (1.0 + w * epf - c);
                    *h = (a * (1.0 + y * b)) as f32;
                }
            }
        });
    }

    fn boost_from_score(&self, _output: usize) -> f64 {
        label_mean(&self.label, self.weight.as_deref()).exp_m1().ln()
    }

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out[0] = raw[0].exp().ln_1p();
    }

    fn to_model_string(&self) -> String {
        "cross_entropy_lambda".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_max_sum_pairs_like_upstream() {
        assert_eq!(obtain_min_max_sum(&[3.0]), (3.0, 3.0, 3.0));
        assert_eq!(obtain_min_max_sum(&[2.0, 1.0]), (1.0, 2.0, 3.0));
        assert_eq!(obtain_min_max_sum(&[1.0, 5.0, -2.0]), (-2.0, 5.0, 4.0));
        assert_eq!(obtain_min_max_sum(&[4.0, 1.0, 0.5, 6.0]), (0.5, 6.0, 11.5));
    }

    #[test]
    fn label_check_reports_upstream_element() {
        let err = check_unit_interval(&[0.5, 1.5, 0.2], "cross_entropy").unwrap_err().to_string();
        assert!(err.contains("[cross_entropy]: does not tolerate element [#1 = 1.5] outside [0, 1]"), "{err}");
        assert!(check_unit_interval(&[0.0, 1.0, 0.25], "x").is_ok());
    }
}
