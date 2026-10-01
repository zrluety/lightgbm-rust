//! upstream: src/objective/regression_objective.hpp `RegressionL2loss`.

use super::{RowObjective, ScoreView};
use crate::config::Config;
use crate::dataset::Metadata;
use crate::error::Result;

pub struct RegressionL2 {
    sqrt: bool,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
}

#[inline]
fn sign(x: f64) -> f64 {
    // upstream Common::Sign: (x > 0) - (x < 0)
    ((x > 0.0) as i32 - (x < 0.0) as i32) as f64
}

impl RegressionL2 {
    pub fn new(cfg: &Config) -> Self {
        Self::with_sqrt(cfg.reg_sqrt)
    }

    pub fn with_sqrt(sqrt: bool) -> Self {
        Self { sqrt, label: Vec::new(), weight: None }
    }
}

impl RowObjective for RegressionL2 {
    fn name(&self) -> &str {
        "regression"
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        self.label = if self.sqrt {
            meta.label
                .iter()
                .map(|&l| (sign(l as f64) as f32) * (l.abs()).sqrt())
                .collect()
        } else {
            meta.label.clone()
        };
        self.weight = meta.weight.clone();
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        use rayon::prelude::*;
        let s = scores.scores;
        let n = scores.num_data;
        let rows = grad[..n].par_iter_mut().zip(&mut hess[..n]).with_min_len(4096).enumerate();
        match &self.weight {
            None => rows.for_each(|(i, (g, h))| {
                *g = (s[i] - self.label[i] as f64) as f32;
                *h = 1.0;
            }),
            Some(w) => rows.for_each(|(i, (g, h))| {
                *g = ((s[i] - self.label[i] as f64) as f32) * w[i];
                *h = w[i];
            }),
        }
    }

    fn boost_from_score(&self, _output: usize) -> f64 {
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

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out[0] = if self.sqrt { sign(raw[0]) * raw[0] * raw[0] } else { raw[0] };
    }

    fn to_model_string(&self) -> String {
        if self.sqrt { "regression sqrt".into() } else { "regression".into() }
    }
}
