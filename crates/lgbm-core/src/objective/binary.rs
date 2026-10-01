//! upstream: src/objective/binary_objective.hpp `BinaryLogloss`.

use super::{RowObjective, ScoreView};
use crate::config::Config;
use crate::consts::K_EPSILON;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};
use crate::fmt::fmt_g6;

pub struct BinaryLogloss {
    sigmoid: f64,
    is_unbalance: bool,
    scale_pos_weight: f64,
    /// `Some(c)`: a row is positive when `(int)label == c` (one-vs-all);
    /// `None`: when `label > 0`.
    pos_class: Option<i32>,
    label_weights: [f64; 2],
    is_pos: Vec<bool>,
    weight: Option<Vec<f32>>,
    need_train: bool,
    num_pos_data: usize,
    warnings: Vec<String>,
}

impl BinaryLogloss {
    pub fn new(cfg: &Config) -> Result<Self> {
        Self::with_pos_class(cfg, None)
    }

    pub(crate) fn with_pos_class(cfg: &Config, pos_class: Option<i32>) -> Result<Self> {
        if cfg.sigmoid <= 0.0 {
            return Err(LgbmError::InvalidParameter(format!(
                "Sigmoid parameter {} should be greater than zero",
                cfg.sigmoid
            )));
        }
        Ok(Self {
            sigmoid: cfg.sigmoid,
            is_unbalance: cfg.is_unbalance,
            scale_pos_weight: cfg.scale_pos_weight,
            pos_class,
            label_weights: [1.0, 1.0],
            is_pos: Vec::new(),
            weight: None,
            need_train: true,
            num_pos_data: 0,
            warnings: Vec::new(),
        })
    }

    pub fn for_prediction(sigmoid: f64) -> Self {
        Self {
            sigmoid,
            is_unbalance: false,
            scale_pos_weight: 1.0,
            pos_class: None,
            label_weights: [1.0, 1.0],
            is_pos: Vec::new(),
            weight: None,
            need_train: true,
            num_pos_data: 0,
            warnings: Vec::new(),
        }
    }
}

impl RowObjective for BinaryLogloss {
    fn name(&self) -> &str {
        "binary"
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        self.is_pos = match self.pos_class {
            None => meta.label.iter().map(|&l| l > 0.0).collect(),
            Some(c) => meta.label.iter().map(|&l| l as i32 == c).collect(),
        };
        self.weight = meta.weight.clone();
        let cnt_positive = self.is_pos.iter().filter(|&&p| p).count();
        let cnt_negative = self.is_pos.len() - cnt_positive;
        self.num_pos_data = cnt_positive;
        self.need_train = !(cnt_negative == 0 || cnt_positive == 0);
        if !self.need_train {
            self.warnings.push("Contains only one class".into());
        }
        self.label_weights = [1.0, 1.0];
        if self.is_unbalance && cnt_positive > 0 && cnt_negative > 0 {
            if cnt_positive > cnt_negative {
                self.label_weights[1] = 1.0;
                self.label_weights[0] = cnt_positive as f64 / cnt_negative as f64;
            } else {
                self.label_weights[1] = cnt_negative as f64 / cnt_positive as f64;
                self.label_weights[0] = 1.0;
            }
        }
        self.label_weights[1] *= self.scale_pos_weight;
        Ok(())
    }

    fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        if !self.need_train {
            return;
        }
        use rayon::prelude::*;
        let s = scores.scores;
        let n = scores.num_data;
        grad[..n].par_iter_mut().zip(&mut hess[..n]).with_min_len(4096).enumerate().for_each(|(i, (g, h))| {
            let is_pos = self.is_pos[i] as usize;
            let label = if is_pos == 1 { 1.0f64 } else { -1.0f64 };
            let label_weight = self.label_weights[is_pos];
            let response =
                -label * self.sigmoid / (1.0f32 as f64 + (label * self.sigmoid * s[i]).exp());
            let abs_response = response.abs();
            match &self.weight {
                None => {
                    *g = (response * label_weight) as f32;
                    *h = (abs_response * (self.sigmoid - abs_response) * label_weight) as f32;
                }
                Some(w) => {
                    let wi = w[i] as f64;
                    *g = (response * label_weight * wi) as f32;
                    *h = (abs_response * (self.sigmoid - abs_response) * label_weight * wi) as f32;
                }
            }
        });
    }

    fn boost_from_score(&self, _output: usize) -> f64 {
        let mut suml = 0.0f64;
        let sumw;
        match &self.weight {
            Some(w) => {
                let mut sw = 0.0f64;
                for (p, wi) in self.is_pos.iter().zip(w) {
                    suml += (*p as i32 as f64) * *wi as f64;
                    sw += *wi as f64;
                }
                sumw = sw;
            }
            None => {
                sumw = self.is_pos.len() as f64;
                for p in &self.is_pos {
                    suml += *p as i32 as f64;
                }
            }
        }
        let mut pavg = suml / sumw;
        pavg = pavg.min(1.0 - K_EPSILON);
        pavg = pavg.max(K_EPSILON);
        (pavg / (1.0f32 as f64 - pavg)).ln() / self.sigmoid
    }

    fn class_need_train(&self, _output: usize) -> bool {
        self.need_train
    }

    fn num_positive_data(&self) -> usize {
        self.num_pos_data
    }

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        out[0] = 1.0 / (1.0 + (-self.sigmoid * raw[0]).exp());
    }

    fn to_model_string(&self) -> String {
        format!("binary sigmoid:{}", fmt_g6(self.sigmoid))
    }
}
