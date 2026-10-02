//! upstream: src/objective/multiclass_objective.hpp `MulticlassSoftmax`,
//! `MulticlassOVA`.

use rayon::prelude::*;

use super::binary::BinaryLogloss;
use super::{RowObjective, ScoreView};
use crate::config::Config;
use crate::consts::K_EPSILON;
use crate::dataset::Metadata;
use crate::error::{LgbmError, Result};
use crate::fmt::fmt_g6;
use crate::threading::SharedMut;

/// upstream `Common::Softmax(const double*, double*, int)`.
pub(crate) fn softmax(input: &[f64], out: &mut [f64]) {
    let mut wmax = input[0];
    for &v in &input[1..] {
        // std::max(v, wmax)
        wmax = if v < wmax { wmax } else { v };
    }
    let mut wsum = 0.0f64;
    for (o, &v) in out.iter_mut().zip(input) {
        *o = (v - wmax).exp();
        wsum += *o;
    }
    for o in out.iter_mut() {
        *o /= wsum;
    }
}

/// Parse `num_class:K` (and `sigmoid:S`) from a model `objective=` line.
fn parse_tokens(tokens: &[&str]) -> Result<(i32, Option<f64>)> {
    let mut num_class = -1;
    let mut sigmoid = None;
    for t in tokens {
        if let Some((k, v)) = t.split_once(':') {
            match k {
                "num_class" => {
                    num_class = v.parse().map_err(|_| LgbmError::ModelFormat(format!("bad num_class: {v}")))?
                }
                "sigmoid" => {
                    sigmoid = Some(v.parse().map_err(|_| LgbmError::ModelFormat(format!("bad sigmoid: {v}")))?)
                }
                _ => {}
            }
        }
    }
    if num_class < 0 {
        return Err(LgbmError::ModelFormat("Objective should contain num_class field".into()));
    }
    Ok((num_class, sigmoid))
}

pub struct MulticlassSoftmax {
    num_class: usize,
    factor: f64,
    label_int: Vec<i32>,
    weight: Option<Vec<f32>>,
    class_init_probs: Vec<f64>,
}

impl MulticlassSoftmax {
    pub fn new(cfg: &Config) -> Self {
        Self::with_num_class(cfg.num_class)
    }

    fn with_num_class(num_class: i32) -> Self {
        Self {
            num_class: num_class.max(0) as usize,
            // num_class_ / (num_class_ - 1.0f): the denominator is a float
            factor: num_class as f64 / (num_class as f32 - 1.0f32) as f64,
            label_int: Vec::new(),
            weight: None,
            class_init_probs: Vec::new(),
        }
    }

    pub fn for_prediction(tokens: &[&str]) -> Result<Self> {
        Ok(Self::with_num_class(parse_tokens(tokens)?.0))
    }
}

impl RowObjective for MulticlassSoftmax {
    fn name(&self) -> &str {
        "multiclass"
    }

    fn need_accurate_prediction(&self) -> bool {
        false
    }

    fn num_outputs(&self) -> usize {
        self.num_class
    }

    fn init(&mut self, meta: &Metadata, num_data: usize) -> Result<()> {
        let k = self.num_class;
        self.weight = meta.weight.clone();
        self.label_int = Vec::with_capacity(num_data);
        self.class_init_probs = vec![0.0; k];
        let mut sum_weight = 0.0f64;
        for (i, &l) in meta.label.iter().enumerate() {
            let li = l as i32;
            if li < 0 || li >= k as i32 {
                return Err(LgbmError::InvalidData(format!("Label must be in [0, {k}), but found {li} in label")));
            }
            self.label_int.push(li);
            match &self.weight {
                None => self.class_init_probs[li as usize] += 1.0,
                Some(w) => {
                    self.class_init_probs[li as usize] += w[i] as f64;
                    sum_weight += w[i] as f64;
                }
            }
        }
        if self.weight.is_none() {
            sum_weight = num_data as f64;
        }
        for p in &mut self.class_init_probs {
            *p /= sum_weight;
        }
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        let n = scores.num_data;
        let k = self.num_class;
        let s = scores.scores;
        let (g_out, h_out) = (SharedMut::new(grad), SharedMut::new(hess));
        (0..n).into_par_iter().with_min_len(1024).for_each_init(
            || (vec![0.0f64; k], vec![0.0f64; k]),
            |(raw, rec), i| {
                for c in 0..k {
                    raw[c] = s[c * n + i];
                }
                softmax(raw, rec);
                let w = self.weight.as_ref().map(|w| w[i] as f64);
                for (c, &p) in rec.iter().enumerate() {
                    let g = if self.label_int[i] == c as i32 { p - 1.0f32 as f64 } else { p };
                    let h = self.factor * p * (1.0f32 as f64 - p);
                    let (g, h) = match w {
                        None => (g, h),
                        Some(w) => (g * w, h * w),
                    };
                    // SAFETY: each (class, row) slot is written by exactly one row task.
                    unsafe {
                        *g_out.get(c * n + i) = g as f32;
                        *h_out.get(c * n + i) = h as f32;
                    }
                }
            },
        );
    }

    fn boost_from_score(&self, output: usize) -> f64 {
        K_EPSILON.max(self.class_init_probs[output]).ln()
    }

    fn class_need_train(&self, output: usize) -> bool {
        let p = self.class_init_probs[output].abs();
        !(p <= K_EPSILON || p >= 1.0 - K_EPSILON)
    }

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        softmax(raw, out);
    }

    fn to_model_string(&self) -> String {
        format!("multiclass num_class:{}", self.num_class)
    }
}

pub struct MulticlassOva {
    sigmoid: f64,
    binary: Vec<BinaryLogloss>,
}

impl MulticlassOva {
    pub fn new(cfg: &Config) -> Result<Self> {
        let binary =
            (0..cfg.num_class).map(|c| BinaryLogloss::with_pos_class(cfg, Some(c))).collect::<Result<Vec<_>>>()?;
        Ok(Self { sigmoid: cfg.sigmoid, binary })
    }

    pub fn for_prediction(tokens: &[&str]) -> Result<Self> {
        let (num_class, sigmoid) = parse_tokens(tokens)?;
        let sigmoid = sigmoid.unwrap_or(-1.0);
        if sigmoid <= 0.0 {
            return Err(LgbmError::ModelFormat(format!("Sigmoid parameter {sigmoid:.6} should be greater than zero")));
        }
        let binary = (0..num_class).map(|_| BinaryLogloss::for_prediction(sigmoid)).collect();
        Ok(Self { sigmoid, binary })
    }
}

impl RowObjective for MulticlassOva {
    fn name(&self) -> &str {
        "multiclassova"
    }

    fn need_accurate_prediction(&self) -> bool {
        false
    }

    fn num_outputs(&self) -> usize {
        self.binary.len()
    }

    fn init(&mut self, meta: &Metadata, num_data: usize) -> Result<()> {
        for b in &mut self.binary {
            b.init(meta, num_data)?;
        }
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        let n = scores.num_data;
        for (c, b) in self.binary.iter().enumerate() {
            let view = ScoreView { scores: &scores.scores[c * n..(c + 1) * n], num_data: n, num_outputs: 1 };
            b.gradients(view, &mut grad[c * n..(c + 1) * n], &mut hess[c * n..(c + 1) * n]);
        }
    }

    fn boost_from_score(&self, output: usize) -> f64 {
        self.binary[output].boost_from_score(0)
    }

    fn class_need_train(&self, output: usize) -> bool {
        self.binary[output].class_need_train(0)
    }

    fn convert_output(&self, raw: &[f64], out: &mut [f64]) {
        for (o, &r) in out.iter_mut().zip(raw) {
            *o = 1.0 / (1.0 + (-self.sigmoid * r).exp());
        }
    }

    fn to_model_string(&self) -> String {
        format!("multiclassova num_class:{} sigmoid:{}", self.binary.len(), fmt_g6(self.sigmoid))
    }
}
