//! upstream: src/objective/rank_objective.hpp (`RankingObjective`,
//! `LambdarankNDCG`, `RankXENDCG`).

use std::sync::Mutex;

use rayon::prelude::*;

use super::{RowObjective, ScoreView};
use crate::config::Config;
use crate::consts::{K_EPSILON, K_MIN_SCORE};
use crate::dataset::Metadata;
use crate::dcg::{self, DcgCalculator};
use crate::error::{LgbmError, Result};
use crate::random::Random;

/// Query data shared by the ranking objectives (upstream `RankingObjective`).
#[derive(Default)]
struct RankingBase {
    seed: i32,
    learning_rate: f64,
    position_bias_regularization: f64,
    label: Vec<f32>,
    weight: Option<Vec<f32>>,
    positions: Option<Vec<i32>>,
    num_position_ids: usize,
    boundaries: Vec<i32>,
    /// upstream `pos_biases_` (`label_t`), updated after every gradient pass.
    pos_biases: Mutex<Vec<f32>>,
}

impl RankingBase {
    fn new(cfg: &Config) -> Self {
        Self {
            seed: cfg.objective_seed,
            learning_rate: cfg.learning_rate,
            position_bias_regularization: cfg.lambdarank_position_bias_regularization,
            ..Default::default()
        }
    }

    fn init(&mut self, meta: &Metadata) -> Result<()> {
        self.label = meta.label.clone();
        self.weight = meta.weight.clone();
        self.positions = meta.positions.clone();
        self.num_position_ids = if self.positions.is_some() { meta.position_ids.len() } else { 0 };
        self.boundaries = meta
            .query_boundaries
            .clone()
            .ok_or_else(|| LgbmError::InvalidData("Ranking tasks require query information".into()))?;
        *self.pos_biases.lock().unwrap() = vec![0.0; self.num_position_ids];
        Ok(())
    }

    fn num_queries(&self) -> usize {
        self.boundaries.len() - 1
    }

    /// upstream `RankingObjective::GetGradients`: per-query gradients from
    /// position-adjusted scores, scaled by the row weights.
    fn gradients(
        &self,
        score: &[f64],
        grad: &mut [f32],
        hess: &mut [f32],
        one_query: impl Fn(usize, &[f32], &[f64], &mut [f32], &mut [f32]) + Sync,
    ) {
        let biases = self.pos_biases.lock().unwrap().clone();
        let mut chunks = Vec::with_capacity(self.num_queries());
        let (mut g_rest, mut h_rest) = (&mut grad[..], &mut hess[..]);
        for q in self.boundaries.windows(2) {
            let cnt = (q[1] - q[0]) as usize;
            let (g, gr) = std::mem::take(&mut g_rest).split_at_mut(cnt);
            let (h, hr) = std::mem::take(&mut h_rest).split_at_mut(cnt);
            chunks.push((q[0] as usize, g, h));
            g_rest = gr;
            h_rest = hr;
        }
        chunks.into_par_iter().enumerate().for_each(|(q, (start, g, h))| {
            let cnt = g.len();
            let label = &self.label[start..start + cnt];
            match (&self.positions, self.num_position_ids > 0) {
                (Some(pos), true) => {
                    let adjusted: Vec<f64> = (start..start + cnt)
                        .map(|i| score[i] + biases[pos[i] as usize] as f64)
                        .collect();
                    one_query(q, label, &adjusted, g, h);
                }
                _ => one_query(q, label, &score[start..start + cnt], g, h),
            }
            if let Some(w) = &self.weight {
                for j in 0..cnt {
                    g[j] *= w[start + j];
                    h[j] *= w[start + j];
                }
            }
        });
    }
}

/// upstream `Common::Pow(T base, int power)` with an `int` base: the
/// squaring happens in `int` (wrapping like the upstream binary).
fn pow_int(base: i32, power: i32) -> f64 {
    if power < 0 {
        1.0 / pow_int(base, -power)
    } else if power == 0 {
        1.0
    } else if power % 2 == 0 {
        pow_int(base.wrapping_mul(base), power / 2)
    } else if power % 3 == 0 {
        pow_int(base.wrapping_mul(base).wrapping_mul(base), power / 3)
    } else {
        base as f64 * pow_int(base, power - 1)
    }
}

const SIGMOID_BINS: usize = 1024 * 1024;

pub struct LambdarankNdcg {
    base: RankingBase,
    sigmoid: f64,
    norm: bool,
    truncation_level: i32,
    dcg: DcgCalculator,
    inverse_max_dcgs: Vec<f64>,
    sigmoid_table: Vec<f64>,
    min_sigmoid_input: f64,
    max_sigmoid_input: f64,
    sigmoid_table_idx_factor: f64,
}

impl LambdarankNdcg {
    pub fn new(cfg: &Config) -> Result<Self> {
        let label_gain = dcg::default_label_gain(&cfg.label_gain);
        if cfg.sigmoid <= 0.0 {
            return Err(LgbmError::InvalidParameter(format!(
                "Sigmoid param {:.6} should be greater than zero",
                cfg.sigmoid
            )));
        }
        Ok(Self {
            base: RankingBase::new(cfg),
            sigmoid: cfg.sigmoid,
            norm: cfg.lambdarank_norm,
            truncation_level: cfg.lambdarank_truncation_level,
            dcg: DcgCalculator::new(label_gain),
            inverse_max_dcgs: Vec::new(),
            sigmoid_table: Vec::new(),
            min_sigmoid_input: -50.0,
            max_sigmoid_input: 50.0,
            sigmoid_table_idx_factor: 0.0,
        })
    }

    /// upstream `LambdarankNDCG::ConstructSigmoidTable`.
    fn construct_sigmoid_table(&mut self) {
        self.min_sigmoid_input = self.min_sigmoid_input / self.sigmoid / 2.0;
        self.max_sigmoid_input = -self.min_sigmoid_input;
        self.sigmoid_table_idx_factor = SIGMOID_BINS as f64 / (self.max_sigmoid_input - self.min_sigmoid_input);
        let (factor, min, sigmoid) = (self.sigmoid_table_idx_factor, self.min_sigmoid_input, self.sigmoid);
        self.sigmoid_table = (0..SIGMOID_BINS)
            .into_par_iter()
            .map(|i| {
                let score = i as f64 / factor + min;
                1.0 / (1.0 + (score * sigmoid).exp())
            })
            .collect();
    }

    #[inline]
    fn get_sigmoid(&self, score: f64) -> f64 {
        if score <= self.min_sigmoid_input {
            self.sigmoid_table[0]
        } else if score >= self.max_sigmoid_input {
            self.sigmoid_table[SIGMOID_BINS - 1]
        } else {
            self.sigmoid_table[((score - self.min_sigmoid_input) * self.sigmoid_table_idx_factor) as usize]
        }
    }

    /// upstream `LambdarankNDCG::GetGradientsForOneQuery`.
    fn one_query(&self, q: usize, label: &[f32], score: &[f64], lambdas: &mut [f32], hessians: &mut [f32]) {
        let cnt = label.len();
        let inverse_max_dcg = self.inverse_max_dcgs[q];
        lambdas.fill(0.0);
        hessians.fill(0.0);
        if cnt == 0 {
            return;
        }
        let sorted = dcg::sort_by_score_desc(score);
        let best_score = score[sorted[0]];
        let mut worst_idx = cnt - 1;
        if worst_idx > 0 && score[sorted[worst_idx]] == K_MIN_SCORE {
            worst_idx -= 1;
        }
        let worst_score = score[sorted[worst_idx]];
        let gain = self.dcg.label_gain();
        let mut sum_lambdas = 0.0f64;
        let mut i = 0usize;
        while i + 1 < cnt && (i as i64) < self.truncation_level as i64 {
            if score[sorted[i]] == K_MIN_SCORE {
                i += 1;
                continue;
            }
            for j in i + 1..cnt {
                if score[sorted[j]] == K_MIN_SCORE {
                    continue;
                }
                if label[sorted[i]] == label[sorted[j]] {
                    continue;
                }
                let (high_rank, low_rank) = if label[sorted[i]] > label[sorted[j]] { (i, j) } else { (j, i) };
                let high = sorted[high_rank];
                let low = sorted[low_rank];
                let high_score = score[high];
                let low_score = score[low];
                let high_label_gain = gain[label[high] as i32 as usize];
                let low_label_gain = gain[label[low] as i32 as usize];
                let high_discount = self.dcg.discount(high_rank);
                let low_discount = self.dcg.discount(low_rank);
                let delta_score = high_score - low_score;
                let dcg_gap = high_label_gain - low_label_gain;
                let paired_discount = (high_discount - low_discount).abs();
                let mut delta_pair_ndcg = dcg_gap * paired_discount * inverse_max_dcg;
                if self.norm && best_score != worst_score {
                    delta_pair_ndcg /= 0.01f32 as f64 + delta_score.abs();
                }
                let mut p_lambda = self.get_sigmoid(delta_score);
                let mut p_hessian = p_lambda * (1.0 - p_lambda);
                p_lambda *= -self.sigmoid * delta_pair_ndcg;
                p_hessian *= self.sigmoid * self.sigmoid * delta_pair_ndcg;
                lambdas[low] -= p_lambda as f32;
                hessians[low] += p_hessian as f32;
                lambdas[high] += p_lambda as f32;
                hessians[high] += p_hessian as f32;
                sum_lambdas -= 2.0 * p_lambda;
            }
            i += 1;
        }
        if self.norm && sum_lambdas > 0.0 {
            let norm_factor = (1.0 + sum_lambdas).log2() / sum_lambdas;
            for (l, h) in lambdas.iter_mut().zip(hessians.iter_mut()) {
                *l = (*l as f64 * norm_factor) as f32;
                *h = (*h as f64 * norm_factor) as f32;
            }
        }
    }

    /// upstream `LambdarankNDCG::UpdatePositionBiasFactors`, accumulated in
    /// row order (upstream's single-thread order; with several OpenMP threads
    /// upstream sums per-thread partials, which can differ in the last bits).
    fn update_position_bias_factors(&self, lambdas: &[f32], hessians: &[f32]) {
        let (Some(pos), npos) = (&self.base.positions, self.base.num_position_ids) else { return };
        let mut first = vec![0.0f64; npos];
        let mut second = vec![0.0f64; npos];
        let mut count = vec![0i32; npos];
        for i in 0..lambdas.len() {
            let p = pos[i] as usize;
            first[p] -= lambdas[i] as f64;
            second[p] -= hessians[i] as f64;
            count[p] += 1;
        }
        let reg = self.base.position_bias_regularization;
        let mut biases = self.base.pos_biases.lock().unwrap();
        for i in 0..npos {
            let mut d1 = 0.0 + first[i];
            let mut d2 = 0.0 + second[i];
            d1 -= biases[i] as f64 * reg * count[i] as f64;
            d2 -= reg * count[i] as f64;
            biases[i] = (biases[i] as f64 + self.base.learning_rate * d1 / (d2.abs() + 0.001)) as f32;
        }
    }
}

impl RowObjective for LambdarankNdcg {
    fn name(&self) -> &str {
        "lambdarank"
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        self.base.init(meta)?;
        dcg::check_metadata(meta)?;
        self.dcg.check_label(&self.base.label)?;
        let b = &self.base.boundaries;
        let k = self.truncation_level.max(0) as usize;
        self.inverse_max_dcgs = b
            .windows(2)
            .map(|q| {
                let v = self.dcg.max_dcg_at_k(k, &self.base.label[q[0] as usize..q[1] as usize]);
                if v > 0.0 { 1.0 / v } else { v }
            })
            .collect();
        self.construct_sigmoid_table();
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        let n = scores.num_data;
        self.base.gradients(scores.scores, &mut grad[..n], &mut hess[..n], |q, l, s, g, h| {
            self.one_query(q, l, s, g, h)
        });
        if self.base.num_position_ids > 0 {
            self.update_position_bias_factors(&grad[..n], &hess[..n]);
        }
    }

    fn to_model_string(&self) -> String {
        "lambdarank".into()
    }
}

pub struct RankXendcg {
    base: RankingBase,
    /// upstream `rands_`: one generator per query, seeded `objective_seed + q`.
    rands: Vec<Mutex<Random>>,
}

impl RankXendcg {
    pub fn new(cfg: &Config) -> Self {
        Self { base: RankingBase::new(cfg), rands: Vec::new() }
    }

    /// upstream `RankXENDCG::GetGradientsForOneQuery`.
    fn one_query(&self, q: usize, label: &[f32], score: &[f64], lambdas: &mut [f32], hessians: &mut [f32]) {
        let cnt = label.len();
        if cnt <= 1 {
            lambdas.fill(0.0);
            hessians.fill(0.0);
            return;
        }
        // upstream Common::Softmax
        let mut rho = vec![0.0f64; cnt];
        let mut wmax = score[0];
        for &s in &score[1..] {
            // std::max(s, wmax)
            wmax = if s < wmax { wmax } else { s };
        }
        let mut wsum = 0.0f64;
        for (r, &s) in rho.iter_mut().zip(score) {
            *r = (s - wmax).exp();
            wsum += *r;
        }
        for r in rho.iter_mut() {
            *r /= wsum;
        }

        let mut params = vec![0.0f64; cnt];
        let mut inv_denominator = 0.0f64;
        {
            let mut rand = self.rands[q].lock().unwrap();
            for (p, &l) in params.iter_mut().zip(label) {
                *p = pow_int(2, l as i32) - rand.next_float() as f64;
                inv_denominator += *p;
            }
        }
        inv_denominator = 1.0 / K_EPSILON.max(inv_denominator);

        let mut sum_l1 = 0.0f64;
        for i in 0..cnt {
            let term = -params[i] * inv_denominator + rho[i];
            lambdas[i] = term as f32;
            params[i] = term / (1.0 - rho[i]);
            sum_l1 += params[i];
        }
        let mut sum_l2 = 0.0f64;
        for i in 0..cnt {
            let term = rho[i] * (sum_l1 - params[i]);
            lambdas[i] += term as f32;
            params[i] = term / (1.0 - rho[i]);
            sum_l2 += params[i];
        }
        for i in 0..cnt {
            lambdas[i] += (rho[i] * (sum_l2 - params[i])) as f32;
            hessians[i] = (rho[i] * (1.0 - rho[i])) as f32;
        }
    }
}

impl RowObjective for RankXendcg {
    fn name(&self) -> &str {
        "rank_xendcg"
    }

    fn init(&mut self, meta: &Metadata, _num_data: usize) -> Result<()> {
        self.base.init(meta)?;
        self.rands = (0..self.base.num_queries())
            .map(|q| Mutex::new(Random::new(self.base.seed.wrapping_add(q as i32))))
            .collect();
        Ok(())
    }

    fn gradients(&self, scores: ScoreView<'_>, grad: &mut [f32], hess: &mut [f32]) {
        let n = scores.num_data;
        self.base.gradients(scores.scores, &mut grad[..n], &mut hess[..n], |q, l, s, g, h| {
            self.one_query(q, l, s, g, h)
        });
    }

    fn to_model_string(&self) -> String {
        "rank_xendcg".into()
    }
}

/// Prediction-time ranking objective loaded from a model file: identity
/// output (upstream constructs the objective from the model string).
pub struct RankingForPrediction(pub &'static str);

impl RowObjective for RankingForPrediction {
    fn name(&self) -> &str {
        self.0
    }

    fn init(&mut self, _meta: &Metadata, _num_data: usize) -> Result<()> {
        Err(LgbmError::InvalidParameter(format!("objective {} loaded from a model cannot be trained", self.0)))
    }

    fn gradients(&self, _scores: ScoreView<'_>, _grad: &mut [f32], _hess: &mut [f32]) {
        unreachable!("prediction-only objective")
    }

    fn to_model_string(&self) -> String {
        self.0.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_int_matches_common_pow() {
        assert_eq!(pow_int(2, 0), 1.0);
        assert_eq!(pow_int(2, 5), 32.0);
        assert_eq!(pow_int(2, 30), (1u64 << 30) as f64);
        assert_eq!(pow_int(2, -1), 0.5);
    }

    fn grouped(label: Vec<f32>, counts: &[i32]) -> Metadata {
        let n = label.len();
        let mut m = Metadata { label, ..Default::default() };
        m.set_query(n, Some(counts)).unwrap();
        m
    }

    #[test]
    fn lambdarank_pair_gradients() {
        // one query, two documents: only the (high, low) pair contributes
        let cfg = Config::from_pairs([("objective", "lambdarank"), ("lambdarank_norm", "false")]).unwrap();
        let mut o = LambdarankNdcg::new(&cfg).unwrap();
        let meta = grouped(vec![1.0, 0.0], &[2]);
        o.init(&meta, 2).unwrap();
        let scores = [0.0, 0.0];
        let (mut g, mut h) = (vec![0.0f32; 2], vec![0.0f32; 2]);
        o.gradients(ScoreView { scores: &scores, num_data: 2, num_outputs: 1 }, &mut g, &mut h);
        // max DCG = 1 (gain 1 at rank 0); |discount(0) - discount(1)| = 1 - 1/log2(3)
        let dndcg = 1.0 * (1.0 - 1.0 / 3f64.log2());
        let p = o.get_sigmoid(0.0);
        assert!((p - 0.5).abs() < 1e-4);
        let lam = (p * -dndcg) as f32;
        assert_eq!(g, vec![lam, -lam]);
        let hh = (p * (1.0 - p) * dndcg) as f32;
        assert_eq!(h, vec![hh, hh]);
    }

    #[test]
    fn ranking_requires_queries_and_valid_labels() {
        let cfg = Config::from_pairs([("objective", "lambdarank")]).unwrap();
        let meta = Metadata { label: vec![1.0, 0.0], ..Default::default() };
        let e = LambdarankNdcg::new(&cfg).unwrap().init(&meta, 2).unwrap_err();
        assert!(e.to_string().contains("Ranking tasks require query information"));
        let e = LambdarankNdcg::new(&cfg).unwrap().init(&grouped(vec![0.5, 0.0], &[2]), 2).unwrap_err();
        assert!(e.to_string().contains("label should be int type"));
        let bad = Config::from_pairs([("objective", "lambdarank"), ("sigmoid", "-1")]);
        assert!(bad.is_err() || LambdarankNdcg::new(&bad.unwrap()).is_err());
    }

    #[test]
    fn xendcg_single_document_queries_are_zero() {
        let cfg = Config::from_pairs([("objective", "rank_xendcg")]).unwrap();
        let mut o = RankXendcg::new(&cfg);
        o.init(&grouped(vec![1.0, 0.0, 2.0], &[1, 2]), 3).unwrap();
        let scores = [0.3, 0.1, -0.2];
        let (mut g, mut h) = (vec![9.0f32; 3], vec![9.0f32; 3]);
        o.gradients(ScoreView { scores: &scores, num_data: 3, num_outputs: 1 }, &mut g, &mut h);
        assert_eq!((g[0], h[0]), (0.0, 0.0));
        // the higher-labelled document is pushed up
        assert!(g[2] < 0.0 && g[1] > 0.0, "{g:?}");
        assert!(h[1] > 0.0 && h[2] > 0.0);
    }
}
