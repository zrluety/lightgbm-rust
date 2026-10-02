//! Gradient boosting driver.
//!
//! upstream: src/boosting/gbdt.cpp (`GBDT::Init`, `TrainOneIter`,
//! `BoostFromAverage`, `UpdateScore`, `GetEvalAt`, `InitPredict`,
//! `PredictRaw`, `FeatureImportance`) and src/boosting/dart.hpp.

use std::sync::Arc;

use crate::config::Config;
use crate::consts::K_EPSILON;
use crate::dataset::Dataset;
use crate::error::{LgbmError, Result};
use crate::learner::SerialTreeLearner;
use crate::metric::{Metric, MetricKind};
use crate::objective::{Objective, ScoreView, create_objective};
use crate::random::Random;
use crate::sample_strategy::SampleStrategy;
use crate::threading::resolve_num_threads;
use crate::tree::Tree;

/// Which prediction to produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictKind {
    /// Objective-transformed output (e.g. probability for `binary`).
    Normal,
    /// Raw additive score.
    Raw,
    /// Leaf index of every tree.
    LeafIndex,
    /// SHAP feature contributions plus the expected value, per class.
    Contrib,
}

/// One evaluation result: (dataset name, metric name, value, higher_is_better).
pub type EvalResult = (String, String, f64, bool);

struct ValidSet {
    name: String,
    data: Arc<Dataset>,
    scores: Vec<f64>,
    metrics: Vec<Metric>,
}

struct TrainState {
    data: Arc<Dataset>,
    learner: SerialTreeLearner,
    sampler: Option<SampleStrategy>,
    scores: Vec<f64>,
    has_init_score: bool,
    grad: Vec<f32>,
    hess: Vec<f32>,
    class_need_train: Vec<bool>,
    metrics: Vec<Metric>,
    valid: Vec<ValidSet>,
    iter: usize,
    /// upstream `shrinkage_rate_`.
    shrinkage_rate: f64,
    dart: Option<Dart>,
    /// Random forest: the constant scores the gradients were computed at
    /// (upstream `RF::init_scores_`).
    rf_init_scores: Option<Vec<f64>>,
}

/// upstream: src/boosting/dart.hpp (`DART` members).
struct Dart {
    tree_weight: Vec<f64>,
    sum_weight: f64,
    /// Iteration indices (including merged init iterations).
    drop_index: Vec<usize>,
    random_for_drop: Random,
    is_update_score_cur_iter: bool,
}

impl Dart {
    fn new(cfg: &Config) -> Self {
        Self {
            tree_weight: Vec::new(),
            sum_weight: 0.0,
            drop_index: Vec::new(),
            random_for_drop: Random::new(cfg.drop_seed),
            // upstream leaves this uninitialized until the first TrainOneIter
            is_update_score_cur_iter: false,
        }
    }
}

/// `std::min` (returns `a` when either is NaN).
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

pub struct Gbdt {
    pub(crate) config: Option<Config>,
    pub(crate) loaded_parameters: Option<String>,
    pub(crate) objective: Option<Objective>,
    pub(crate) models: Vec<Tree>,
    pub(crate) num_tree_per_iteration: usize,
    pub(crate) num_class: usize,
    pub(crate) label_index: i32,
    pub(crate) max_feature_idx: i32,
    pub(crate) feature_names: Vec<String>,
    pub(crate) feature_infos: Vec<String>,
    /// By real feature index; empty when unconstrained.
    pub(crate) monotone_constraints: Vec<i8>,
    /// Iterations merged from an init model (upstream `num_init_iteration_`).
    num_init_iteration: usize,
    /// Normal predictions are divided by the number of iterations (random
    /// forest models; upstream `average_output_`).
    pub(crate) average_output: bool,
    /// Trained with `boosting=rf` (upstream `RF`, which never uses
    /// prediction early stopping); models loaded from text are plain GBDT.
    pub(crate) is_rf: bool,
    train: Option<TrainState>,
    pool: Option<Arc<rayon::ThreadPool>>,
    warnings: Vec<String>,
}

fn build_pool(num_threads: i32) -> Result<Option<Arc<rayon::ThreadPool>>> {
    if num_threads <= 0 {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .build()
        .map(|p| Some(Arc::new(p)))
        .map_err(|e| LgbmError::Internal(format!("cannot build thread pool: {e}")))
}

impl Gbdt {
    /// A model with no trees and no training state.
    pub(crate) fn empty() -> Self {
        Self {
            config: None,
            loaded_parameters: None,
            objective: None,
            models: Vec::new(),
            num_tree_per_iteration: 1,
            num_class: 1,
            label_index: 0,
            max_feature_idx: -1,
            feature_names: Vec::new(),
            feature_infos: Vec::new(),
            monotone_constraints: Vec::new(),
            num_init_iteration: 0,
            average_output: false,
            is_rf: false,
            train: None,
            pool: None,
            warnings: Vec::new(),
        }
    }

    /// Drain the upstream warnings raised while creating the booster
    /// (objective construction and initialization).
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    /// Create a booster for training. `objective` overrides the built-in one
    /// chosen by `config.objective` (pass `None` to use the config).
    pub fn new(config: Config, train: Arc<Dataset>, objective: Option<Objective>) -> Result<Self> {
        // upstream: Dataset::GetShareStates
        if config.force_col_wise && config.force_row_wise {
            return Err(LgbmError::InvalidParameter(
                "Cannot set both of `force_col_wise` and `force_row_wise` to `true` at the same time".into(),
            ));
        }
        let mut objective = match objective {
            Some(o) => Some(o),
            None => create_objective(&config)?,
        };
        let n = train.num_data();
        let mut warnings = Vec::new();
        if let Some(o) = objective.as_mut() {
            o.init(&train.metadata, n)?;
            warnings = o.take_warnings();
        }
        let is_rf = config.boosting == "rf";
        // upstream: RF::Init, before GBDT::Init
        if is_rf
            && config.data_sample_strategy == "bagging"
            && !((config.bagging_freq > 0 && config.bagging_fraction < 1.0 && config.bagging_fraction > 0.0)
                || (config.feature_fraction < 1.0 && config.feature_fraction > 0.0))
        {
            return Err(LgbmError::InvalidParameter(
                "Check failed: (config->bagging_freq > 0 && config->bagging_fraction < 1.0f && \
                 config->bagging_fraction > 0.0f) || (config->feature_fraction < 1.0f && \
                 config->feature_fraction > 0.0f)"
                    .into(),
            ));
        }
        if is_rf && config.bagging_by_query {
            return Err(LgbmError::Unsupported(
                "boosting=rf with bagging_by_query (upstream RF bags in TrainOneIter, outside GBDT::Boosting)".into(),
            ));
        }
        // upstream: GBDT::Init
        if !config.monotone_constraints.is_empty() && train.num_total_features() != config.monotone_constraints.len() {
            return Err(LgbmError::InvalidParameter(
                "Check failed: (static_cast<size_t>(train_data_->num_total_features())) == \
                 (config->monotone_constraints.size())"
                    .into(),
            ));
        }
        if !config.feature_contri.is_empty() && train.num_total_features() != config.feature_contri.len() {
            return Err(LgbmError::InvalidParameter(
                "Check failed: (static_cast<size_t>(train_data_->num_total_features())) == \
                 (config->feature_contri.size())"
                    .into(),
            ));
        }
        if !config.monotone_constraints.is_empty() {
            if let Some(o) = objective.as_ref().filter(|o| o.is_renew_tree_output()) {
                return Err(LgbmError::InvalidParameter(format!(
                    "Cannot use ``monotone_constraints`` in {} objective, please disable it.",
                    o.name()
                )));
            }
        }
        // upstream GBDT::Init: num_class trees per iteration unless the objective says otherwise
        let num_class = config.num_class.max(1) as usize;
        let ntpi = objective.as_ref().map_or(num_class, |o| o.num_outputs());
        let class_need_train = (0..ntpi)
            .map(|k| objective.as_ref().is_none_or(|o| o.class_need_train(k)))
            .collect();
        let mut scores = vec![0.0; n * ntpi];
        let has_init_score = match train.init_score() {
            Some(s) => {
                if s.len() != n * ntpi {
                    return Err(LgbmError::InvalidData(format!(
                        "init_score has {} values, expected {}",
                        s.len(),
                        n * ntpi
                    )));
                }
                scores.copy_from_slice(s);
                true
            }
            None => false,
        };
        // upstream: c_api.cpp `CreateObjectiveAndMetrics` inits every training
        // metric (and so runs its checks) whether or not it is reported
        let metrics = Self::make_metrics(&config, &train)?;
        let metrics = if config.is_provide_training_metric { metrics } else { Vec::new() };
        let pool = build_pool(config.num_threads)?;
        let sampler =
            SampleStrategy::new(&config, &train, objective.as_ref(), ntpi, resolve_num_threads(config.num_threads))?;
        let learner = SerialTreeLearner::new(train.clone(), &config);
        if is_rf {
            // upstream: RF::Init after GBDT::Init
            if has_init_score {
                return Err(LgbmError::InvalidParameter(
                    "Check failed: (train_data->metadata().init_score()) == (nullptr)".into(),
                ));
            }
            if objective.is_none() {
                return Err(LgbmError::InvalidParameter(
                    "RF mode do not support custom objective function, please use built-in objectives.".into(),
                ));
            }
        }
        let mut g = Self {
            num_tree_per_iteration: ntpi,
            num_class,
            label_index: train.label_idx,
            max_feature_idx: train.num_total_features() as i32 - 1,
            feature_names: train.feature_names().to_vec(),
            feature_infos: train.feature_infos(),
            monotone_constraints: config.monotone_constraints.clone(),
            objective,
            models: Vec::new(),
            num_init_iteration: 0,
            average_output: is_rf,
            is_rf,
            train: Some(TrainState {
                data: train,
                learner,
                sampler,
                scores,
                has_init_score,
                grad: vec![0.0; n * ntpi],
                hess: vec![0.0; n * ntpi],
                class_need_train,
                metrics,
                valid: Vec::new(),
                iter: 0,
                shrinkage_rate: if is_rf { 1.0 } else { config.learning_rate },
                dart: (config.boosting == "dart").then(|| Dart::new(&config)),
                rf_init_scores: None,
            }),
            config: Some(config),
            loaded_parameters: None,
            pool,
            warnings,
        };
        if is_rf {
            g.rf_boosting();
        }
        Ok(g)
    }

    /// Gradients at the constant initial scores, computed once.
    ///
    /// upstream: `RF::Boosting`.
    fn rf_boosting(&mut self) {
        let ntpi = self.num_tree_per_iteration;
        let init: Vec<f64> = (0..ntpi).map(|k| self.boost_from_average_impl(k, false)).collect();
        let st = self.train.as_mut().expect("training state");
        let obj = self.objective.as_ref().expect("RF objective");
        let n = st.data.num_data();
        let mut tmp = vec![0.0; n * ntpi];
        for (k, &v) in init.iter().enumerate() {
            tmp[k * n..(k + 1) * n].fill(v);
        }
        let view = ScoreView { scores: &tmp, num_data: n, num_outputs: ntpi };
        let mut compute = || obj.gradients(view, &mut st.grad, &mut st.hess);
        match &self.pool {
            Some(p) => p.install(compute),
            None => compute(),
        }
        st.rf_init_scores = Some(init);
    }

    fn make_metrics(config: &Config, data: &Dataset) -> Result<Vec<Metric>> {
        config
            .metric
            .iter()
            .filter(|m| m.as_str() != "custom")
            .map(|m| Metric::new(MetricKind::from_name(m)?, &data.metadata, config))
            .collect()
    }

    pub(crate) fn install<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match &self.pool {
            Some(p) => p.install(f),
            None => f(),
        }
    }

    /// Attach a validation set (must share the training set's bin mappers).
    pub fn add_valid(&mut self, data: Arc<Dataset>, name: &str) -> Result<()> {
        let config = self.config.as_ref().ok_or_else(|| LgbmError::Internal("no config".into()))?;
        let metrics = Self::make_metrics(config, &data)?;
        let ntpi = self.num_tree_per_iteration;
        let st = self.train.as_mut().ok_or_else(|| LgbmError::InvalidData("booster is not training".into()))?;
        if !data.same_bins_as(&st.data) {
            return Err(LgbmError::InvalidData(
                "validation data must be constructed with reference to the training data".into(),
            ));
        }
        let n = data.num_data();
        let mut scores = vec![0.0; n * ntpi];
        if let Some(s) = data.init_score() {
            if s.len() != n * ntpi {
                return Err(LgbmError::InvalidData("validation init_score size mismatch".into()));
            }
            scores.copy_from_slice(s);
        }
        // Bring scores up to date with trees trained by this booster; merged
        // init-model trees are already part of the dataset's init_score.
        let start = self.num_init_iteration * ntpi;
        for (i, t) in self.models.iter().enumerate().skip(start) {
            let k = i % ntpi;
            t.add_prediction_to_score(&data, &mut scores[k * n..(k + 1) * n]);
        }
        // upstream: RF::AddValidDataset (1.0f / iterations, a float division)
        let iters = st.iter + self.num_init_iteration;
        if self.is_rf && iters > 0 {
            let f = (1.0f32 / iters as f32) as f64;
            for s in scores.iter_mut() {
                *s *= f;
            }
        }
        st.valid.push(ValidSet { name: name.to_string(), data, scores, metrics });
        Ok(())
    }

    /// Put `other`'s trees in front of this booster's (continued training).
    /// Their contribution must already be in the datasets' init scores.
    ///
    /// upstream: gbdt.h `GBDT::MergeFrom`.
    pub fn merge_from(&mut self, other: &Gbdt) {
        let mut models = other.models.clone();
        self.num_init_iteration = models.len() / self.num_tree_per_iteration;
        models.append(&mut self.models);
        self.models = models;
    }

    pub fn num_tree_per_iteration(&self) -> usize {
        self.num_tree_per_iteration
    }

    pub fn num_trees(&self) -> usize {
        self.models.len()
    }

    pub fn current_iteration(&self) -> usize {
        self.models.len() / self.num_tree_per_iteration
    }

    pub fn num_feature(&self) -> usize {
        (self.max_feature_idx + 1) as usize
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    pub fn trees(&self) -> &[Tree] {
        &self.models
    }

    pub fn objective(&self) -> Option<&Objective> {
        self.objective.as_ref()
    }

    pub fn config(&self) -> Option<&Config> {
        self.config.as_ref()
    }

    /// Enable split tracing on the tree learner (for differential tests).
    pub fn enable_trace(&mut self) {
        if let Some(st) = self.train.as_mut() {
            st.learner.trace = Some(Default::default());
        }
    }

    pub fn last_trace(&self) -> Option<&crate::learner::TreeTrace> {
        self.train.as_ref().and_then(|s| s.learner.trace.as_ref())
    }

    /// Current training scores (class-major).
    pub fn train_scores(&self) -> Option<&[f64]> {
        self.train.as_ref().map(|s| s.scores.as_slice())
    }

    /// Gradients and Hessians computed for the most recent iteration.
    pub fn last_gradients(&self) -> Option<(&[f32], &[f32])> {
        self.train.as_ref().map(|s| (s.grad.as_slice(), s.hess.as_slice()))
    }

    /// upstream: `GBDT::BoostFromAverage`.
    fn boost_from_average(&mut self, k: usize) -> f64 {
        self.boost_from_average_impl(k, true)
    }

    fn boost_from_average_impl(&mut self, k: usize, update_scorer: bool) -> f64 {
        let cfg = self.config.as_ref().expect("training config");
        let st = self.train.as_mut().expect("training state");
        if self.models.is_empty() && !st.has_init_score {
            if let Some(obj) = self.objective.as_ref() {
                if cfg.boost_from_average || st.data.num_features() == 0 {
                    let init = obj.boost_from_score(k);
                    if init.abs() > K_EPSILON {
                        if update_scorer {
                            add_const(&mut st.scores, k, st.data.num_data(), init);
                            for v in st.valid.iter_mut() {
                                let n = v.data.num_data();
                                add_const(&mut v.scores, k, n, init);
                            }
                        }
                        return init;
                    }
                }
            }
        }
        0.0
    }

    /// Training scores as the objective sees them; for DART this drops the
    /// iteration's trees first (once per iteration).
    ///
    /// upstream: `GBDT::GetTrainingScore` / `DART::GetTrainingScore`.
    pub fn training_score(&mut self) -> Option<&[f64]> {
        let pending = self
            .train
            .as_ref()
            .and_then(|s| s.dart.as_ref())
            .is_some_and(|d| !d.is_update_score_cur_iter);
        if pending {
            self.dart_dropping_trees();
            self.train.as_mut().unwrap().dart.as_mut().unwrap().is_update_score_cur_iter = true;
        }
        self.train_scores()
    }

    /// upstream: `DART::DroppingTrees`.
    fn dart_dropping_trees(&mut self) {
        let Gbdt { config, models, train, pool, num_tree_per_iteration, num_init_iteration, .. } = self;
        let cfg = config.as_ref().expect("training config");
        let st = train.as_mut().expect("training state");
        let d = st.dart.as_mut().expect("dart state");
        let (ntpi, init) = (*num_tree_per_iteration, *num_init_iteration);
        d.drop_index.clear();
        // static_cast<size_t>(max_drop): a negative limit never stops the draw
        let max_drop = cfg.max_drop as usize;
        let is_skip = (d.random_for_drop.next_float() as f64) < cfg.skip_drop;
        if !is_skip {
            let mut drop_rate = cfg.drop_rate;
            if !cfg.uniform_drop {
                let inv_average_weight = d.tree_weight.len() as f64 / d.sum_weight;
                if cfg.max_drop > 0 {
                    drop_rate = cpp_min(drop_rate, cfg.max_drop as f64 * inv_average_weight / d.sum_weight);
                }
                for i in 0..st.iter {
                    if (d.random_for_drop.next_float() as f64) < drop_rate * d.tree_weight[i] * inv_average_weight {
                        d.drop_index.push(init + i);
                        if d.drop_index.len() >= max_drop {
                            break;
                        }
                    }
                }
            } else {
                if cfg.max_drop > 0 {
                    drop_rate = cpp_min(drop_rate, cfg.max_drop as f64 / st.iter as f64);
                }
                for i in 0..st.iter {
                    if (d.random_for_drop.next_float() as f64) < drop_rate {
                        d.drop_index.push(init + i);
                        if d.drop_index.len() >= max_drop {
                            break;
                        }
                    }
                }
            }
        }
        let n = st.data.num_data();
        let mut drop = || {
            for &i in &d.drop_index {
                for k in 0..ntpi {
                    let t = &mut models[i * ntpi + k];
                    t.shrink(-1.0);
                    t.add_prediction_to_score(&st.data, &mut st.scores[k * n..(k + 1) * n]);
                }
            }
        };
        match pool {
            Some(p) => p.install(drop),
            None => drop(),
        }
        let num_drop = d.drop_index.len() as f64;
        st.shrinkage_rate = if !cfg.xgboost_dart_mode {
            cfg.learning_rate / (1.0 + num_drop)
        } else if d.drop_index.is_empty() {
            cfg.learning_rate
        } else {
            cfg.learning_rate / (cfg.learning_rate + num_drop)
        };
    }

    /// upstream: `DART::Normalize` (dropped trees end at `k / (k + 1)` of
    /// their weight, or `k / (k + learning_rate)` in xgboost mode).
    fn dart_normalize(&mut self) {
        let Gbdt { config, models, train, pool, num_tree_per_iteration, num_init_iteration, .. } = self;
        let cfg = config.as_ref().expect("training config");
        let st = train.as_mut().expect("training state");
        let d = st.dart.as_mut().expect("dart state");
        let (ntpi, init) = (*num_tree_per_iteration, *num_init_iteration);
        let k = d.drop_index.len() as f64;
        let lr = cfg.learning_rate;
        let (valid_rate, train_rate, weight_den) = if !cfg.xgboost_dart_mode {
            (1.0 / (k + 1.0), -k, k + 1.0)
        } else {
            (st.shrinkage_rate, -k / lr, k + lr)
        };
        let n = st.data.num_data();
        let mut normalize = || {
            for &i in &d.drop_index {
                for c in 0..ntpi {
                    let t = &mut models[i * ntpi + c];
                    t.shrink(valid_rate);
                    for v in st.valid.iter_mut() {
                        let vn = v.data.num_data();
                        t.add_prediction_to_score(&v.data, &mut v.scores[c * vn..(c + 1) * vn]);
                    }
                    t.shrink(train_rate);
                    t.add_prediction_to_score(&st.data, &mut st.scores[c * n..(c + 1) * n]);
                }
                if !cfg.uniform_drop {
                    d.sum_weight -= d.tree_weight[i - init] * (1.0 / weight_den);
                    d.tree_weight[i - init] *= k / weight_den;
                }
            }
        };
        match pool {
            Some(p) => p.install(normalize),
            None => normalize(),
        }
    }

    /// upstream: `Booster::ResetConfig` -> `GBDT::ResetConfig` (and
    /// `DART::ResetConfig`, which reseeds the drop generator).
    fn reset_config(&mut self) {
        let Some(cfg) = self.config.as_ref() else { return };
        if let Some(st) = self.train.as_mut() {
            st.shrinkage_rate = cfg.learning_rate;
            if let Some(d) = st.dart.as_mut() {
                d.random_for_drop = Random::new(cfg.drop_seed);
                d.sum_weight = 0.0;
            }
        }
    }

    /// One random-forest iteration: a tree on the fixed gradients, then the
    /// scores are kept as the average over iterations.
    ///
    /// upstream: `RF::TrainOneIter`.
    fn rf_train_one_iter(&mut self, custom: Option<(&[f32], &[f32])>) -> Result<bool> {
        if custom.is_some() {
            return Err(LgbmError::InvalidParameter("Check failed: (gradients) == (nullptr)".into()));
        }
        let ntpi = self.num_tree_per_iteration;
        let init_iter = self.num_init_iteration;
        {
            let st = self.train.as_mut().unwrap();
            if let Some(s) = st.sampler.as_mut() {
                if s.bagging(st.iter, &mut st.grad, &mut st.hess) {
                    st.learner.set_bagging_data(Some(s.in_bag()));
                }
            }
        }
        for k in 0..ntpi {
            let st = self.train.as_mut().unwrap();
            let n = st.data.num_data();
            let offset = k * n;
            let init_score = st.rf_init_scores.as_ref().expect("RF init scores")[k];
            let mut tree = if st.class_need_train[k] && st.data.num_features() > 0 {
                let (g, h) = (&st.grad[offset..offset + n], &st.hess[offset..offset + n]);
                let learner = &mut st.learner;
                match &self.pool {
                    Some(p) => p.install(|| learner.train(g, h)),
                    None => learner.train(g, h),
                }
            } else {
                Tree::new(2)
            };
            let iters = (st.iter + init_iter) as f64;
            if tree.num_leaves > 1 {
                if let Some(obj) = self.objective.as_ref().filter(|o| o.is_renew_tree_output()) {
                    // residuals against the constant initial score
                    let part = st.learner.partition();
                    let scores = vec![init_score; n];
                    let renew = || -> Vec<Option<f64>> {
                        use rayon::prelude::*;
                        (0..tree.num_leaves)
                            .into_par_iter()
                            .map(|leaf| {
                                let idx = part.indices_on_leaf(leaf);
                                (!idx.is_empty()).then(|| obj.renew_leaf_output(idx, &scores))
                            })
                            .collect()
                    };
                    let outputs = match &self.pool {
                        Some(p) => p.install(renew),
                        None => renew(),
                    };
                    for (leaf, v) in outputs.into_iter().enumerate() {
                        if let Some(v) = v {
                            tree.set_leaf_output(leaf, v);
                        }
                    }
                }
                if init_score.abs() > K_EPSILON {
                    tree.add_bias(init_score);
                }
                multiply_score(st, k, iters);
                update_score(st, &self.pool, &tree, k);
                multiply_score(st, k, 1.0 / (iters + 1.0));
            } else if self.models.len() < ntpi {
                let output = if !st.class_need_train[k] {
                    match self.objective.as_ref() {
                        Some(o) => o.boost_from_score(k),
                        None => init_score,
                    }
                } else {
                    0.0
                };
                tree.as_constant(output, n as i32);
                multiply_score(st, k, iters);
                update_score(st, &self.pool, &tree, k);
                multiply_score(st, k, 1.0 / (iters + 1.0));
            }
            self.models.push(tree);
        }
        self.train.as_mut().unwrap().iter += 1;
        Ok(false)
    }

    /// upstream: `RF::RollbackOneIter`.
    fn rf_rollback_one_iter(&mut self) {
        let ntpi = self.num_tree_per_iteration;
        let st = self.train.as_mut().expect("training state");
        if st.iter == 0 {
            return;
        }
        let iters = st.iter + self.num_init_iteration;
        let cur_iter = iters - 1;
        let n = st.data.num_data();
        for k in 0..ntpi {
            let t = &mut self.models[cur_iter * ntpi + k];
            t.shrink(-1.0);
            multiply_score(st, k, iters as f64);
            t.add_prediction_to_score(&st.data, &mut st.scores[k * n..(k + 1) * n]);
            for v in st.valid.iter_mut() {
                let vn = v.data.num_data();
                t.add_prediction_to_score(&v.data, &mut v.scores[k * vn..(k + 1) * vn]);
            }
            // upstream: 1.0f / (iter_ + num_init_iteration_ - 1), a float division
            multiply_score(st, k, (1.0f32 / cur_iter as f32) as f64);
        }
        self.models.truncate(self.models.len() - ntpi);
        st.iter -= 1;
    }

    /// One boosting iteration. With `custom = Some((grad, hess))` the given
    /// gradients are used instead of the objective's. Returns `true` when
    /// training cannot continue (no tree could split).
    pub fn train_one_iter(&mut self, custom: Option<(&[f32], &[f32])>) -> Result<bool> {
        if self.train.is_none() {
            return Err(LgbmError::InvalidData("booster has no training data".into()));
        }
        if self.is_rf {
            return self.rf_train_one_iter(custom);
        }
        let ntpi = self.num_tree_per_iteration;
        let n = self.train.as_ref().unwrap().data.num_data();
        let by_query = self.config.as_ref().is_some_and(|c| c.bagging_by_query);
        if let Some(d) = self.train.as_mut().unwrap().dart.as_mut() {
            d.is_update_score_cur_iter = false;
        }
        let mut init_scores = vec![0.0; ntpi];
        match custom {
            None => {
                if self.objective.is_none() {
                    return Err(LgbmError::InvalidParameter(
                        "no objective function; provide gradients and hessians".into(),
                    ));
                }
                for (k, s) in init_scores.iter_mut().enumerate() {
                    *s = self.boost_from_average(k);
                }
                if by_query {
                    let st = self.train.as_mut().unwrap();
                    if let Some(s) = st.sampler.as_mut() {
                        if s.bagging(st.iter, &mut st.grad, &mut st.hess) {
                            st.learner.set_bagging_data(Some(s.in_bag()));
                        }
                    }
                }
                self.training_score();
                let st = self.train.as_mut().unwrap();
                let obj = self.objective.as_ref().unwrap();
                let sampled = st.sampler.as_ref().and_then(|s| s.sampled_queries());
                let view = ScoreView { scores: &st.scores, num_data: n, num_outputs: ntpi };
                let pool = self.pool.clone();
                let mut compute = || match sampled {
                    Some(q) => obj.gradients_with_sampled_queries(view, q, &mut st.grad, &mut st.hess),
                    None => obj.gradients(view, &mut st.grad, &mut st.hess),
                };
                match pool {
                    Some(p) => p.install(compute),
                    None => compute(),
                }
            }
            Some((g, h)) => {
                if self.objective.is_some() {
                    return Err(LgbmError::InvalidParameter(
                        "custom gradients require objective=none/custom".into(),
                    ));
                }
                if by_query && self.train.as_ref().unwrap().sampler.is_some() {
                    return Err(LgbmError::Unsupported(
                        "bagging_by_query with custom gradients: upstream skips bagging but still \
                         updates the scores of rows from its never-filled bagging buffer"
                            .into(),
                    ));
                }
                if g.len() != n * ntpi || h.len() != n * ntpi {
                    return Err(LgbmError::InvalidData(format!(
                        "gradient/hessian length must be {} (got {} and {})",
                        n * ntpi,
                        g.len(),
                        h.len()
                    )));
                }
                let st = self.train.as_mut().unwrap();
                st.grad.copy_from_slice(g);
                st.hess.copy_from_slice(h);
            }
        }

        // upstream: data_sample_strategy_->Bagging
        if !by_query {
            let st = self.train.as_mut().unwrap();
            if let Some(s) = st.sampler.as_mut() {
                if s.bagging(st.iter, &mut st.grad, &mut st.hess) {
                    st.learner.set_bagging_data(Some(s.in_bag()));
                }
            }
        }

        let cfg = self.config.clone().expect("training config");
        let mut should_continue = false;
        for k in 0..ntpi {
            let offset = k * n;
            let need_train = {
                let st = self.train.as_ref().unwrap();
                st.class_need_train[k] && st.data.num_features() > 0
            };
            let mut tree = if need_train {
                let st = self.train.as_mut().unwrap();
                let (g, h) = (&st.grad[offset..offset + n], &st.hess[offset..offset + n]);
                let learner = &mut st.learner;
                let tree = match &self.pool {
                    Some(p) => p.install(|| learner.train(g, h)),
                    None => learner.train(g, h),
                };
                if let Some(s) = st.sampler.as_ref().filter(|s| s.by_query_subset()) {
                    for (i, &row) in s.in_bag().iter().enumerate() {
                        st.grad[offset + i] = st.grad[offset + row as usize];
                        st.hess[offset + i] = st.hess[offset + row as usize];
                    }
                }
                tree
            } else {
                Tree::new(2)
            };

            if tree.num_leaves > 1 {
                should_continue = true;
                if let Some(obj) = self.objective.as_ref().filter(|o| o.is_renew_tree_output()) {
                    // upstream: SerialTreeLearner::RenewTreeOutput, before shrinkage
                    let st = self.train.as_ref().unwrap();
                    let part = st.learner.partition();
                    let scores = &st.scores[offset..offset + n];
                    let renew = || -> Vec<Option<f64>> {
                        use rayon::prelude::*;
                        (0..tree.num_leaves)
                            .into_par_iter()
                            .map(|leaf| {
                                let idx = part.indices_on_leaf(leaf);
                                (!idx.is_empty()).then(|| obj.renew_leaf_output(idx, scores))
                            })
                            .collect()
                    };
                    let outputs = match &self.pool {
                        Some(p) => p.install(renew),
                        None => renew(),
                    };
                    for (leaf, v) in outputs.into_iter().enumerate() {
                        if let Some(v) = v {
                            tree.set_leaf_output(leaf, v);
                        }
                    }
                }
                let st = self.train.as_mut().unwrap();
                tree.shrink(st.shrinkage_rate);
                update_score(st, &self.pool, &tree, k);
                if init_scores[k].abs() > K_EPSILON {
                    tree.add_bias(init_scores[k]);
                }
            } else if self.models.len() < ntpi {
                let has_init = self.train.as_ref().unwrap().has_init_score;
                if self.objective.is_some() && !cfg.boost_from_average && !has_init {
                    init_scores[k] = self.objective.as_ref().unwrap().boost_from_score(k);
                    let st = self.train.as_mut().unwrap();
                    add_const(&mut st.scores, k, n, init_scores[k]);
                    for v in st.valid.iter_mut() {
                        let vn = v.data.num_data();
                        add_const(&mut v.scores, k, vn, init_scores[k]);
                    }
                }
                tree.as_constant(init_scores[k], n as i32);
            } else {
                tree.as_constant(0.0, n as i32);
            }
            self.models.push(tree);
        }

        if !should_continue {
            if self.models.len() > ntpi {
                for _ in 0..ntpi {
                    self.models.pop();
                }
            }
            return Ok(true);
        }
        self.train.as_mut().unwrap().iter += 1;
        if self.train.as_ref().unwrap().dart.is_some() {
            self.dart_normalize();
            let st = self.train.as_mut().unwrap();
            let rate = st.shrinkage_rate;
            let d = st.dart.as_mut().unwrap();
            if !cfg.uniform_drop {
                d.tree_weight.push(rate);
                d.sum_weight += rate;
            }
        }
        Ok(false)
    }

    /// Refit every tree's leaf outputs on the training data, keeping the
    /// structure. `leaf_preds` is row-major `nrow x ncol`: the leaf of every
    /// training row in every tree (as from `predict(pred_leaf=True)`).
    ///
    /// upstream: `GBDT::RefitTree`.
    pub fn refit_tree(&mut self, leaf_preds: &[i32], nrow: usize, ncol: usize) -> Result<()> {
        let check = |ok: bool, what: &str| {
            if ok { Ok(()) } else { Err(LgbmError::InvalidData(format!("Check failed: {what}"))) }
        };
        check(nrow * ncol > 0, "(nrow * ncol) > (0)")?;
        let st = self.train.as_ref().ok_or_else(|| LgbmError::InvalidData("booster has no training data".into()))?;
        let n = st.data.num_data();
        check(n == nrow, "(static_cast<size_t>(num_data_)) == (nrow)")?;
        check(self.models.len() == ncol, "(models_.size()) == (ncol)")?;
        check(leaf_preds.len() == nrow * ncol, "(leaf_preds.size()) == (nrow * ncol)")?;
        if self.objective.is_none() {
            return Err(LgbmError::InvalidParameter("No objective function provided".into()));
        }
        let cfg = self.config.clone().expect("training config");
        if cfg.bagging_by_query {
            return Err(LgbmError::Unsupported("Booster.refit() with bagging_by_query".into()));
        }
        let ntpi = self.num_tree_per_iteration;
        let num_iterations = self.models.len() / ntpi;
        let mut leaf_pred = vec![0i32; n];
        for iter in 0..num_iterations {
            // upstream GBDT::Boosting; RF::Boosting with trees present starts from zero scores
            {
                let st = self.train.as_mut().unwrap();
                let obj = self.objective.as_ref().unwrap();
                let zeros;
                let scores = if self.is_rf {
                    zeros = vec![0.0; n * ntpi];
                    &zeros
                } else {
                    &st.scores
                };
                let view = ScoreView { scores, num_data: n, num_outputs: ntpi };
                let mut compute = || obj.gradients(view, &mut st.grad, &mut st.hess);
                match &self.pool {
                    Some(p) => p.install(compute),
                    None => compute(),
                }
            }
            for k in 0..ntpi {
                let mi = iter * ntpi + k;
                let num_leaves = self.models[mi].num_leaves;
                for (i, l) in leaf_pred.iter_mut().enumerate() {
                    *l = leaf_preds[i * ncol + mi];
                    check(
                        *l >= 0 && (*l as usize) < num_leaves,
                        "(leaf_pred[i]) < (models_[model_index]->num_leaves())",
                    )?;
                }
                let st = self.train.as_mut().unwrap();
                let offset = k * n;
                let tree = st.learner.fit_by_existing_tree(
                    &self.models[mi],
                    &leaf_pred,
                    &st.grad[offset..offset + n],
                    &st.hess[offset..offset + n],
                    cfg.refit_decay_rate,
                );
                for (s, &l) in st.scores[offset..offset + n].iter_mut().zip(&leaf_pred) {
                    *s += tree.leaf_value[l as usize];
                }
                self.models[mi] = tree;
            }
        }
        Ok(())
    }

    /// Remove the last iteration's trees and their score contributions.
    pub fn rollback_one_iter(&mut self) -> Result<()> {
        let ntpi = self.num_tree_per_iteration;
        if self.is_rf && self.train.is_some() {
            self.rf_rollback_one_iter();
            return Ok(());
        }
        let st = self.train.as_mut().ok_or_else(|| LgbmError::InvalidData("not training".into()))?;
        if st.iter == 0 || self.models.len() < ntpi {
            return Ok(());
        }
        let n = st.data.num_data();
        let start = self.models.len() - ntpi;
        for k in 0..ntpi {
            let mut t = self.models[start + k].clone();
            t.shrink(-1.0);
            t.add_prediction_to_score(&st.data, &mut st.scores[k * n..(k + 1) * n]);
            for v in st.valid.iter_mut() {
                let vn = v.data.num_data();
                t.add_prediction_to_score(&v.data, &mut v.scores[k * vn..(k + 1) * vn]);
            }
        }
        self.models.truncate(start);
        st.iter -= 1;
        Ok(())
    }

    /// Evaluate the built-in metrics on the training data.
    pub fn eval_train(&self) -> Vec<EvalResult> {
        let Some(st) = self.train.as_ref() else { return Vec::new() };
        let metrics: Vec<Metric> = if st.metrics.is_empty() {
            self.config
                .as_ref()
                .and_then(|c| Self::make_metrics(c, &st.data).ok())
                .unwrap_or_default()
        } else {
            st.metrics.clone()
        };
        let mut out = Vec::new();
        for m in &metrics {
            self.push_evals(&mut out, "training", m, &st.scores);
        }
        out
    }

    fn push_evals(&self, out: &mut Vec<EvalResult>, data_name: &str, m: &Metric, scores: &[f64]) {
        for (name, v) in m.names().iter().zip(m.eval(scores, self.objective.as_ref())) {
            out.push((data_name.to_string(), name.clone(), v, m.kind.higher_better()));
        }
    }

    /// Evaluate the built-in metrics on every validation set.
    pub fn eval_valid(&self) -> Vec<EvalResult> {
        self.eval_valid_marked().into_iter().map(|(e, _)| e).collect()
    }

    /// [`eval_valid`](Self::eval_valid), each result flagged when it comes
    /// from its set's first metric.
    fn eval_valid_marked(&self) -> Vec<(EvalResult, bool)> {
        let Some(st) = self.train.as_ref() else { return Vec::new() };
        let mut out = Vec::new();
        for v in &st.valid {
            for (j, m) in v.metrics.iter().enumerate() {
                let mut evals = Vec::new();
                self.push_evals(&mut evals, &v.name, m, &v.scores);
                out.extend(evals.into_iter().map(|e| (e, j == 0)));
            }
        }
        out
    }

    pub fn num_valid(&self) -> usize {
        self.train.as_ref().map_or(0, |s| s.valid.len())
    }

    /// Raw scores of validation set `i` (class-major).
    pub fn valid_scores(&self, i: usize) -> Option<&[f64]> {
        self.train.as_ref().and_then(|s| s.valid.get(i)).map(|v| v.scores.as_slice())
    }

    /// Train up to `num_iterations` with optional early stopping on the
    /// validation sets' metrics (upstream `GBDT::Train` semantics with
    /// `early_stopping_round`). Returns the best iteration (1-based) when
    /// early stopping triggered, else 0.
    pub fn train(&mut self) -> Result<usize> {
        let cfg = self.config.clone().ok_or_else(|| LgbmError::Internal("no config".into()))?;
        // upstream: DART::EvalAndCheckEarlyStopping never stops
        let rounds = if cfg.boosting == "dart" { 0 } else { cfg.early_stopping_round.max(0) as usize };
        let mut best: Vec<f64> = Vec::new();
        let mut best_iter: Vec<usize> = Vec::new();
        for _ in 0..cfg.num_iterations.max(0) {
            if self.train_one_iter(None)? {
                break;
            }
            if rounds == 0 {
                continue;
            }
            let evals = self.eval_valid_marked();
            if evals.is_empty() {
                continue;
            }
            let iter = self.current_iteration();
            if best.is_empty() {
                best = evals.iter().map(|(e, _)| if e.3 { f64::NEG_INFINITY } else { f64::INFINITY }).collect();
                best_iter = vec![0; evals.len()];
            }
            for (j, (e, first)) in evals.iter().enumerate() {
                if cfg.first_metric_only && !first {
                    continue;
                }
                let improved = if e.3 {
                    e.2 > best[j] + cfg.early_stopping_min_delta
                } else {
                    e.2 < best[j] - cfg.early_stopping_min_delta
                };
                if improved {
                    best[j] = e.2;
                    best_iter[j] = iter;
                } else if iter - best_iter[j] >= rounds {
                    let keep = best_iter[j];
                    while self.current_iteration() > keep {
                        self.rollback_one_iter()?;
                    }
                    return Ok(keep);
                }
            }
        }
        Ok(0)
    }

    /// upstream: `GBDT::FeatureImportance` (0 = split count, 1 = total gain).
    pub fn feature_importance(&self, num_iteration: i32, importance_type: i32) -> Result<Vec<f64>> {
        let mut n_models = self.models.len();
        if num_iteration > 0 {
            n_models = n_models.min(num_iteration as usize * self.num_tree_per_iteration);
        }
        let mut imp = vec![0.0f64; self.num_feature()];
        for t in &self.models[..n_models] {
            for s in 0..t.num_leaves.saturating_sub(1) {
                if t.split_gain[s] > 0.0 {
                    let f = t.split_feature[s] as usize;
                    match importance_type {
                        0 => imp[f] += 1.0,
                        1 => imp[f] += t.split_gain[s] as f64,
                        _ => {
                            return Err(LgbmError::InvalidParameter(
                                "Unknown importance type: only support split=0 and gain=1".into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(imp)
    }

    /// Parameters block of a loaded model (`[name: value]` lines).
    pub fn loaded_parameters(&self) -> Option<&str> {
        self.loaded_parameters.as_deref()
    }

    /// Switch to caller-supplied gradients (upstream: `reset_parameter({"objective": "none"})`).
    /// Scores accumulated so far are kept.
    pub fn clear_objective(&mut self) {
        self.objective = None;
        if let Some(c) = self.config.as_mut() {
            c.objective = "custom".into();
        }
        self.reset_config();
    }

    /// upstream: `Booster::ResetConfig` with only `learning_rate` changed
    /// (`GBDT::ResetConfig` sets `shrinkage_rate_`); applies to later iterations.
    pub fn set_learning_rate(&mut self, value: &str) -> Result<()> {
        let Some(c) = self.config.as_mut() else {
            return Err(LgbmError::Unsupported("reset_parameter on a loaded model".into()));
        };
        c.learning_rate = crate::config::parse_checked_double("learning_rate", value)?;
        c.explicit.insert("learning_rate".into(), value.trim().to_string());
        self.reset_config();
        Ok(())
    }

    /// Drop the training state (datasets, scores, learner) and keep the model.
    pub fn free_training_state(&mut self) {
        self.train = None;
    }

    /// Set the thread count used by prediction.
    pub fn set_num_threads(&mut self, n: i32) -> Result<()> {
        self.pool = build_pool(n)?;
        Ok(())
    }
}

/// upstream: `GBDT::UpdateScore` (in-bag rows from the learner's partition,
/// out-of-bag rows and validation sets by prediction).
fn update_score(st: &mut TrainState, pool: &Option<Arc<rayon::ThreadPool>>, tree: &Tree, k: usize) {
    let n = st.data.num_data();
    let offset = k * n;
    let mut update = || {
        st.learner.add_leaf_outputs(&tree.leaf_value, &mut st.scores[offset..offset + n]);
        if let Some(s) = st.sampler.as_ref().filter(|s| s.bag_cnt() < n) {
            tree.add_prediction_to_score_rows(&st.data, s.out_of_bag(), &mut st.scores[offset..offset + n]);
        }
        for v in st.valid.iter_mut() {
            let vn = v.data.num_data();
            tree.add_prediction_to_score(&v.data, &mut v.scores[k * vn..(k + 1) * vn]);
        }
    };
    match pool {
        Some(p) => p.install(update),
        None => update(),
    }
}

/// upstream: `RF::MultiplyScore` (training and validation scores of class `k`).
fn multiply_score(st: &mut TrainState, k: usize, val: f64) {
    let n = st.data.num_data();
    for s in &mut st.scores[k * n..(k + 1) * n] {
        *s *= val;
    }
    for v in st.valid.iter_mut() {
        let vn = v.data.num_data();
        for s in &mut v.scores[k * vn..(k + 1) * vn] {
            *s *= val;
        }
    }
}

fn add_const(scores: &mut [f64], k: usize, n: usize, v: f64) {
    for s in &mut scores[k * n..(k + 1) * n] {
        *s += v;
    }
}
