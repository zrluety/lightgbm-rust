//! Parameter parsing, alias resolution, validation, and support gating.
//!
//! The parameter table (names, aliases, defaults, checks) is generated from
//! upstream `include/LightGBM/config.h` by `scripts/extract_params.py` and
//! embedded here, so alias handling cannot drift from the pinned release.
//!
//! Every upstream parameter falls in one of three classes:
//! * **honored**: implemented and affects training/prediction;
//! * **no-effect**: accepted because it cannot change results in this
//!   implementation (e.g. `num_threads`, `force_row_wise`);
//! * **gated**: not implemented yet. Supplying a non-default value returns
//!   [`LgbmError::Unsupported`] instead of being silently ignored.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use serde::Deserialize;

use crate::error::{LgbmError, Result};
use crate::random::Random;

const PARAMS_JSON: &str = include_str!("../data/params.json");

#[derive(Debug, Deserialize)]
struct ParamTable {
    parameters: Vec<ParamSpec>,
}

/// One upstream parameter as described in `config.h`.
#[derive(Debug, Deserialize, Clone)]
pub struct ParamSpec {
    pub name: String,
    pub section: String,
    pub cpp_type: String,
    pub cpp_default: Option<String>,
    #[serde(default)]
    pub doc_default: Option<String>,
    pub aliases: Vec<String>,
    pub checks: Vec<String>,
    pub flags: Vec<String>,
}

struct Registry {
    specs: Vec<ParamSpec>,
    by_name: HashMap<String, usize>,
    alias_to_name: HashMap<String, String>,
}

fn registry() -> &'static Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| {
        let table: ParamTable =
            serde_json::from_str(PARAMS_JSON).expect("embedded params.json is valid");
        let mut by_name = HashMap::new();
        let mut alias_to_name = HashMap::new();
        for (i, p) in table.parameters.iter().enumerate() {
            by_name.insert(p.name.clone(), i);
            for a in &p.aliases {
                alias_to_name.insert(a.clone(), p.name.clone());
            }
        }
        Registry { specs: table.parameters, by_name, alias_to_name }
    })
}

/// All upstream parameter specs, in `config.h` order.
pub fn param_specs() -> &'static [ParamSpec] {
    &registry().specs
}

/// Resolve an alias to its canonical upstream name (identity for canonical names).
pub fn canonical_name(key: &str) -> Option<&'static str> {
    let reg = registry();
    if let Some(&i) = reg.by_name.get(key) {
        return Some(reg.specs[i].name.as_str());
    }
    reg.alias_to_name
        .get(key)
        .and_then(|n| reg.by_name.get(n))
        .map(|&i| reg.specs[i].name.as_str())
}

/// Parameters implemented by this engine.
const HONORED: &[&str] = &[
    "objective", "boosting", "num_iterations", "learning_rate", "num_leaves", "seed",
    "max_depth", "min_data_in_leaf", "min_sum_hessian_in_leaf", "early_stopping_round",
    "early_stopping_min_delta", "first_metric_only", "max_delta_step", "lambda_l1",
    "lambda_l2", "min_gain_to_split", "path_smooth", "max_bin", "min_data_in_bin",
    "bin_construct_sample_cnt", "data_random_seed", "use_missing", "zero_as_missing",
    "feature_pre_filter", "is_unbalance", "scale_pos_weight", "sigmoid",
    "boost_from_average", "reg_sqrt", "num_class", "metric", "saved_feature_importance_type",
    "start_iteration_predict", "num_iteration_predict", "predict_raw_score",
    "predict_leaf_index", "predict_disable_shape_check", "is_provide_training_metric",
    "force_col_wise", "force_row_wise", "alpha", "fair_c", "poisson_max_delta_step",
    "tweedie_variance_power", "multi_error_top_k",
];

/// Parameters that cannot change results here (threading, layout, logging,
/// CLI-only I/O, or sampling knobs that are inert at their other defaults).
const NO_EFFECT: &[&str] = &[
    "num_threads", "deterministic", "histogram_pool_size",
    "verbosity", "is_enable_sparse", "enable_bundle", "metric_freq", "snapshot_freq",
    "output_model", "input_model", "output_result", "data", "valid", "config", "task",
    "header", "label_column", "weight_column", "group_column", "ignore_column",
    "save_binary", "precise_float_parser", "two_round", "pre_partition",
    // Seeds only matter when the corresponding sampler is enabled (gated separately).
    "bagging_seed", "feature_fraction_seed", "extra_seed", "drop_seed", "objective_seed",
    "data_sample_strategy",
    // Only read by non-CPU devices / multi-machine learners, which are gated via
    // device_type / num_machines / tree_learner.
    "gpu_platform_id", "gpu_device_id", "gpu_device_id_list", "gpu_use_dp", "num_gpu",
    "local_listen_port", "time_out", "machine_list_filename", "machines",
    // Objective-specific knobs of objectives that are gated via `objective`.
    "lambdarank_truncation_level", "lambdarank_norm", "label_gain",
    "lambdarank_position_bias_regularization", "eval_at",
    "auc_mu_weights",
    // DART/GOSS knobs are inert unless boosting/data_sample_strategy selects them.
    "drop_rate", "max_drop", "skip_drop", "xgboost_dart_mode", "uniform_drop",
    "top_rate", "other_rate",
    // Categorical knobs are inert without categorical features (gated separately).
    "min_data_per_group", "max_cat_threshold", "cat_l2", "cat_smooth", "max_cat_to_onehot",
    // Quantization sub-options are inert unless use_quantized_grad (gated).
    "num_grad_quant_bins", "quant_train_renew_leaf", "stochastic_rounding",
    "monotone_constraints_method", "monotone_penalty", "top_k", "refit_decay_rate",
    "linear_lambda", "pred_early_stop_freq", "pred_early_stop_margin",
    "convert_model_language", "convert_model", "parser_config_file",
];

pub const SUPPORTED_OBJECTIVES: &[&str] = &[
    "regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie",
    "binary", "multiclass", "multiclassova",
];
pub const SUPPORTED_METRICS: &[&str] = &[
    "l2", "rmse", "l1", "quantile", "huber", "fair", "poisson", "mape", "gamma", "gamma_deviance",
    "tweedie", "binary_logloss", "binary_error", "auc", "multi_logloss", "multi_error",
];

/// upstream: include/LightGBM/config.h `ParseObjectiveAlias`.
pub fn parse_objective_alias(t: &str) -> String {
    match t {
        "regression" | "regression_l2" | "mean_squared_error" | "mse" | "l2" | "l2_root"
        | "root_mean_squared_error" | "rmse" => "regression",
        "regression_l1" | "mean_absolute_error" | "l1" | "mae" => "regression_l1",
        "multiclass" | "softmax" => "multiclass",
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => "multiclassova",
        "xentropy" | "cross_entropy" => "cross_entropy",
        "xentlambda" | "cross_entropy_lambda" => "cross_entropy_lambda",
        "mean_absolute_percentage_error" | "mape" => "mape",
        "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => "rank_xendcg",
        "none" | "null" | "custom" | "na" => "custom",
        other => other,
    }
    .to_string()
}

/// upstream: include/LightGBM/config.h `ParseMetricAlias`.
pub fn parse_metric_alias(t: &str) -> String {
    match t {
        "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse" => "l2",
        "l2_root" | "root_mean_squared_error" | "rmse" => "rmse",
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => "l1",
        "binary_logloss" | "binary" => "binary_logloss",
        "ndcg" | "lambdarank" | "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart"
        | "xendcg_mart" => "ndcg",
        "map" | "mean_average_precision" => "map",
        "multi_logloss" | "multiclass" | "softmax" | "multiclassova" | "multiclass_ova" | "ova"
        | "ovr" => "multi_logloss",
        "xentropy" | "cross_entropy" => "cross_entropy",
        "xentlambda" | "cross_entropy_lambda" => "cross_entropy_lambda",
        "kldiv" | "kullback_leibler" => "kullback_leibler",
        "mean_absolute_percentage_error" | "mape" => "mape",
        "none" | "null" | "custom" | "na" => "custom",
        other => other,
    }
    .to_string()
}

/// upstream: src/io/config.cpp `ParseMetrics` (split on ',', alias, dedupe in order).
fn parse_metrics(value: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for m in value.split(',') {
        let t = parse_metric_alias(m);
        if !out.contains(&t) {
            out.push(t);
        }
    }
    out
}

fn parse_bool(key: &str, v: &str) -> Result<bool> {
    // upstream: Config::GetBool accepts only "true"/"+" and "false"/"-" (case-insensitive).
    match v.to_ascii_lowercase().as_str() {
        "true" | "+" => Ok(true),
        "false" | "-" => Ok(false),
        _ => Err(LgbmError::InvalidParameter(format!(
            "Parameter {key} should be \"true\"/\"+\" or \"false\"/\"-\", got \"{v}\""
        ))),
    }
}

fn parse_int(key: &str, v: &str) -> Result<i32> {
    v.trim().parse::<i32>().map_err(|_| {
        LgbmError::InvalidParameter(format!("Parameter {key} should be of type int, got \"{v}\""))
    })
}

fn parse_double(key: &str, v: &str) -> Result<f64> {
    let t = v.trim();
    let lower = t.to_ascii_lowercase();
    let parsed = match lower.as_str() {
        "inf" | "+inf" | "infinity" => Ok(f64::INFINITY),
        "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
        "nan" | "na" | "null" => Ok(f64::NAN),
        _ => t.parse::<f64>(),
    };
    parsed.map_err(|_| {
        LgbmError::InvalidParameter(format!("Parameter {key} should be of type double, got \"{v}\""))
    })
}

fn check_value(key: &str, value: f64, check: &str) -> Result<()> {
    let c = check.replace(' ', "");
    let (op, rhs) = if let Some(r) = c.strip_prefix(">=") {
        (">=", r)
    } else if let Some(r) = c.strip_prefix("<=") {
        ("<=", r)
    } else if let Some(r) = c.strip_prefix('>') {
        (">", r)
    } else if let Some(r) = c.strip_prefix('<') {
        ("<", r)
    } else {
        return Ok(());
    };
    let Ok(bound) = rhs.parse::<f64>() else { return Ok(()) };
    let ok = match op {
        ">=" => value >= bound,
        "<=" => value <= bound,
        ">" => value > bound,
        "<" => value < bound,
        _ => true,
    };
    if ok {
        Ok(())
    } else {
        // Same wording as upstream CHECK macros, e.g. "Check failed: (num_leaves) > (1)".
        Err(LgbmError::InvalidParameter(format!("Check failed: ({key}) {op} ({rhs})")))
    }
}

fn spec_default_string(spec: &ParamSpec) -> String {
    let raw = spec
        .doc_default
        .clone()
        .or_else(|| spec.cpp_default.clone())
        .unwrap_or_default();
    raw.trim_matches('"').to_string()
}

fn values_equal_to_default(spec: &ParamSpec, value: &str) -> bool {
    let d = spec_default_string(spec);
    let t = spec.cpp_type.as_str();
    match t {
        "bool" => parse_bool(&spec.name, value).ok() == parse_bool(&spec.name, &d).ok(),
        "int" | "double" => match (value.trim().parse::<f64>(), d.parse::<f64>()) {
            (Ok(a), Ok(b)) => a == b,
            _ => value.trim() == d,
        },
        _ => value.trim().is_empty() && (d.is_empty() || d.contains("...")) || value.trim() == d,
    }
}

/// Validated configuration. Field names follow upstream `Config`.
#[derive(Debug, Clone)]
pub struct Config {
    pub objective: String,
    pub boosting: String,
    pub metric: Vec<String>,
    pub num_iterations: i32,
    pub learning_rate: f64,
    pub num_leaves: i32,
    pub num_threads: i32,
    pub seed: i32,
    pub deterministic: bool,
    /// Feature-parallel histograms; results do not depend on `num_threads`.
    pub force_col_wise: bool,
    /// Row-block histograms (the default here); summation order matches
    /// upstream's row-wise mode for the same `num_threads`.
    pub force_row_wise: bool,
    pub max_depth: i32,
    pub min_data_in_leaf: i32,
    pub min_sum_hessian_in_leaf: f64,
    pub early_stopping_round: i32,
    pub early_stopping_min_delta: f64,
    pub first_metric_only: bool,
    pub max_delta_step: f64,
    pub lambda_l1: f64,
    pub lambda_l2: f64,
    pub min_gain_to_split: f64,
    pub path_smooth: f64,
    pub verbosity: i32,
    pub max_bin: i32,
    pub min_data_in_bin: i32,
    pub bin_construct_sample_cnt: i32,
    pub data_random_seed: i32,
    /// Sub-seeds of samplers that are not implemented yet; kept so that
    /// `seed` derivation and the saved parameters match upstream.
    pub bagging_seed: i32,
    pub drop_seed: i32,
    pub feature_fraction_seed: i32,
    pub objective_seed: i32,
    pub extra_seed: i32,
    pub use_missing: bool,
    pub zero_as_missing: bool,
    pub feature_pre_filter: bool,
    pub is_unbalance: bool,
    pub scale_pos_weight: f64,
    pub sigmoid: f64,
    pub boost_from_average: bool,
    pub reg_sqrt: bool,
    /// Huber delta and quantile level.
    pub alpha: f64,
    pub fair_c: f64,
    pub poisson_max_delta_step: f64,
    pub tweedie_variance_power: f64,
    pub num_class: i32,
    pub multi_error_top_k: i32,
    pub saved_feature_importance_type: i32,
    pub is_provide_training_metric: bool,
    /// Canonical key -> value string as supplied (after alias resolution).
    pub explicit: BTreeMap<String, String>,
    /// Non-fatal diagnostics (unknown keys, duplicate aliases), mirroring upstream warnings.
    pub warnings: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            objective: "regression".into(),
            boosting: "gbdt".into(),
            metric: vec!["l2".into()],
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            num_threads: 0,
            seed: 0,
            deterministic: false,
            force_col_wise: false,
            force_row_wise: false,
            max_depth: -1,
            min_data_in_leaf: 20,
            min_sum_hessian_in_leaf: 1e-3,
            early_stopping_round: 0,
            early_stopping_min_delta: 0.0,
            first_metric_only: false,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            verbosity: 1,
            max_bin: 255,
            min_data_in_bin: 3,
            bin_construct_sample_cnt: 200_000,
            data_random_seed: 1,
            bagging_seed: 3,
            drop_seed: 4,
            feature_fraction_seed: 2,
            objective_seed: 5,
            extra_seed: 6,
            use_missing: true,
            zero_as_missing: false,
            feature_pre_filter: true,
            is_unbalance: false,
            scale_pos_weight: 1.0,
            sigmoid: 1.0,
            boost_from_average: true,
            reg_sqrt: false,
            alpha: 0.9,
            fair_c: 1.0,
            poisson_max_delta_step: 0.7,
            tweedie_variance_power: 1.5,
            num_class: 1,
            multi_error_top_k: 1,
            saved_feature_importance_type: 0,
            is_provide_training_metric: false,
            explicit: BTreeMap::new(),
            warnings: Vec::new(),
        }
    }
}

impl Config {
    /// Build a config from key/value pairs (keys may be aliases).
    ///
    /// Mirrors upstream `Config::Str2Map` (`KeepFirstValues` +
    /// `ParameterAlias::KeyAliasTransform`) and `Config::Set`: the first value
    /// of a repeated key wins; a canonical name wins over its aliases; among
    /// several aliases the shortest (then lexicographically smallest) wins;
    /// `seed` derives the sub-seeds unless they are given explicitly.
    /// Warnings use upstream's wording.
    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut cfg = Config::default();
        let unquote = |s: &str| s.trim().trim_matches(|c| c == '"' || c == '\'').to_string();

        // KeepFirstValues
        let mut first: Vec<(String, String)> = Vec::new();
        for (k, v) in pairs {
            let (key, val) = (unquote(k.as_ref()), unquote(v.as_ref()));
            if key.is_empty() {
                continue;
            }
            match first.iter().find(|(fk, _)| *fk == key) {
                Some((_, v0)) => cfg.warnings.push(format!(
                    "{key} is set={v0}, {key}={val} will be ignored. Current value: {key}={v0}"
                )),
                None => first.push((key, val)),
            }
        }

        // KeyAliasTransform
        let sort_alias = |x: &str, y: &str| x.len() < y.len() || (x.len() == y.len() && x < y);
        let mut explicit: BTreeMap<String, String> = BTreeMap::new();
        let mut chosen_alias: BTreeMap<String, (String, String)> = BTreeMap::new();
        for (key, val) in &first {
            match canonical_name(key) {
                None => cfg.warnings.push(format!("Unknown parameter: {key}")),
                Some(name) if name == key => {
                    explicit.insert(key.clone(), val.clone());
                }
                Some(name) => match chosen_alias.get(name) {
                    None => {
                        chosen_alias.insert(name.to_string(), (key.clone(), val.clone()));
                    }
                    Some((a, av)) => {
                        if sort_alias(a, key) {
                            cfg.warnings.push(format!(
                                "{name} is set with {a}={av}, {key}={val} will be ignored. Current value: {name}={av}"
                            ));
                        } else {
                            cfg.warnings.push(format!(
                                "{name} is set with {a}={av}, will be overridden by {key}={val}. Current value: {name}={val}"
                            ));
                            chosen_alias.insert(name.to_string(), (key.clone(), val.clone()));
                        }
                    }
                },
            }
        }
        for (name, (alias, av)) in chosen_alias {
            match explicit.get(&name) {
                Some(cv) => cfg.warnings.push(format!(
                    "{name} is set={cv}, {alias}={av} will be ignored. Current value: {name}={cv}"
                )),
                None => {
                    explicit.insert(name, av);
                }
            }
        }
        cfg.apply(&explicit)?;
        cfg.explicit = explicit;
        Ok(cfg)
    }

    fn apply(&mut self, p: &BTreeMap<String, String>) -> Result<()> {
        let reg = registry();
        for (name, value) in p {
            let spec = &reg.specs[reg.by_name[name]];
            if spec.cpp_type == "int" || spec.cpp_type == "double" {
                let x = if spec.cpp_type == "int" {
                    parse_int(name, value)? as f64
                } else {
                    parse_double(name, value)?
                };
                for c in &spec.checks {
                    check_value(name, x, c)?;
                }
            } else if spec.cpp_type == "bool" {
                parse_bool(name, value)?;
            }
            let honored = HONORED.contains(&name.as_str());
            let no_effect = NO_EFFECT.contains(&name.as_str());
            if !honored && !no_effect && !values_equal_to_default(spec, value) {
                return Err(LgbmError::Unsupported(format!(
                    "parameter `{name}={value}` (only the upstream default is accepted)"
                )));
            }
        }

        if let Some(v) = p.get("seed") {
            // upstream: src/io/config.cpp Config::Set
            self.seed = parse_int("seed", v)?;
            let mut rand = Random::new(self.seed);
            let int_max = i16::MAX as i32;
            self.data_random_seed = rand.next_short(0, int_max);
            self.bagging_seed = rand.next_short(0, int_max);
            self.drop_seed = rand.next_short(0, int_max);
            self.feature_fraction_seed = rand.next_short(0, int_max);
            self.objective_seed = rand.next_short(0, int_max);
            self.extra_seed = rand.next_short(0, int_max);
        }

        macro_rules! set_int {
            ($field:ident) => {
                if let Some(v) = p.get(stringify!($field)) {
                    self.$field = parse_int(stringify!($field), v)?;
                }
            };
        }
        macro_rules! set_f64 {
            ($field:ident) => {
                if let Some(v) = p.get(stringify!($field)) {
                    self.$field = parse_double(stringify!($field), v)?;
                }
            };
        }
        macro_rules! set_bool {
            ($field:ident) => {
                if let Some(v) = p.get(stringify!($field)) {
                    self.$field = parse_bool(stringify!($field), v)?;
                }
            };
        }

        if let Some(v) = p.get("objective") {
            self.objective = parse_objective_alias(&v.to_ascii_lowercase());
        }
        if let Some(v) = p.get("boosting") {
            self.boosting = match v.to_ascii_lowercase().as_str() {
                "gbdt" | "gbrt" => "gbdt".to_string(),
                "dart" | "goss" | "rf" | "random_forest" => {
                    return Err(LgbmError::Unsupported(format!("boosting={v}")));
                }
                _ => {
                    return Err(LgbmError::InvalidParameter(format!("Unknown boosting type {v}")));
                }
            };
        }
        let metric_value = p.get("metric").map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        self.metric = if metric_value.is_empty() {
            parse_metrics(&self.objective)
        } else {
            parse_metrics(&metric_value)
        };

        set_int!(num_iterations);
        set_f64!(learning_rate);
        set_int!(num_leaves);
        set_int!(num_threads);
        set_bool!(deterministic);
        set_bool!(force_col_wise);
        set_bool!(force_row_wise);
        set_int!(max_depth);
        set_int!(min_data_in_leaf);
        set_f64!(min_sum_hessian_in_leaf);
        set_int!(early_stopping_round);
        set_f64!(early_stopping_min_delta);
        set_bool!(first_metric_only);
        set_f64!(max_delta_step);
        set_f64!(lambda_l1);
        set_f64!(lambda_l2);
        set_f64!(min_gain_to_split);
        set_f64!(path_smooth);
        set_int!(verbosity);
        set_int!(max_bin);
        set_int!(min_data_in_bin);
        set_int!(bin_construct_sample_cnt);
        set_int!(data_random_seed);
        set_int!(bagging_seed);
        set_int!(drop_seed);
        set_int!(feature_fraction_seed);
        set_int!(objective_seed);
        set_int!(extra_seed);
        set_bool!(use_missing);
        set_bool!(zero_as_missing);
        set_bool!(feature_pre_filter);
        set_bool!(is_unbalance);
        set_f64!(scale_pos_weight);
        set_f64!(sigmoid);
        set_bool!(boost_from_average);
        set_bool!(reg_sqrt);
        set_f64!(alpha);
        set_f64!(fair_c);
        set_f64!(poisson_max_delta_step);
        set_f64!(tweedie_variance_power);
        set_int!(num_class);
        set_int!(multi_error_top_k);
        set_int!(saved_feature_importance_type);
        set_bool!(is_provide_training_metric);

        if self.objective != "custom" && !SUPPORTED_OBJECTIVES.contains(&self.objective.as_str()) {
            return Err(LgbmError::Unsupported(format!("objective={}", self.objective)));
        }
        for m in &self.metric {
            if m != "custom" && !SUPPORTED_METRICS.contains(&m.as_str()) {
                return Err(LgbmError::Unsupported(format!("metric={m}")));
            }
        }
        // upstream: Config::CheckParamConflict (objective / metric / num_class)
        let is_multiclass = |o: &str| o == "multiclass" || o == "multiclassova";
        let objective_multiclass =
            is_multiclass(&self.objective) || (self.objective == "custom" && self.num_class > 1);
        if objective_multiclass {
            if self.num_class <= 1 {
                return Err(LgbmError::InvalidParameter(
                    "Number of classes should be specified and greater than 1 for multiclass training".into(),
                ));
            }
        } else if self.num_class != 1 {
            return Err(LgbmError::InvalidParameter("Number of classes must be 1 for non-multiclass training".into()));
        }
        for m in &self.metric {
            let metric_multiclass = is_multiclass(m)
                || matches!(m.as_str(), "multi_logloss" | "multi_error" | "auc_mu")
                || (m == "custom" && self.num_class > 1);
            if objective_multiclass != metric_multiclass {
                return Err(LgbmError::InvalidParameter("Multiclass objective and metrics don't match".into()));
            }
        }
        if self.is_unbalance && (self.scale_pos_weight - 1.0).abs() > 1e-6 {
            return Err(LgbmError::InvalidParameter(
                "Cannot set is_unbalance and scale_pos_weight at the same time".into(),
            ));
        }
        if self.saved_feature_importance_type != 0 && self.saved_feature_importance_type != 1 {
            return Err(LgbmError::InvalidParameter(
                "saved_feature_importance_type must be 0 (split) or 1 (gain)".into(),
            ));
        }
        // upstream: Config::CheckParamConflict (max_depth without num_leaves)
        if self.max_depth > 0 && p.get("num_leaves").is_none_or(|v| v.is_empty()) {
            let full_num_leaves = 2f64.powi(self.max_depth);
            if full_num_leaves > self.num_leaves as f64 {
                self.warnings.push(format!(
                    "Provided parameters constrain tree depth (max_depth={}) without explicitly setting 'num_leaves'. \
                     This can lead to underfitting. To resolve this warning, pass 'num_leaves' (<={full_num_leaves:.0}) in params. \
                     Alternatively, pass (max_depth=-1) and just use 'num_leaves' to constrain model complexity.",
                    self.max_depth
                ));
            }
            if full_num_leaves < self.num_leaves as f64 {
                self.num_leaves = full_num_leaves as i32;
            }
        }
        Ok(())
    }

    /// Render the `parameters:` block of the model text format.
    ///
    /// upstream: src/io/config.cpp `Config::ToString` + config_auto.cpp
    /// `SaveMembersToString`. Parameters flagged `[no-save]` are omitted.
    pub fn to_model_string(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("[boosting: {}]\n", self.boosting));
        out.push_str(&format!("[objective: {}]\n", self.objective));
        out.push_str(&format!("[metric: {}]\n", self.metric.join(",")));
        out.push_str("[tree_learner: serial]\n");
        out.push_str("[device_type: cpu]\n");
        for spec in param_specs() {
            if spec.flags.iter().any(|f| f == "no-save") {
                continue;
            }
            let v = match self.value_string(&spec.name) {
                Some(v) => normalize_value(spec, &v),
                None => normalize_value(spec, &model_default_string(spec)),
            };
            out.push_str(&format!("[{}: {}]\n", spec.name, v));
        }
        out
    }

    fn value_string(&self, name: &str) -> Option<String> {
        let b = |x: bool| if x { "1".to_string() } else { "0".to_string() };
        let g = crate::fmt::fmt_g6;
        Some(match name {
            "num_iterations" => self.num_iterations.to_string(),
            "learning_rate" => g(self.learning_rate),
            "num_leaves" => self.num_leaves.to_string(),
            "num_threads" => self.num_threads.to_string(),
            "seed" => self.seed.to_string(),
            "deterministic" => b(self.deterministic),
            "force_col_wise" => b(self.force_col_wise),
            "force_row_wise" => b(self.force_row_wise),
            "max_depth" => self.max_depth.to_string(),
            "min_data_in_leaf" => self.min_data_in_leaf.to_string(),
            "min_sum_hessian_in_leaf" => g(self.min_sum_hessian_in_leaf),
            "early_stopping_round" => self.early_stopping_round.to_string(),
            "early_stopping_min_delta" => g(self.early_stopping_min_delta),
            "first_metric_only" => b(self.first_metric_only),
            "max_delta_step" => g(self.max_delta_step),
            "lambda_l1" => g(self.lambda_l1),
            "lambda_l2" => g(self.lambda_l2),
            "min_gain_to_split" => g(self.min_gain_to_split),
            "path_smooth" => g(self.path_smooth),
            "verbosity" => self.verbosity.to_string(),
            "max_bin" => self.max_bin.to_string(),
            "min_data_in_bin" => self.min_data_in_bin.to_string(),
            "bin_construct_sample_cnt" => self.bin_construct_sample_cnt.to_string(),
            "data_random_seed" => self.data_random_seed.to_string(),
            "bagging_seed" => self.bagging_seed.to_string(),
            "drop_seed" => self.drop_seed.to_string(),
            "feature_fraction_seed" => self.feature_fraction_seed.to_string(),
            "objective_seed" => self.objective_seed.to_string(),
            "extra_seed" => self.extra_seed.to_string(),
            "use_missing" => b(self.use_missing),
            "zero_as_missing" => b(self.zero_as_missing),
            "feature_pre_filter" => b(self.feature_pre_filter),
            "is_unbalance" => b(self.is_unbalance),
            "scale_pos_weight" => g(self.scale_pos_weight),
            "sigmoid" => g(self.sigmoid),
            "boost_from_average" => b(self.boost_from_average),
            "reg_sqrt" => b(self.reg_sqrt),
            "alpha" => g(self.alpha),
            "fair_c" => g(self.fair_c),
            "poisson_max_delta_step" => g(self.poisson_max_delta_step),
            "tweedie_variance_power" => g(self.tweedie_variance_power),
            "num_class" => self.num_class.to_string(),
            "saved_feature_importance_type" => self.saved_feature_importance_type.to_string(),
            _ => return self.explicit.get(name).cloned(),
        })
    }
}

/// Reject changes to dataset-construction parameters once a dataset exists.
///
/// upstream: src/c_api.cpp `LGBM_DatasetUpdateParamChecking`. A key counts
/// as "changed" only when it is present in `new` and its value differs from
/// the value in effect for `old`.
pub fn dataset_update_param_checking(old: &Config, new: &Config) -> Result<()> {
    let has = |k: &str| new.explicit.contains_key(k);
    let fatal = |k: &str| -> Result<()> {
        Err(LgbmError::InvalidParameter(format!("Cannot change {k} after constructed Dataset handle.")))
    };
    macro_rules! typed {
        ($($f:ident),*) => {$(
            if has(stringify!($f)) && new.$f != old.$f {
                return fatal(stringify!($f));
            }
        )*};
    }
    typed!(data_random_seed, max_bin, bin_construct_sample_cnt, min_data_in_bin, use_missing,
        zero_as_missing, feature_pre_filter);
    let reg = registry();
    for k in [
        "max_bin_by_feature", "categorical_feature", "is_enable_sparse", "pre_partition",
        "enable_bundle", "header", "two_round", "label_column", "weight_column", "group_column",
        "ignore_column",
    ] {
        if let Some(v) = new.explicit.get(k) {
            let spec = &reg.specs[reg.by_name[k]];
            let same = match old.explicit.get(k) {
                Some(o) => same_value(spec, o, v),
                None => values_equal_to_default(spec, v),
            };
            if !same {
                return fatal(k);
            }
        }
    }
    if has("forcedbins_filename") {
        return Err(LgbmError::InvalidParameter(
            "Cannot change forced bins after constructed Dataset handle.".into(),
        ));
    }
    if has("min_data_in_leaf") && new.min_data_in_leaf < old.min_data_in_leaf && old.feature_pre_filter {
        return Err(LgbmError::InvalidParameter(
            "Reducing `min_data_in_leaf` with `feature_pre_filter=true` may cause unexpected behaviour \
             for features that were pre-filtered by the larger `min_data_in_leaf`.\n\
             You need to set `feature_pre_filter=false` to dynamically change the `min_data_in_leaf`."
                .into(),
        ));
    }
    for k in ["linear_tree", "precise_float_parser"] {
        if let Some(v) = new.explicit.get(k) {
            let spec = &reg.specs[reg.by_name[k]];
            let same = match old.explicit.get(k) {
                Some(o) => same_value(spec, o, v),
                None => values_equal_to_default(spec, v),
            };
            if !same {
                return fatal(k);
            }
        }
    }
    Ok(())
}

fn same_value(spec: &ParamSpec, a: &str, b: &str) -> bool {
    match spec.cpp_type.as_str() {
        "bool" => parse_bool(&spec.name, a).ok() == parse_bool(&spec.name, b).ok(),
        "int" | "double" => match (a.trim().parse::<f64>(), b.trim().parse::<f64>()) {
            (Ok(x), Ok(y)) => x == y,
            _ => a.trim() == b.trim(),
        },
        _ => a.trim() == b.trim(),
    }
}

/// The value a default-constructed upstream `Config` holds (what
/// `SaveMembersToString` prints), as opposed to the documented default.
/// Vector parameters are empty until an objective/metric fills them.
fn model_default_string(spec: &ParamSpec) -> String {
    if spec.cpp_type.starts_with("std::vector") {
        return String::new();
    }
    match spec.cpp_default.as_deref() {
        Some(d) if !d.starts_with('k') => d.trim_matches('"').to_string(),
        _ => spec_default_string(spec),
    }
}

/// Render a value the way upstream streams it: bools as 0/1, doubles with `%g`.
fn normalize_value(spec: &ParamSpec, v: &str) -> String {
    match spec.cpp_type.as_str() {
        "bool" => match parse_bool(&spec.name, v) {
            Ok(true) => "1".into(),
            Ok(false) => "0".into(),
            Err(_) => v.to_string(),
        },
        "double" => parse_double(&spec.name, v).map(crate::fmt::fmt_g6).unwrap_or_else(|_| v.to_string()),
        "int" => parse_int(&spec.name, v).map(|i| i.to_string()).unwrap_or_else(|_| v.to_string()),
        _ => v.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_all_upstream_params() {
        assert_eq!(param_specs().len(), 141);
        assert_eq!(canonical_name("eta"), Some("learning_rate"));
        assert_eq!(canonical_name("num_boost_round"), Some("num_iterations"));
        assert_eq!(canonical_name("reg_lambda"), Some("lambda_l2"));
        assert_eq!(canonical_name("min_child_samples"), Some("min_data_in_leaf"));
        assert_eq!(canonical_name("not_a_param"), None);
    }

    #[test]
    fn aliases_and_defaults() {
        let c = Config::from_pairs([("objective", "binary"), ("eta", "0.05"), ("num_leaf", "7")])
            .unwrap();
        assert_eq!(c.objective, "binary");
        assert_eq!(c.metric, vec!["binary_logloss"]);
        assert_eq!(c.learning_rate, 0.05);
        assert_eq!(c.num_leaves, 7);
        assert_eq!(c.min_data_in_leaf, 20);
    }

    #[test]
    fn canonical_name_wins_over_alias() {
        let c = Config::from_pairs([("eta", "0.5"), ("learning_rate", "0.2")]).unwrap();
        assert_eq!(c.learning_rate, 0.2);
        assert_eq!(
            c.warnings,
            vec!["learning_rate is set=0.2, eta=0.5 will be ignored. Current value: learning_rate=0.2"]
        );
    }

    #[test]
    fn max_depth_without_num_leaves_matches_check_param_conflict() {
        let c = Config::from_pairs([("max_depth", "3")]).unwrap();
        assert_eq!(c.num_leaves, 8);
        assert!(c.warnings.is_empty());
        let c = Config::from_pairs([("max_depth", "5")]).unwrap();
        assert_eq!(c.num_leaves, 31);
        assert!(c.warnings[0].contains("pass 'num_leaves' (<=32) in params"));
        let c = Config::from_pairs([("max_depth", "3"), ("num_leaves", "31")]).unwrap();
        assert_eq!(c.num_leaves, 31);
    }

    #[test]
    fn shortest_alias_wins_like_upstream_sort_alias() {
        // upstream: Config::SortAlias (shorter, then lexicographically smaller)
        let c = Config::from_pairs([("shrinkage_rate", "0.3"), ("eta", "0.5")]).unwrap();
        assert_eq!(c.learning_rate, 0.5);
        let c = Config::from_pairs([("eta", "0.5"), ("shrinkage_rate", "0.3")]).unwrap();
        assert_eq!(c.learning_rate, 0.5);
        let c = Config::from_pairs([("max_bin", "15"), ("max_bin", "31")]).unwrap();
        assert_eq!(c.max_bin, 15);
        assert!(c.warnings[0].starts_with("max_bin is set=15, max_bin=31 will be ignored"));
    }

    #[test]
    fn checks_are_enforced() {
        let e = Config::from_pairs([("num_leaves", "1")]).unwrap_err();
        assert!(matches!(e, LgbmError::InvalidParameter(_)), "{e:?}");
        let e = Config::from_pairs([("learning_rate", "0")]).unwrap_err();
        assert!(matches!(e, LgbmError::InvalidParameter(_)));
    }

    #[test]
    fn bool_parsing_matches_upstream() {
        assert!(Config::from_pairs([("use_missing", "False")]).is_ok());
        assert!(Config::from_pairs([("use_missing", "1")]).is_err());
    }

    #[test]
    fn unimplemented_non_default_is_rejected() {
        let e = Config::from_pairs([("bagging_fraction", "0.5")]).unwrap_err();
        assert!(matches!(e, LgbmError::Unsupported(_)));
        // the default value is accepted
        assert!(Config::from_pairs([("bagging_fraction", "1.0")]).is_ok());
        assert!(Config::from_pairs([("objective", "multiclass")]).is_err());
    }

    #[test]
    fn seed_derives_data_random_seed() {
        let c = Config::from_pairs([("seed", "42")]).unwrap();
        let mut r = Random::new(42);
        assert_eq!(c.data_random_seed, r.next_short(0, i16::MAX as i32));
        let c = Config::from_pairs([("seed", "42"), ("data_random_seed", "9")]).unwrap();
        assert_eq!(c.data_random_seed, 9);
    }

    #[test]
    fn dataset_param_update_checking() {
        let old = Config::from_pairs([("max_bin", "63")]).unwrap();
        let same = Config::from_pairs([("max_bin", "63"), ("learning_rate", "0.3")]).unwrap();
        assert!(dataset_update_param_checking(&old, &same).is_ok());
        let changed = Config::from_pairs([("max_bin", "15")]).unwrap();
        let e = dataset_update_param_checking(&old, &changed).unwrap_err();
        assert!(e.to_string().contains("Cannot change max_bin after constructed Dataset handle."));
        let smaller_leaf = Config::from_pairs([("min_data_in_leaf", "5")]).unwrap();
        assert!(dataset_update_param_checking(&old, &smaller_leaf).is_err());
        let default_bundle = Config::from_pairs([("enable_bundle", "true")]).unwrap();
        assert!(dataset_update_param_checking(&old, &default_bundle).is_ok());
    }

    #[test]
    fn metric_none_and_explicit_list() {
        let c = Config::from_pairs([("objective", "binary"), ("metric", "auc,binary_error,auc")])
            .unwrap();
        assert_eq!(c.metric, vec!["auc", "binary_error"]);
        let c = Config::from_pairs([("metric", "None")]).unwrap();
        assert_eq!(c.metric, vec!["custom"]);
    }
}
