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

use crate::consts::K_ZERO_THRESHOLD;
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
    "tweedie_variance_power", "multi_error_top_k", "bagging_fraction", "bagging_freq",
    "pos_bagging_fraction", "neg_bagging_fraction", "bagging_seed", "feature_fraction",
    "feature_fraction_bynode", "feature_fraction_seed", "extra_trees", "extra_seed",
    "data_sample_strategy", "top_rate", "other_rate", "objective_seed",
    "lambdarank_truncation_level", "lambdarank_norm", "label_gain",
    "lambdarank_position_bias_regularization", "eval_at", "categorical_feature",
    "min_data_per_group", "max_cat_threshold", "cat_l2", "cat_smooth", "max_cat_to_onehot",
    "pred_early_stop", "pred_early_stop_freq", "pred_early_stop_margin", "bagging_by_query",
    "auc_mu_weights", "refit_decay_rate", "monotone_constraints", "monotone_constraints_method",
    "monotone_penalty", "feature_contri", "cegb_tradeoff", "cegb_penalty_split", "cegb_penalty_feature_lazy",
    "cegb_penalty_feature_coupled", "interaction_constraints", "drop_rate", "max_drop", "skip_drop",
    "xgboost_dart_mode", "uniform_drop", "drop_seed", "header", "label_column", "weight_column",
    "group_column", "ignore_column", "precise_float_parser", "two_round", "max_bin_by_feature",
    "forcedbins_filename", "forcedsplits_filename", "use_quantized_grad", "num_grad_quant_bins",
    "quant_train_renew_leaf", "stochastic_rounding", "linear_tree", "linear_lambda",
];

/// Parameters that cannot change results here (threading, layout, logging,
/// CLI-only I/O, or sampling knobs that are inert at their other defaults).
const NO_EFFECT: &[&str] = &[
    "num_threads", "deterministic", "histogram_pool_size",
    "verbosity", "is_enable_sparse", "enable_bundle", "metric_freq", "snapshot_freq",
    "output_model", "input_model", "output_result", "data", "valid", "config", "task",
    "save_binary", "pre_partition",
    // Only read by non-CPU devices / multi-machine learners, which are gated via
    // device_type / num_machines / tree_learner.
    "gpu_platform_id", "gpu_device_id", "gpu_device_id_list", "gpu_use_dp", "num_gpu",
    "local_listen_port", "time_out", "machine_list_filename", "machines",
    "top_k",
    "convert_model_language", "convert_model",
];

pub const SUPPORTED_OBJECTIVES: &[&str] = &[
    "regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma", "tweedie",
    "binary", "multiclass", "multiclassova", "lambdarank", "rank_xendcg", "cross_entropy",
    "cross_entropy_lambda",
];
pub const SUPPORTED_METRICS: &[&str] = &[
    "l2", "rmse", "l1", "quantile", "huber", "fair", "poisson", "mape", "gamma", "gamma_deviance",
    "tweedie", "binary_logloss", "binary_error", "auc", "average_precision", "r2", "multi_logloss",
    "multi_error", "auc_mu", "ndcg", "map", "cross_entropy", "cross_entropy_lambda", "kullback_leibler",
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

/// upstream: utils/common.h `Split(str, ',')` (empty tokens are dropped).
fn split_tokens(s: &str) -> impl Iterator<Item = &str> {
    s.split(',').filter(|t| !t.is_empty())
}

/// upstream: utils/common.h `SplitBrackets` (contents of each `[...]`; empty
/// brackets and text outside brackets are dropped).
fn split_brackets(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = None;
    for (pos, c) in s.char_indices() {
        if c == '[' {
            start = Some(pos + 1);
        } else if c == ']' {
            if let Some(i) = start.take() {
                if i < pos {
                    out.push(&s[i..pos]);
                }
            }
        }
    }
    out
}

/// upstream: utils/common.h `Atoi` (leading spaces, sign, digits; stops at
/// the first other character).
fn atoi(s: &str) -> i32 {
    let b = s.trim_start_matches(' ').as_bytes();
    let (neg, digits) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        Some(b'+') => (false, &b[1..]),
        _ => (false, b),
    };
    let mut v: i32 = 0;
    for &c in digits.iter().take_while(|c| c.is_ascii_digit()) {
        v = v.wrapping_mul(10).wrapping_add((c - b'0') as i32);
    }
    if neg { v.wrapping_neg() } else { v }
}

/// upstream: utils/common.h `Atoi<int8_t>` (the accumulator is `int8_t`).
pub(crate) fn atoi_i8(s: &str) -> i8 {
    let b = s.trim_start_matches(' ').as_bytes();
    let (neg, digits) = match b.first() {
        Some(b'-') => (true, &b[1..]),
        Some(b'+') => (false, &b[1..]),
        _ => (false, b),
    };
    let mut v: i8 = 0;
    for &c in digits.iter().take_while(|c| c.is_ascii_digit()) {
        v = (v as i32 * 10 + (c - b'0') as i32) as i8;
    }
    if neg { (-(v as i32)) as i8 } else { v }
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

/// upstream: src/io/config.cpp `Config::GetAucMuWeights`.
fn auc_mu_weights_matrix(weights: &[f64], num_class: i32) -> Result<Vec<Vec<f64>>> {
    let n = num_class.max(0) as usize;
    if weights.is_empty() {
        let mut m = vec![vec![1.0; n]; n];
        for (i, row) in m.iter_mut().enumerate() {
            row[i] = 0.0;
        }
        return Ok(m);
    }
    if weights.len() != n * n {
        return Err(LgbmError::InvalidParameter(format!(
            "auc_mu_weights must have {} elements, but found {}",
            n * n,
            weights.len()
        )));
    }
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                if weights[i * n + j].abs() < K_ZERO_THRESHOLD {
                    return Err(LgbmError::InvalidParameter(format!(
                        "AUC-mu matrix must have non-zero values for non-diagonal entries. Found zero value in position {} of auc_mu_weights.",
                        i * n + j
                    )));
                }
                m[i][j] = weights[i * n + j];
            } else if weights[i * n + j].abs() > K_ZERO_THRESHOLD {
                crate::log::info(&format!(
                    "AUC-mu matrix must have zeros on diagonal. Overwriting value in position {} of auc_mu_weights with 0.",
                    i * n + j
                ));
            }
        }
    }
    Ok(m)
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

/// upstream: `Config::GetDouble` -> `Common::AtofAndCheck`, i.e. the legacy,
/// not correctly rounded `Common::Atof` ("inf" is 1e308).
pub(crate) fn parse_double(key: &str, v: &str) -> Result<f64> {
    let t = v.trim();
    if let Some(x) = crate::fmt::atof_legacy(t) {
        return Ok(x);
    }
    let body = t.strip_prefix(['-', '+']).unwrap_or(t);
    if !body.starts_with(|c: char| c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E')) {
        let token: String = body
            .chars()
            .take_while(|c| !matches!(c, ' ' | '\t' | ',' | '\n' | '\r' | ':'))
            .collect::<String>()
            .to_ascii_lowercase();
        if !matches!(token.as_str(), "" | "na" | "nan" | "null" | "inf" | "infinity") {
            return Err(LgbmError::InvalidParameter(format!("Unknown token {token} in data file")));
        }
    }
    Err(LgbmError::InvalidParameter(format!("Parameter {key} should be of type double, got \"{v}\"")))
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
    pub refit_decay_rate: f64,
    /// By real feature index; empty means unconstrained.
    pub monotone_constraints: Vec<i8>,
    pub monotone_constraints_method: String,
    pub monotone_penalty: f64,
    /// Split-gain multiplier by real feature index; empty means 1 for all.
    pub feature_contri: Vec<f64>,
    pub cegb_tradeoff: f64,
    pub cegb_penalty_split: f64,
    /// By real feature index; empty means no lazy penalty.
    pub cegb_penalty_feature_lazy: Vec<f64>,
    /// By real feature index; empty means no coupled penalty.
    pub cegb_penalty_feature_coupled: Vec<f64>,
    pub drop_rate: f64,
    pub max_drop: i32,
    pub skip_drop: f64,
    pub xgboost_dart_mode: bool,
    pub uniform_drop: bool,
    pub interaction_constraints: String,
    /// Real feature indices per constraint set.
    pub interaction_constraints_vector: Vec<Vec<i32>>,
    pub verbosity: i32,
    pub max_bin: i32,
    /// By column; empty means `max_bin` for every feature.
    pub max_bin_by_feature: Vec<i32>,
    /// JSON file of forced numerical bin bounds (empty: none).
    pub forcedbins_filename: String,
    /// JSON file of splits forced at the top of every tree (empty: none).
    pub forcedsplits_filename: String,
    pub min_data_in_bin: i32,
    pub bin_construct_sample_cnt: i32,
    pub data_random_seed: i32,
    pub bagging_seed: i32,
    /// Kept so that `seed` derivation and the saved parameters match upstream.
    pub drop_seed: i32,
    pub feature_fraction_seed: i32,
    pub objective_seed: i32,
    pub extra_seed: i32,
    /// `"bagging"` or `"goss"`.
    pub data_sample_strategy: String,
    pub bagging_fraction: f64,
    pub bagging_freq: i32,
    pub pos_bagging_fraction: f64,
    pub neg_bagging_fraction: f64,
    pub top_rate: f64,
    pub other_rate: f64,
    pub feature_fraction: f64,
    pub feature_fraction_bynode: f64,
    pub extra_trees: bool,
    pub linear_tree: bool,
    pub linear_lambda: f64,
    pub use_quantized_grad: bool,
    pub num_grad_quant_bins: i32,
    pub quant_train_renew_leaf: bool,
    pub stochastic_rounding: bool,
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
    pub lambdarank_truncation_level: i32,
    pub lambdarank_norm: bool,
    /// Empty means `2^i - 1` (filled by the ranking objective / metric).
    pub label_gain: Vec<f64>,
    pub lambdarank_position_bias_regularization: f64,
    /// Sorted; empty means `1..=5`.
    pub eval_at: Vec<i32>,
    pub auc_mu_weights: Vec<f64>,
    /// upstream `Config::GetAucMuWeights`: `num_class x num_class`, zero diagonal.
    pub auc_mu_weights_matrix: Vec<Vec<f64>>,
    pub saved_feature_importance_type: i32,
    pub is_provide_training_metric: bool,
    /// Raw `categorical_feature` value; see [`Config::categorical_indices`].
    pub categorical_feature: String,
    pub max_cat_to_onehot: i32,
    pub max_cat_threshold: i32,
    pub cat_l2: f64,
    pub cat_smooth: f64,
    pub min_data_per_group: i32,
    pub bagging_by_query: bool,
    pub pred_early_stop: bool,
    pub pred_early_stop_freq: i32,
    pub pred_early_stop_margin: f64,
    /// Text-file loading (upstream `DatasetLoader`).
    pub header: bool,
    pub label_column: String,
    pub weight_column: String,
    pub group_column: String,
    pub ignore_column: String,
    pub two_round: bool,
    pub precise_float_parser: bool,
    pub predict_disable_shape_check: bool,
    /// Canonical key -> value string as supplied (after alias resolution).
    pub explicit: BTreeMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            objective: "regression".into(),
            boosting: "gbdt".into(),
            metric: Vec::new(),
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
            refit_decay_rate: 0.9,
            monotone_constraints: Vec::new(),
            monotone_constraints_method: "basic".into(),
            monotone_penalty: 0.0,
            feature_contri: Vec::new(),
            cegb_tradeoff: 1.0,
            cegb_penalty_split: 0.0,
            cegb_penalty_feature_lazy: Vec::new(),
            cegb_penalty_feature_coupled: Vec::new(),
            drop_rate: 0.1,
            max_drop: 50,
            skip_drop: 0.5,
            xgboost_dart_mode: false,
            uniform_drop: false,
            interaction_constraints: String::new(),
            interaction_constraints_vector: Vec::new(),
            verbosity: 1,
            max_bin: 255,
            max_bin_by_feature: Vec::new(),
            forcedbins_filename: String::new(),
            forcedsplits_filename: String::new(),
            min_data_in_bin: 3,
            bin_construct_sample_cnt: 200_000,
            data_random_seed: 1,
            bagging_seed: 3,
            drop_seed: 4,
            feature_fraction_seed: 2,
            objective_seed: 5,
            extra_seed: 6,
            data_sample_strategy: "bagging".into(),
            bagging_fraction: 1.0,
            bagging_freq: 0,
            pos_bagging_fraction: 1.0,
            neg_bagging_fraction: 1.0,
            top_rate: 0.2,
            other_rate: 0.1,
            feature_fraction: 1.0,
            feature_fraction_bynode: 1.0,
            extra_trees: false,
            linear_tree: false,
            linear_lambda: 0.0,
            use_quantized_grad: false,
            num_grad_quant_bins: 4,
            quant_train_renew_leaf: false,
            stochastic_rounding: true,
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
            lambdarank_truncation_level: 30,
            lambdarank_norm: true,
            label_gain: Vec::new(),
            lambdarank_position_bias_regularization: 0.0,
            eval_at: Vec::new(),
            auc_mu_weights: Vec::new(),
            auc_mu_weights_matrix: vec![vec![0.0]],
            saved_feature_importance_type: 0,
            is_provide_training_metric: false,
            categorical_feature: String::new(),
            max_cat_to_onehot: 4,
            max_cat_threshold: 32,
            cat_l2: 10.0,
            cat_smooth: 10.0,
            min_data_per_group: 100,
            bagging_by_query: false,
            pred_early_stop: false,
            pred_early_stop_freq: 10,
            pred_early_stop_margin: 10.0,
            header: false,
            label_column: String::new(),
            weight_column: String::new(),
            group_column: String::new(),
            ignore_column: String::new(),
            two_round: false,
            precise_float_parser: false,
            predict_disable_shape_check: false,
            explicit: BTreeMap::new(),
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
    /// As upstream, `verbosity` (or `verbose`) sets this thread's log level
    /// ([`crate::log`]) and warnings are logged with upstream's wording.
    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut cfg = Config::default();
        cfg.set(pairs)?;
        Ok(cfg)
    }

    /// upstream `Config::Set` on an existing config (`Booster::ResetConfig`):
    /// only the given keys change, then the conflict checks run again.
    /// Returns the canonical keys that were given.
    pub fn set<I, K, V>(&mut self, pairs: I) -> Result<BTreeMap<String, String>>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let explicit = Self::str2map(pairs)?;
        self.set_map(&explicit)?;
        Ok(explicit)
    }

    /// [`Config::set`] with keys already through `Str2Map` (canonical names).
    pub fn set_map(&mut self, explicit: &BTreeMap<String, String>) -> Result<()> {
        self.apply(explicit)?;
        self.explicit.extend(explicit.iter().map(|(k, v)| (k.clone(), v.clone())));
        Ok(())
    }

    /// upstream `Config::Str2Map`, including `SetVerbosity`.
    fn str2map<I, K, V>(pairs: I) -> Result<BTreeMap<String, String>>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let unquote = |s: &str| s.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        let all: Vec<(String, String)> = pairs
            .into_iter()
            .map(|(k, v)| (unquote(k.as_ref()), unquote(v.as_ref())))
            .filter(|(k, _)| !k.is_empty())
            .collect();

        // SetVerbosity: `verbosity` is preferred to `verbose`; other aliases are ignored
        let first_of = |name: &str| all.iter().find(|(k, _)| k == name);
        if let Some((key, val)) = first_of("verbosity").or_else(|| first_of("verbose")) {
            let v = crate::text_parser::atoi_and_check(val).ok_or_else(|| {
                LgbmError::InvalidParameter(format!("Parameter {key} should be of type int, got \"{val}\""))
            })?;
            crate::log::reset_log_level(crate::log::level_for_verbosity(v));
        }

        // KeepFirstValues
        let mut first: Vec<(String, String)> = Vec::new();
        for (key, val) in all {
            match first.iter().find(|(fk, _)| *fk == key) {
                Some((_, v0)) => crate::log::warning(&format!(
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
                None => crate::log::warning(&format!("Unknown parameter: {key}")),
                Some(name) if name == key => {
                    explicit.insert(key.clone(), val.clone());
                }
                Some(name) => match chosen_alias.get(name) {
                    None => {
                        chosen_alias.insert(name.to_string(), (key.clone(), val.clone()));
                    }
                    Some((a, av)) => {
                        if sort_alias(a, key) {
                            crate::log::warning(&format!(
                                "{name} is set with {a}={av}, {key}={val} will be ignored. Current value: {name}={av}"
                            ));
                        } else {
                            crate::log::warning(&format!(
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
                Some(cv) => crate::log::warning(&format!(
                    "{name} is set={cv}, {alias}={av} will be ignored. Current value: {name}={cv}"
                )),
                None => {
                    explicit.insert(name, av);
                }
            }
        }
        Ok(explicit)
    }

    fn apply(&mut self, p: &BTreeMap<String, String>) -> Result<()> {
        let reg = registry();
        // upstream `GetString` (string and list parameters) ignores an empty
        // value; `metric` reads it as "derive from the objective"
        let given: BTreeMap<String, String> = p
            .iter()
            .filter(|(name, value)| {
                let ty = reg.specs[reg.by_name[name.as_str()]].cpp_type.as_str();
                !value.is_empty() || name.as_str() == "metric" || matches!(ty, "int" | "double" | "bool")
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let p = &given;
        let linear = match p.get("linear_tree") {
            Some(v) => parse_bool("linear_tree", v)?,
            None => self.linear_tree,
        };
        // upstream: CheckParamConflict forces the serial learner and the CPU
        // device (from CUDA) for linear trees
        let forced_by_linear = |name: &str, value: &str| {
            let v = value.trim().to_ascii_lowercase();
            linear
                && match name {
                    "tree_learner" => matches!(
                        v.as_str(),
                        "serial" | "feature" | "feature_parallel" | "data" | "data_parallel" | "voting" | "voting_parallel"
                    ),
                    "device_type" => v == "cuda",
                    _ => false,
                }
        };
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
            if !honored && !no_effect && !values_equal_to_default(spec, value) && !forced_by_linear(name, value) {
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
                "goss" => "goss".to_string(),
                "dart" => "dart".to_string(),
                "rf" | "random_forest" => "rf".to_string(),
                _ => {
                    return Err(LgbmError::InvalidParameter(format!("Unknown boosting type {v}")));
                }
            };
        }
        if let Some(v) = p.get("data_sample_strategy") {
            // upstream: GetDataSampleStrategy
            self.data_sample_strategy = match v.to_ascii_lowercase().as_str() {
                s @ ("goss" | "bagging") => s.to_string(),
                _ => return Err(LgbmError::InvalidParameter(format!("Unknown sample strategy {v}"))),
            };
        }
        // upstream GetMetricType: the objective's metric only when none is set
        let metric_value = p.get("metric").map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        if p.contains_key("metric") {
            self.metric = if metric_value.is_empty() { Vec::new() } else { parse_metrics(&metric_value) };
        }
        if self.metric.is_empty() && metric_value.is_empty() {
            self.metric = parse_metrics(&self.objective);
        }

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
        set_f64!(refit_decay_rate);
        set_f64!(drop_rate);
        set_int!(max_drop);
        set_f64!(skip_drop);
        set_bool!(xgboost_dart_mode);
        set_bool!(uniform_drop);
        if let Some(v) = p.get("monotone_constraints") {
            // upstream: Common::StringToArray<int8_t> (Atoi<int8_t> per token)
            self.monotone_constraints = split_tokens(v).map(atoi_i8).collect();
        }
        if let Some(v) = p.get("monotone_constraints_method") {
            self.monotone_constraints_method = v.clone();
        }
        set_f64!(monotone_penalty);
        // upstream: Common::StringToArray<double> (std::stod per token)
        let doubles = |key: &str, v: &str| -> Result<Vec<f64>> {
            split_tokens(v)
                .map(|t| {
                    crate::fmt::parse_f64(t.trim())
                        .ok_or_else(|| LgbmError::InvalidParameter(format!("cannot parse {key} value `{t}`")))
                })
                .collect()
        };
        if let Some(v) = p.get("feature_contri") {
            self.feature_contri = doubles("feature_contri", v)?;
        }
        set_f64!(cegb_tradeoff);
        set_f64!(cegb_penalty_split);
        if let Some(v) = p.get("cegb_penalty_feature_lazy") {
            self.cegb_penalty_feature_lazy = doubles("cegb_penalty_feature_lazy", v)?;
        }
        if let Some(v) = p.get("cegb_penalty_feature_coupled") {
            self.cegb_penalty_feature_coupled = doubles("cegb_penalty_feature_coupled", v)?;
        }
        if let Some(v) = p.get("interaction_constraints") {
            // upstream: Config::Set, Common::StringToArrayofArrays<int>(s, '[', ']', ',')
            self.interaction_constraints = v.clone();
            self.interaction_constraints_vector =
                split_brackets(v).into_iter().map(|s| split_tokens(s).map(atoi).collect()).collect();
        }
        set_bool!(header);
        set_bool!(two_round);
        set_bool!(precise_float_parser);
        set_bool!(predict_disable_shape_check);
        for (field, key) in [
            (&mut self.label_column, "label_column"),
            (&mut self.weight_column, "weight_column"),
            (&mut self.group_column, "group_column"),
            (&mut self.ignore_column, "ignore_column"),
        ] {
            if let Some(v) = p.get(key) {
                *field = v.clone();
            }
        }
        set_int!(verbosity);
        set_int!(max_bin);
        if let Some(v) = p.get("max_bin_by_feature") {
            // upstream: Common::StringToArray<int32_t> (Atoi per token)
            self.max_bin_by_feature = split_tokens(v).map(atoi).collect();
        }
        if let Some(v) = p.get("forcedbins_filename") {
            self.forcedbins_filename = v.clone();
        }
        if let Some(v) = p.get("forcedsplits_filename") {
            self.forcedsplits_filename = v.clone();
        }
        set_int!(min_data_in_bin);
        set_int!(bin_construct_sample_cnt);
        set_int!(data_random_seed);
        set_int!(bagging_seed);
        set_int!(drop_seed);
        set_int!(feature_fraction_seed);
        set_int!(objective_seed);
        set_int!(extra_seed);
        set_f64!(bagging_fraction);
        set_int!(bagging_freq);
        set_f64!(pos_bagging_fraction);
        set_f64!(neg_bagging_fraction);
        set_f64!(top_rate);
        set_f64!(other_rate);
        set_f64!(feature_fraction);
        set_f64!(feature_fraction_bynode);
        set_bool!(extra_trees);
        set_bool!(linear_tree);
        set_f64!(linear_lambda);
        set_bool!(use_quantized_grad);
        set_int!(num_grad_quant_bins);
        set_bool!(quant_train_renew_leaf);
        set_bool!(stochastic_rounding);
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
        set_int!(lambdarank_truncation_level);
        set_bool!(lambdarank_norm);
        set_f64!(lambdarank_position_bias_regularization);
        if let Some(v) = p.get("label_gain") {
            // upstream: Common::StringToArray<double> (std::stod per token)
            self.label_gain = split_tokens(v)
                .map(|t| {
                    crate::fmt::parse_f64(t.trim()).ok_or_else(|| {
                        LgbmError::InvalidParameter(format!("cannot parse label_gain value `{t}`"))
                    })
                })
                .collect::<Result<_>>()?;
        }
        if let Some(v) = p.get("eval_at") {
            // upstream: Common::StringToArray<int> (Atoi per token), sorted in Config::Set
            self.eval_at = split_tokens(v).map(atoi).collect();
            self.eval_at.sort();
        }
        if let Some(v) = p.get("auc_mu_weights") {
            self.auc_mu_weights = split_tokens(v)
                .map(|t| {
                    crate::fmt::parse_f64(t.trim()).ok_or_else(|| {
                        LgbmError::InvalidParameter(format!("cannot parse auc_mu_weights value `{t}`"))
                    })
                })
                .collect::<Result<_>>()?;
        }
        self.auc_mu_weights_matrix = auc_mu_weights_matrix(&self.auc_mu_weights, self.num_class)?;
        set_int!(saved_feature_importance_type);
        set_bool!(is_provide_training_metric);
        if let Some(v) = p.get("categorical_feature") {
            self.categorical_feature = v.clone();
        }
        set_int!(max_cat_to_onehot);
        set_int!(max_cat_threshold);
        set_f64!(cat_l2);
        set_f64!(cat_smooth);
        set_int!(min_data_per_group);
        set_bool!(bagging_by_query);
        set_bool!(pred_early_stop);
        set_int!(pred_early_stop_freq);
        set_f64!(pred_early_stop_margin);

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
                crate::log::warning(&format!(
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
        if self.boosting == "goss" {
            self.boosting = "gbdt".into();
            self.data_sample_strategy = "goss".into();
            crate::log::warning(
                "Found boosting=goss. For backwards compatibility reasons, LightGBM interprets this as \
                 boosting=gbdt, data_sample_strategy=goss.To suppress this warning, set data_sample_strategy=goss instead.",
            );
        }
        // upstream: Config::CheckParamConflict (monotone constraints)
        let precise_mc = matches!(self.monotone_constraints_method.as_str(), "intermediate" | "advanced");
        if self.feature_fraction_bynode != 1.0 && precise_mc {
            crate::log::warning(
                "Cannot use \"intermediate\" or \"advanced\" monotone constraints with feature fraction different from 1, \
                 auto set monotone constraints to \"basic\" method.",
            );
            self.monotone_constraints_method = "basic".into();
        }
        if self.max_depth > 0 && self.monotone_penalty >= self.max_depth as f64 {
            crate::log::warning("Monotone penalty greater than tree depth. Monotone features won't be used.");
        }
        if self.bagging_by_query && self.data_sample_strategy != "bagging" {
            crate::log::warning("bagging_by_query=true is only compatible with data_sample_strategy=bagging. Setting bagging_by_query=false.");
            self.bagging_by_query = false;
        }
        // upstream: Config::CheckParamConflict (linear tree learner)
        if self.linear_tree {
            let current = |k: &str| p.get(k).or_else(|| self.explicit.get(k)).map(|v| v.trim().to_ascii_lowercase());
            if current("device_type").is_some_and(|v| v == "cuda") {
                crate::log::warning("Linear tree learner only works with CPU and GPU. Falling back to CPU now.");
            }
            if current("tree_learner").is_some_and(|v| v != "serial") {
                crate::log::warning("Linear tree learner must be serial.");
            }
            if self.zero_as_missing {
                return Err(LgbmError::InvalidParameter("zero_as_missing must be false when fitting linear trees.".into()));
            }
            if self.objective == "regression_l1" {
                return Err(LgbmError::InvalidParameter(
                    "Cannot use regression_l1 objective when fitting linear trees.".into(),
                ));
            }
            if self.use_quantized_grad {
                return Err(LgbmError::Unsupported(
                    "linear_tree with use_quantized_grad (upstream's linear tree learner does not discretize the \
                     gradients its quantized histograms read)"
                        .into(),
                ));
            }
        }
        Ok(())
    }

    /// Column indices listed in `categorical_feature`. Out-of-range indices
    /// are ignored by dataset construction, like upstream.
    ///
    /// upstream: src/io/dataset_loader.cpp `DatasetLoader::SetHeader`. Only
    /// file-based loading has column names, so `name:` lists always fail here.
    pub fn categorical_indices(&self) -> Result<Vec<i32>> {
        let v = self.categorical_feature.as_str();
        if v.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(names) = v.strip_prefix("name:") {
            let name = split_tokens(names).next().unwrap_or("");
            return Err(LgbmError::InvalidParameter(format!(
                "Could not find categorical_feature {name} in data file"
            )));
        }
        split_tokens(v)
            .map(|t| {
                let b = t.trim_start_matches(' ');
                let digits = b.strip_prefix(['-', '+']).unwrap_or(b);
                let n = digits.bytes().take_while(u8::is_ascii_digit).count();
                if digits[n..].trim_start_matches(' ').is_empty() {
                    Ok(atoi(t))
                } else {
                    Err(LgbmError::InvalidParameter(
                        "categorical_feature is not a number,\nif you want to use a column name,\n\
                         please add the prefix \"name:\" to the column name"
                            .into(),
                    ))
                }
            })
            .collect()
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
        let g17_join = |v: &[f64]| v.iter().map(|&x| crate::fmt::fmt_g17(x)).collect::<Vec<_>>().join(",");
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
            "drop_rate" => g(self.drop_rate),
            "max_drop" => self.max_drop.to_string(),
            "skip_drop" => g(self.skip_drop),
            "xgboost_dart_mode" => b(self.xgboost_dart_mode),
            "uniform_drop" => b(self.uniform_drop),
            "max_delta_step" => g(self.max_delta_step),
            "lambda_l1" => g(self.lambda_l1),
            "lambda_l2" => g(self.lambda_l2),
            "min_gain_to_split" => g(self.min_gain_to_split),
            "path_smooth" => g(self.path_smooth),
            "refit_decay_rate" => g(self.refit_decay_rate),
            "monotone_constraints" => {
                self.monotone_constraints.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(",")
            }
            "monotone_constraints_method" => self.monotone_constraints_method.clone(),
            "monotone_penalty" => g(self.monotone_penalty),
            "feature_contri" => g17_join(&self.feature_contri),
            "cegb_tradeoff" => g(self.cegb_tradeoff),
            "cegb_penalty_split" => g(self.cegb_penalty_split),
            "cegb_penalty_feature_lazy" => g17_join(&self.cegb_penalty_feature_lazy),
            "cegb_penalty_feature_coupled" => g17_join(&self.cegb_penalty_feature_coupled),
            "interaction_constraints" => self.interaction_constraints.clone(),
            "verbosity" => self.verbosity.to_string(),
            "max_bin" => self.max_bin.to_string(),
            "max_bin_by_feature" => {
                self.max_bin_by_feature.iter().map(|m| m.to_string()).collect::<Vec<_>>().join(",")
            }
            "forcedbins_filename" => self.forcedbins_filename.clone(),
            "forcedsplits_filename" => self.forcedsplits_filename.clone(),
            "min_data_in_bin" => self.min_data_in_bin.to_string(),
            "bin_construct_sample_cnt" => self.bin_construct_sample_cnt.to_string(),
            "data_random_seed" => self.data_random_seed.to_string(),
            "bagging_seed" => self.bagging_seed.to_string(),
            "drop_seed" => self.drop_seed.to_string(),
            "feature_fraction_seed" => self.feature_fraction_seed.to_string(),
            "objective_seed" => self.objective_seed.to_string(),
            "extra_seed" => self.extra_seed.to_string(),
            "data_sample_strategy" => self.data_sample_strategy.clone(),
            "bagging_fraction" => g(self.bagging_fraction),
            "bagging_freq" => self.bagging_freq.to_string(),
            "pos_bagging_fraction" => g(self.pos_bagging_fraction),
            "neg_bagging_fraction" => g(self.neg_bagging_fraction),
            "top_rate" => g(self.top_rate),
            "other_rate" => g(self.other_rate),
            "feature_fraction" => g(self.feature_fraction),
            "feature_fraction_bynode" => g(self.feature_fraction_bynode),
            "extra_trees" => b(self.extra_trees),
            "linear_tree" => b(self.linear_tree),
            "linear_lambda" => g(self.linear_lambda),
            "use_quantized_grad" => b(self.use_quantized_grad),
            "num_grad_quant_bins" => self.num_grad_quant_bins.to_string(),
            "quant_train_renew_leaf" => b(self.quant_train_renew_leaf),
            "stochastic_rounding" => b(self.stochastic_rounding),
            "bagging_by_query" => b(self.bagging_by_query),
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
            "lambdarank_truncation_level" => self.lambdarank_truncation_level.to_string(),
            "lambdarank_norm" => b(self.lambdarank_norm),
            // upstream: Common::Join (precision digits10 + 2)
            "label_gain" => self.label_gain.iter().map(|&x| crate::fmt::fmt_g17(x)).collect::<Vec<_>>().join(","),
            "lambdarank_position_bias_regularization" => g(self.lambdarank_position_bias_regularization),
            "eval_at" => self.eval_at.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(","),
            "auc_mu_weights" => {
                self.auc_mu_weights.iter().map(|&x| crate::fmt::fmt_g17(x)).collect::<Vec<_>>().join(",")
            }
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
    typed!(data_random_seed, max_bin, max_bin_by_feature, bin_construct_sample_cnt, min_data_in_bin, use_missing,
        zero_as_missing, feature_pre_filter);
    let reg = registry();
    for k in [
        "categorical_feature", "is_enable_sparse", "pre_partition",
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
    fn interaction_constraints_parse_like_string_to_array_of_arrays() {
        let c = Config::from_pairs([("interaction_constraints", "[0,1, 2],[],x[3][[4,,5]")]).unwrap();
        assert_eq!(c.interaction_constraints_vector, vec![vec![0, 1, 2], vec![3], vec![4, 5]]);
        assert_eq!(c.value_string("interaction_constraints").as_deref(), Some("[0,1, 2],[],x[3][[4,,5]"));
    }

    #[test]
    fn canonical_name_wins_over_alias() {
        let (c, warnings) = crate::log::capture(|| Config::from_pairs([("eta", "0.5"), ("learning_rate", "0.2")]));
        assert_eq!(c.unwrap().learning_rate, 0.2);
        assert_eq!(warnings, vec!["learning_rate is set=0.2, eta=0.5 will be ignored. Current value: learning_rate=0.2"]);
    }

    #[test]
    fn max_depth_without_num_leaves_matches_check_param_conflict() {
        let (c, warnings) = crate::log::capture(|| Config::from_pairs([("max_depth", "3")]).unwrap());
        assert_eq!(c.num_leaves, 8);
        assert!(warnings.is_empty());
        let (c, warnings) = crate::log::capture(|| Config::from_pairs([("max_depth", "5")]).unwrap());
        assert_eq!(c.num_leaves, 31);
        assert!(warnings[0].contains("pass 'num_leaves' (<=32) in params"));
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
        let (c, warnings) = crate::log::capture(|| Config::from_pairs([("max_bin", "15"), ("max_bin", "31")]).unwrap());
        assert_eq!(c.max_bin, 15);
        assert!(warnings[0].starts_with("max_bin is set=15, max_bin=31 will be ignored"));
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
        let e = Config::from_pairs([("tree_learner", "data")]).unwrap_err();
        assert!(matches!(e, LgbmError::Unsupported(_)));
        // the default value is accepted
        assert!(Config::from_pairs([("tree_learner", "serial")]).is_ok());
        assert!(Config::from_pairs([("objective", "multiclass")]).is_err());
    }

    #[test]
    fn boosting_goss_is_gbdt_with_goss_sampling() {
        let (c, warnings) = crate::log::capture(|| Config::from_pairs([("boosting", "goss")]).unwrap());
        assert_eq!((c.boosting.as_str(), c.data_sample_strategy.as_str()), ("gbdt", "goss"));
        assert!(warnings[0].starts_with("Found boosting=goss."));
        assert!(Config::from_pairs([("data_sample_strategy", "x")]).is_err());
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
