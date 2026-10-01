# Compatibility matrix

Target: LightGBM **v4.7.0** (commit `8f7036f`, see [UPSTREAM.md](UPSTREAM.md)). State as of the end of milestone 3. The numbers come from `tests/report/summary.md` (regenerate with the commands in [TESTING.md](TESTING.md)).

**lightgbm-rust does not have full parity with LightGBM.** It implements a CPU, dense, numerical-feature subset for L2 regression and binary classification. Within that subset, results match upstream bit for bit on every differential case run so far. Everything outside the subset either raises `LightGBMError("not supported by lightgbm-rust yet: ...")` or is absent.

## How to read this

Each feature is assessed on four separate axes:

| axis | meaning |
|---|---|
| **API** | the Python names, signatures, defaults, and error types/messages match `lightgbm` 4.7.0 |
| **Behavior** | the numerical results match upstream (bins, gradients, trees, predictions, metrics) |
| **Model format** | the model text written and read is interchangeable with upstream |
| **Performance** | speed and memory are comparable with upstream |

Each axis takes one of these statuses:
- `verified`: implemented, with differential or upstream-test evidence cited in the row.
- `implemented`: implemented and covered by our own tests, without upstream-comparison evidence.
- `partial`: some of the feature works; the gaps are listed.
- `not started`
- `n/a`

A dash in the Tolerance column means the comparison is exact or there is no numerical output. Named tolerances refer to sections of [`tests/tolerances.toml`](../tests/tolerances.toml).

The **evidence** referred to throughout comes from three places:
- **Differential tests (D):** `tests/differential`, run against the PyPI `lightgbm==4.7.0` wheel. These are 25 cases of 3,000 to 8,000 rows with 6 features, run single-threaded plus a 4-thread run. There are 12,235 comparisons. All are bitwise exact except 345 values of internal (non-leaf) nodes, `internal_value` and `internal_weight`, which differ by at most 2.2e-16 relative.
- **Upstream tests (U):** upstream's own `tests/python_package_test`, run unmodified through the import shim.
- **Rust tests (R):** `cargo test -p lgbm-core`, including the ported C++ tests.

Upstream-suite totals: **130 passed, 0 failed, 448 unsupported, 14 skipped**. On top of those cases, `test_sklearn.py` (65 functions) cannot be collected, and `test_dask.py` (32 functions) is skipped because dask is not installed.

## Python API surface

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| `lightgbm.Dataset` (construction, lazy `construct`, `create_valid`, `set_*`/`get_*` fields, `get_params`, `set_reference`, `feature_num_bin`, `_dump_text`) | partial | verified (supported inputs) | n/a | implemented | U: test_basic 51 passed (e.g. `test_chunked_dataset`, `test_consistent_state_for_dataset_fields`, `test_equal_datasets_from_row_major_and_col_major_data`, `test_set_field_none_removes_field[weight/init_score]`); D: bins | – | Not implemented: `subset`, `save_binary`, `add_features_from`, `group`/`position`, construction from files, `Sequence`, sparse/Arrow/Polars inputs, categorical features. All of these raise "not supported". |
| `lightgbm.Booster`: `update` (incl. `fobj`), `rollback_one_iter`, `eval`/`eval_train`/`eval_valid`, `add_valid`, `predict`, `save_model`, `model_to_string`, `model_from_string`, `feature_importance`, `feature_name`, `num_trees`, `current_iteration`, `num_feature`, `free_dataset`, pickling | partial | verified (supported params) | verified | partial | U: `test_booster_rollback_one_iter`, `test_booster_eval_adds_new_valid_dataset`, `test_model_size`, `test_feature_name*`; D: all cases | predictions, metrics | Not implemented: `dump_model` (JSON), `refit`, `trees_to_dataframe`, `set_leaf_output`, `get_leaf_output`, `lower_bound`/`upper_bound`, `shuffle_models`, `get_split_value_histogram`, `set_network`. `reset_parameter` supports only switching the objective to none. |
| `lightgbm.train` | implemented | verified | n/a | partial | U: `test_early_stopping*` (7), `test_train_only_raises_num_rounds_warning_when_expected`, `test_verbosity_*`, `test_train_raises_informative_error_*`; D: early stopping | metrics | `keep_training_booster=False` reloads the model from text and frees the training data, as upstream does. `init_model` raises "not supported". |
| `lightgbm.cv`, `CVBooster` | not started | not started | n/a | not started | U: 17 cases, all unsupported | – | Raises "not supported". |
| Callbacks: `early_stopping`, `log_evaluation`, `record_evaluation`, `reset_parameter`, `EarlyStopException`, `CallbackEnv` | implemented (port of upstream `callback.py`) | verified | n/a | n/a | U: test_callback 16/16 passed; engine early-stopping tests | metrics | `reset_parameter` callbacks that change parameters other than the objective raise "not supported", because `Booster.reset_parameter` does not yet support them. |
| `register_logger`, log routing, `verbosity` | implemented | verified | n/a | n/a | U: `test_register_invalid_logger`, `test_verbosity_and_verbose`, `test_verbosity_is_respected_when_using_custom_objective`, `test_max_depth_warning_*` (3) | – | Engine warnings are produced by the Rust core and printed by the Python layer, with upstream text and routing (`[LightGBM] [Warning] ...` through the logger's `info` method). Upstream C++ *info*-level messages (e.g. "Total Bins ...") are not emitted. |
| `EvalResult`, `Sequence` (class only) | implemented | n/a | n/a | n/a | U: `test_eval_result_*` | – | Using a `Sequence` as Dataset input raises "not supported". |
| Parameter handling: aliases, `_choose_param_value`, `_ConfigAliases`, unknown parameters, alias conflicts, type errors | implemented | verified | n/a | n/a | U: `test_param_aliases`, `test_choose_param_value*` (3), `test_dataset_params_with_reference`, `test_train_raises_informative_error_for_params_of_wrong_type`; R: config tests | – | The alias table is generated from upstream `config.h` (141 parameters; it matches upstream `LGBM_DumpParamAliases`). Parameters outside the supported subset are rejected if set to a non-default value ("only the upstream default is accepted"). |
| scikit-learn API (`LGBMModel`, `LGBMRegressor`, `LGBMClassifier`, `LGBMRanker`) | not started | not started | not started | not started | U: test_sklearn.py, 65 functions, not collected | – | – |
| Plotting (`plot_importance`, `plot_tree`, ...) | not started | not started | n/a | n/a | U: test_plotting, 4 unsupported, 11 skipped (no matplotlib/graphviz) | – | – |
| Rust API (`lgbm_core::{Dataset, Gbdt, Config}`) | implemented (unstable) | verified (same engine) | verified | partial | R: `tests/end_to_end.rs` | – | Pre-1.0 and not semver-stable. |
| C API, CLI, R package | n/a | n/a | n/a | n/a | C API internals: 20 C++ tests not portable | – | Out of scope. lightgbm-rust exposes no C ABI. |

## Dataset and binning

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| Numerical bin finding (`max_bin`, `min_data_in_bin`, `bin_construct_sample_cnt`, `data_random_seed`, `use_missing`, `zero_as_missing`, `feature_pre_filter`, `min_data_in_leaf` pre-filter) | implemented | verified | verified (`feature_infos`) | partial | D: `bins` exact in all cases, incl. `reg_max_bin_1023`, `reg_sampled_bins`, `reg_discrete`, `*_nan_zero`; U: `test_missing_value_handle*` (5), `test_small_max_bin`, `test_constant_features_*` | bins (exact) | – |
| Dense NumPy input: float32/float64, C- or F-order; other numeric dtypes converted to float32; non-contiguous copied | implemented | verified | n/a | implemented (borrowed, no copy, for f32/f64 contiguous input) | U: `test_equal_datasets_from_row_major_and_col_major_data`, `test_equal_predict_from_row_major_and_col_major_data`; Python API tests | bins | Upstream's private helper `_np2d_to_np1d` is not provided (4 upstream tests use it directly). |
| List of 2-D NumPy arrays | implemented | verified | n/a | partial (rows are stacked into one copy) | U: `test_chunked_dataset`, `test_equal_datasets_from_one_and_several_matrices_w_different_layouts` | bins | – |
| pandas DataFrame (numeric/bool/nullable columns) | implemented | verified | n/a | partial (always copies) | U: test_pandas 16 passed (`test_pandas_supported_dtypes`, `test_pandas_unsupported_dtypes`, `test_pandas_sparse`, ...) | bins | Upstream's private helper `_data_from_pandas` is not provided (10 upstream tests call it directly). |
| pandas categorical columns, `categorical_feature` | not started | not started | not started | not started | U: 19 unsupported | – | – |
| Labels, weights, `init_score` (incl. NaN/inf clamping by `AvoidInf`, dropping all-ones weights) | implemented | verified | n/a | implemented | D: `*_weighted`, `*_init_score`; U: `test_consistent_state_for_dataset_fields`, `test_init_score_for_multiclass_classification` | gradients, predictions | – |
| Feature names (auto `Column_i`, whitespace → `_`, JSON-character and duplicate checks) | implemented | verified | verified | n/a | U: `test_feature_name`, `test_feature_name_with_non_ascii`, `test_feature_names_are_set_correctly_when_no_feature_names_passed_into_Dataset`, `test_set_feature_name_updates_has_non_default_feature_names` | – | – |
| Validation sets sharing the training bin mappers | implemented | verified | n/a | implemented | D: metrics and early stopping; U: `test_dataset_params_with_reference` | metrics | – |
| Exclusive feature bundling (`enable_bundle`) | not started | partial | n/a | not started | – | – | No bundling is performed. EFB with the default `max_conflict_rate=0` is lossless, so results are expected to be unchanged; this is confirmed only on dense differential data. On sparse, wide data, memory and speed will be worse than upstream. |
| Sparse (SciPy CSR/CSC), Arrow, Polars, `Sequence`, text/binary files, `two_round`, `header`, `label_column`, ... | not started | not started | n/a | not started | U: input-arrow 183, input-polars 65, input-sparse 3 unsupported | – | Planned for milestone 4 (sparse, Arrow, Polars). |
| `max_bin_by_feature`, `forcedbins_filename`, `linear_tree` | not started | not started | not started | not started | U: unsupported | – | – |
| `Dataset.save_binary`, binary dataset files, `subset`, `add_features_from` | not started | not started | not started | not started | U: 24 + 5 + 3 unsupported | – | – |

## Tree learning

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| Leaf-wise growth: `num_leaves`, `max_depth` (incl. upstream's `num_leaves` reduction and warning), `min_data_in_leaf`, `min_sum_hessian_in_leaf` | implemented | verified | verified | partial | D: `tree_structure` all exact (`reg_depth`, `reg_depth_without_num_leaves`, `reg_100_rounds`, `bin_100_rounds`); U: `test_max_depth_warning_*` | tree_structure (exact) | – |
| Split gain and leaf output: `lambda_l1`, `lambda_l2`, `min_gain_to_split`, `max_delta_step`, `path_smooth` | implemented | verified | verified | n/a | D: `reg_regularized`, `split_gain` exact, leaf values and leaf weights exact; U: `test_path_smoothing` | split_gain, leaf_value | Internal-node `internal_value`/`internal_weight` differ in the last bit in 345 of the comparisons (max 2.2e-16 relative); the cause has not been isolated yet. These values are not used for prediction. Upstream saves them with 6 significant digits, so the model text is still byte-identical. |
| Missing values (NaN/zero as missing, default direction search) | implemented | verified | verified | n/a | D: `*_nan_zero`, `reg_zero_as_missing`, `reg_no_missing`; U: `test_missing_value_handle*` | tree_structure | – |
| Histogram construction, subtraction trick, most-frequent-bin fix-up | implemented | verified | n/a | verified (see Performance) | D: all cases | – | – |
| `force_row_wise` / `force_col_wise`, `deterministic`, `num_threads` | implemented | verified | verified (saved in params) | verified | D: `test_multithread` (row-wise 4 threads vs upstream 4 threads; col-wise 4 vs 1 thread); Rust `histogram_layouts_and_thread_counts_agree` | multithread | Setting neither flag selects row-wise without upstream's timing test (upstream picks the faster of the two at run time). Col-wise results do not depend on the thread count. Row-wise results depend on it exactly as upstream's do. |
| Bagging, `feature_fraction`, `feature_fraction_bynode`, `extra_trees`, GOSS sampling | not started | not started | not started | not started | U: unsupported (e.g. `test_node_level_subcol`, `test_goss_boosting_and_strategy_equivalent`) | – | Rejected if set to a non-default value. |
| Monotone and interaction constraints, CEGB, forced splits, quantized gradients, linear trees | not started | not started | not started | not started | U: constraints 7, linear-tree 5, quantized 1 unsupported | – | Rejected if set to a non-default value. |
| Categorical splits | not started | not started | not started | not started | U: unsupported | – | – |

## Boosting

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| GBDT: `learning_rate`, `num_iterations` and aliases, `boost_from_average`, score updates | implemented | verified | verified | partial | D: all 25 cases incl. `reg_no_boost_from_average`, `reg_100_rounds`, `bin_100_rounds` | predictions | – |
| Early stopping (`early_stopping_round`, `early_stopping_min_delta`, `first_metric_only`, train set ignored) | implemented | verified | verified (`best_iteration` in model) | n/a | U: 7 `test_early_stopping*` passed; D: `test_metrics_and_early_stopping` | metrics | – |
| Custom objective (`fobj` / callable `objective`) | implemented | verified | verified (`objective=custom`) | n/a | U: `test_objective_callable_train_regression`, `test_objective_callable_train_binary_classification`, `test_verbosity_is_respected_when_using_custom_objective`; Python API test (custom L2 equals built-in L2 exactly) | predictions | – |
| `rollback_one_iter` | implemented | verified | n/a | n/a | U: `test_booster_rollback_one_iter` | – | – |
| Continued training (`init_model`), `refit`, `keep_training_booster` + reload | not started | not started | n/a | n/a | U: continued-training 9 unsupported | – | – |
| DART, random forest (`rf`), GOSS boosting | not started | not started | not started | not started | U: unsupported | – | – |

## Objectives

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| `regression` (L2), incl. `reg_sqrt` | implemented | verified | verified | partial | D: all `reg_*` cases, `gradients` exact; U: `test_regression[regression]`, `test_constant_features_regression`; R: finite-difference tests | gradients (exact) | – |
| `binary` (logloss), incl. `sigmoid`, `is_unbalance`, `scale_pos_weight` | implemented | verified | verified | partial | D: all `bin_*` cases, `gradients` exact; U: `test_binary`, `test_constant_features_binary`; R: finite-difference tests | gradients (exact) | – |
| `regression_l1`, `huber`, `fair`, `poisson`, `quantile`, `mape`, `gamma`, `tweedie` | not started | not started | not started | not started | U: e.g. `test_regression[huber/fair/poisson/quantile]` unsupported | – | – |
| `multiclass`, `multiclassova` | not started | not started | not started | not started | U: multiclass 8 unsupported | – | The booster supports several trees per iteration internally, but no multi-output objective is implemented. |
| `cross_entropy`, `cross_entropy_lambda`, `lambdarank`, `rank_xendcg` | not started | not started | not started | not started | U: ranking 4 unsupported | – | – |
| Grouped / multi-output objective interface (`GroupedObjective`, `GradHessBlock`, `HessianMode`, `DiagonalReduction`) | implemented (Rust only) | implemented | implemented (`hessian_reduction:` recorded) | n/a | R: finite-difference test of a synthetic grouped objective with cross-row coupling | – | No concrete grouped objective is shipped. The hazard objective is a design note only ([hazard/design-note.md](hazard/design-note.md)). |

## Metrics

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| `l2`, `rmse`, `l1`, `binary_logloss`, `binary_error`, `auc` | implemented | verified | n/a | implemented | D: `metrics` all exact (`reg_heavy_tail` for l1/rmse, `bin_basic` for auc/error); U: `test_default_objective_and_metric`, `test_record_evaluation_with_train` | metrics | – |
| Custom `feval` (single and list) | implemented | verified | n/a | n/a | U: `test_multiple_feval_train`, `test_booster_eval_adds_new_valid_dataset` | – | – |
| Other metrics (`mape`, `r2`, `average_precision`, `multi_logloss`, `ndcg`, `map`, `auc_mu`, ...) | not started | not started | n/a | not started | U: metrics 5 unsupported | – | – |

## Prediction

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| Normal, `raw_score`, `pred_leaf`; `start_iteration`/`num_iteration`; `best_iteration` default | implemented | verified | n/a | verified (faster than upstream, see Performance) | D: `predictions` all exact; U: `test_predict_stump`, `test_equal_predict_from_row_major_and_col_major_data` | predictions | – |
| pandas input with `validate_features` | implemented | implemented | n/a | n/a | Python API tests | – | Upstream's validate-features test is unsupported only because it also calls `refit`. |
| `pred_contrib` (SHAP), `pred_early_stop`, sparse input | not started | not started | n/a | not started | U: prediction 9 unsupported | – | – |

## Persistence

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| Model text write (`save_model`, `model_to_string`, incl. `num_iteration`, `start_iteration`, `importance_type`) | implemented | verified | verified (byte-identical to upstream in all 25 cases) | implemented | D: `model_text` exact in every case | model_text (exact) | – |
| Model text read (`Booster(model_file/model_str)`, `model_from_string`) of models with numerical splits | implemented | verified | verified (upstream models load in lightgbm-rust and vice versa; predictions identical) | implemented | D: `test_model_cross_loading` | predictions | Upstream models that use features outside the subset (categorical splits, linear trees, other objectives) are rejected with "not supported". |
| Pickle / `copy.deepcopy` of `Booster` | implemented | implemented | via model text | n/a | U: callback picklability tests; Python API tests | – | – |
| JSON `dump_model`, `trees_to_dataframe` | not started | not started | not started | n/a | U: 7 + 3 unsupported | – | – |

## Parallelism, GPU, distributed

| feature | API | Behavior | Model format | Performance | upstream tests | tolerance | known differences |
|---|---|---|---|---|---|---|---|
| CPU multithreading (`num_threads`; GIL released during long calls) | implemented | verified | n/a | verified (faster than upstream on the benchmark matrix, see Performance) | D: `test_multithread`; Rust `histogram_layouts_and_thread_counts_agree`; Python API test (GIL release) | multithread | Row-wise results match upstream at the same thread count. `BoostFromScore` is always summed sequentially, which equals upstream with `deterministic=true`. |
| GPU (OpenCL, CUDA) | not started | not started | n/a | not started | U: 1 skipped | – | Out of scope for milestones 1-3. |
| Distributed training (socket/MPI, Dask) | not started | not started | n/a | not started | U: test_dask, 32 functions skipped (dask not installed) | – | Setting `num_machines` > 1 or `machines` raises "not supported". |

## Performance

**Setup.**

- **Host:** a single machine (WSL 2 Ubuntu, 16 logical CPUs).
- **Data:** 1,000,000 synthetic rows, 100 iterations, `force_row_wise=True`, and otherwise identical parameters.
- **Tasks:**
  - `binary` (28 features)
  - `regression` (28 features)
  - `binary_255` (28 features, `num_leaves=255`, `min_data_in_leaf=100`)
  - `binary_wide` (200 features)
- **Timing:** the engines are interleaved, and each time is the best of 3 runs.
- **Predictions:** identical except for regression above 1 thread. There they differ by up to 1.8e-15 because of upstream's non-deterministic `BoostFromScore` reduction (see `docs/TESTING.md`).

Times are in seconds (upstream / lightgbm-rust); the ratio is rust / upstream training time.

| task | threads | construct | train | ratio | predict |
|---|---|---|---|---|---|
| binary | 1 | 1.73 / 1.47 | 8.82 / 6.02 | 0.68 | 0.47 / 0.28 |
| binary | 8 | 0.35 / 0.26 | 2.37 / 1.91 | 0.81 | 0.50 / 0.31 |
| binary | 16 | 0.29 / 0.20 | 5.78 / 2.03 | 0.35 | 0.56 / 0.34 |
| regression | 1 | 1.73 / 1.48 | 7.95 / 5.27 | 0.66 | 0.52 / 0.29 |
| regression | 8 | 0.35 / 0.26 | 2.13 / 1.77 | 0.83 | 0.54 / 0.31 |
| binary_255 | 1 | 1.74 / 1.47 | 15.16 / 10.98 | 0.72 | 0.85 / 0.49 |
| binary_255 | 8 | 0.35 / 0.26 | 5.66 / 5.39 | 0.95 | 0.94 / 0.51 |
| binary_wide | 1 | 13.99 / 13.52 | 59.88 / 44.62 | 0.75 | 0.53 / 0.26 |
| binary_wide | 8 | 2.53 / 1.94 | 12.43 / 10.34 | 0.83 | 0.58 / 0.29 |

The full matrix, including 4 and 16 threads, is in `tests/report/summary.md`.

- **Thread counts.** Upstream's 16-thread training time is erratic on this host, so its 16-thread ratios overstate the gap. The 8-thread ratios (0.81 to 0.95) are the conservative comparison.
- **Scope.** These numbers are indicative only: they come from one host, with no isolation from other load, and cover dense data only. Sparse data, bagging, and feature subsampling are not benchmarked yet.

## Unresolved items and assumptions

- **Error-message prefixes.** Core error messages carry a prefix (`invalid data: ...`). Upstream tests match with `re.search`, so the prefix does not affect them, but the strings are not byte-identical to upstream's `LightGBMError` text.
- **Info-level logs.** Upstream info-level log lines (dataset statistics, "Start training from score ...") are not emitted.
- **Small differential datasets.** The differential cases use small synthetic data (≤ 8,000 rows). The 1e6-row benchmark checked prediction equality only, which is exact, not every intermediate value.
- **Untested platforms.** Native Windows and macOS builds have not been verified in this pass; all evidence above is from Linux (WSL 2).
