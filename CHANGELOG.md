# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

When cutting a release, move the entries under `[Unreleased]` into a new
version section with its ISO 8601 date and update the comparison links.

## [Unreleased]

## [0.4.0] - 2026-10-02

### Added

- Linear trees (`linear_tree`, `linear_lambda`), including refit, model text and `dump_model` fields, and raw-value storage in Dataset objects.
- Quantized gradients (`use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`, `stochastic_rounding`).
- Feature-group storage: bundled, 4-bit dense, sparse and multi-value bins, with sparse input that is not densified.
- Cross-entropy objectives and metrics (`cross_entropy`, `cross_entropy_lambda`, `kullback_leibler`).
- `feature_contri`, `max_bin_by_feature` and `forcedbins_filename`.
- Cost-effective gradient boosting (CEGB).
- Forced splits (`forcedsplits_filename`).
- Parameter resets and training-set replacement (`reset_parameter`, `update(train_set=...)`).
- Plotting on matplotlib and `Booster.get_split_value_histogram`.

### Changed

- The binary Dataset format is now version 6 and stores raw feature values.

## [0.3.0] - 2026-10-02

### Added

- Additional metrics, `refit`, monotone and interaction constraints, DART, random forest and text-file loading (CSV, TSV, LibSVM).

## [0.2.1] - 2026-10-01

### Added

- Wheels for Python 3.11 through 3.14, tested before release.
- `LGBMModel`, `LGBMRegressor`, `LGBMClassifier` and `LGBMRanker`.
- Categorical features, `pred_early_stop`, `bagging_by_query` and `save_binary` with a versioned binary Dataset format.

### Changed

- Platform wheels are published so `pip install` does not need Rust.

### Fixed

- License files are included in the source distribution.

## [0.2.0] - 2026-10-01

### Added

- Multiclass and regression-family objectives and metrics.
- Bagging, GOSS, `feature_fraction`, `feature_fraction_bynode` and `extra_trees`.
- `cv()`, `CVBooster` and `Dataset.subset`.
- `dump_model`, `trees_to_dataframe` and `lower_bound`/`upper_bound`.
- Continued training (`init_model`).
- Arrow input (pyarrow, polars) and scipy sparse input.
- Ranking objectives (`lambdarank`, `rank_xendcg`) with `ndcg` and `map` metrics.
- `pred_contrib` (TreeSHAP).

## [0.1.0] - 2026-10-01

### Added

- First release: a pure-Rust port of LightGBM's core, a PyO3 Python package and the compatibility test harness against LightGBM 4.7.0.

[Unreleased]: https://github.com/zrluety/lightgbm-rust/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/zrluety/lightgbm-rust/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/zrluety/lightgbm-rust/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/zrluety/lightgbm-rust/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/zrluety/lightgbm-rust/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/zrluety/lightgbm-rust/releases/tag/v0.1.0
