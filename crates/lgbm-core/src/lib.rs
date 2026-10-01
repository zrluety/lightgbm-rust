//! `lgbm-core`: a pure-Rust implementation of LightGBM's gradient boosting.
//!
//! This crate does not link to or call the upstream C++ library. Algorithms
//! are ported from LightGBM v4.7.0 (commit 8f7036f, MIT license; see NOTICE),
//! and source comments of the form `upstream: <path>` point to the code each
//! piece follows.
//!
//! Layering (no module depends on a later one):
//! `consts`, `error`, `fmt`, `random`, `arrow` -> `config` -> `binning` ->
//! `dataset` -> `objective`, `metric` -> `tree` -> `histogram` -> `learner` ->
//! `boosting` -> `predict`, `io`.

pub mod array_args;
pub mod arrow;
pub mod binning;
pub mod boosting;
pub mod config;
pub mod consts;
pub mod dataset;
pub mod dcg;
pub mod error;
pub mod feature_groups;
pub mod fmt;
pub mod histogram;
pub mod io;
pub mod learner;
pub mod matrix;
pub mod metric;
pub mod multi_val_bin;
pub mod objective;
pub mod predict;
pub mod random;
pub mod sample_strategy;
pub mod threading;
pub mod tree;

pub use boosting::{EvalResult, Gbdt, PredictKind};
pub use config::Config;
pub use dataset::{Dataset, DatasetFields, DenseMatrix, DenseValues};
pub use matrix::{Matrix, SparseIndptr, SparseMatrix};
pub use error::{LgbmError, Result};

/// Upstream version this crate tracks.
pub const UPSTREAM_VERSION: &str = "4.7.0";
pub const UPSTREAM_COMMIT: &str = "8f7036f03627054d5a54a6f965b13f4b9ff2cb63";
