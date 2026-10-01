use thiserror::Error;

/// All fallible core operations return this error type. The Python bindings
/// map it to `lightgbm_rust.LightGBMError`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum LgbmError {
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("invalid data: {0}")]
    InvalidData(String),
    #[error("model format error: {0}")]
    ModelFormat(String),
    /// A documented upstream feature that this implementation does not support yet.
    #[error("not supported by lightgbm-rust yet: {0}")]
    Unsupported(String),
    #[error("internal error: {0}")]
    Internal(String),
}

pub type Result<T> = std::result::Result<T, LgbmError>;
