//! Numeric constants with the exact values upstream uses.
//!
//! upstream: include/LightGBM/meta.h. Upstream declares these as `float`
//! literals and promotes them to `double`, so the f32 round-trip is required
//! for bit-identical arithmetic.

/// `kEpsilon = 1e-15f`, promoted to double.
pub const K_EPSILON: f64 = 1e-15_f32 as f64;

/// The logloss metrics' clamp value `-std::log(kEpsilon)`: `kEpsilon` is a
/// `float`, so this is the single-precision `logf`, promoted to double.
pub fn neg_log_epsilon() -> f64 {
    -(1e-15_f32.ln()) as f64
}
/// `kZeroThreshold = 1e-35f`, promoted to double.
pub const K_ZERO_THRESHOLD: f64 = 1e-35_f32 as f64;
/// `kMinScore = -inf`.
pub const K_MIN_SCORE: f64 = f64::NEG_INFINITY;
/// upstream: include/LightGBM/bin.h `kSparseThreshold`.
pub const K_SPARSE_THRESHOLD: f64 = 0.7;
