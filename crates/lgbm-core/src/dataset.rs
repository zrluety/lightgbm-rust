//! Binned training/validation data.
//!
//! Construction follows upstream `LGBM_DatasetCreateFromMats` ->
//! `DatasetLoader::ConstructFromSampleData` -> `Dataset::PushOneRow`
//! (src/c_api.cpp, src/io/dataset_loader.cpp): sample rows with the
//! upstream RNG, build one bin mapper per column from the non-zero sampled
//! values, drop trivial features, then bin every row.
//!
//! Ownership: the input matrix is only borrowed during construction. The
//! dataset owns its binned columns, labels, weights, and init scores; no
//! reference to caller memory is retained.

use rayon::prelude::*;

use crate::binning::{BinMapper, BinParams, BinType};
use crate::config::Config;
use crate::consts::K_ZERO_THRESHOLD;
use crate::error::{LgbmError, Result};
use crate::feature_groups::{SampleColumn, upstream_inner_order};
use crate::random::Random;

/// Run `f` on a dedicated rayon pool of `num_threads` threads, or on the
/// global pool when `num_threads <= 0`.
pub fn with_num_threads<R: Send>(num_threads: i32, f: impl FnOnce() -> R + Send) -> Result<R> {
    if num_threads <= 0 {
        return Ok(f());
    }
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads as usize)
        .build()
        .map_err(|e| LgbmError::Internal(format!("cannot build thread pool: {e}")))?;
    Ok(pool.install(f))
}

/// Borrowed dense feature values.
#[derive(Debug, Clone, Copy)]
pub enum DenseValues<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
}

/// A borrowed, contiguous dense matrix in row-major (C) or column-major (F) order.
#[derive(Debug, Clone, Copy)]
pub struct DenseMatrix<'a> {
    values: DenseValues<'a>,
    nrows: usize,
    ncols: usize,
    row_major: bool,
}

impl<'a> DenseMatrix<'a> {
    pub fn new(values: DenseValues<'a>, nrows: usize, ncols: usize, row_major: bool) -> Result<Self> {
        let len = match values {
            DenseValues::F32(v) => v.len(),
            DenseValues::F64(v) => v.len(),
        };
        if nrows.checked_mul(ncols) != Some(len) {
            return Err(LgbmError::InvalidData(format!(
                "matrix buffer has {len} values, expected {nrows} x {ncols}"
            )));
        }
        Ok(Self { values, nrows, ncols, row_major })
    }

    pub fn from_f64_row_major(values: &'a [f64], nrows: usize, ncols: usize) -> Result<Self> {
        Self::new(DenseValues::F64(values), nrows, ncols, true)
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let idx = if self.row_major { r * self.ncols + c } else { c * self.nrows + r };
        match self.values {
            DenseValues::F32(v) => v[idx] as f64,
            DenseValues::F64(v) => v[idx],
        }
    }

    /// Copy one row into `out` as f64 (upstream converts every value to double).
    #[inline]
    pub fn row_into(&self, r: usize, out: &mut [f64]) {
        for (c, o) in out.iter_mut().enumerate().take(self.ncols) {
            *o = self.get(r, c);
        }
    }
}

/// Bin indices of one feature for all rows, using the narrowest integer type.
#[derive(Debug, Clone)]
pub enum BinColumn {
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl BinColumn {
    #[inline]
    pub fn get(&self, i: usize) -> u32 {
        match self {
            BinColumn::U8(v) => v[i] as u32,
            BinColumn::U16(v) => v[i] as u32,
            BinColumn::U32(v) => v[i],
        }
    }

    fn build(num_bin: i32, n: usize, f: impl Fn(usize) -> u32 + Sync) -> Self {
        if num_bin <= 256 {
            BinColumn::U8((0..n).map(|i| f(i) as u8).collect())
        } else if num_bin <= 65536 {
            BinColumn::U16((0..n).map(|i| f(i) as u16).collect())
        } else {
            BinColumn::U32((0..n).map(&f).collect())
        }
    }
}

/// Labels, weights, and initial scores.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub label: Vec<f32>,
    pub weight: Option<Vec<f32>>,
    /// Class-major (`k * num_data + i`), like upstream.
    pub init_score: Option<Vec<f64>>,
}

/// upstream: `Common::AvoidInf(float)`.
#[inline]
pub fn avoid_inf_f32(x: f32) -> f32 {
    if x.is_nan() {
        0.0
    } else if x >= 1e38 {
        1e38
    } else if x <= -1e38 {
        -1e38
    } else {
        x
    }
}

/// upstream: `Common::AvoidInf(double)`.
#[inline]
pub fn avoid_inf_f64(x: f64) -> f64 {
    if x.is_nan() {
        0.0
    } else if x >= 1e300 {
        1e300
    } else if x <= -1e300 {
        -1e300
    } else {
        x
    }
}

impl Metadata {
    /// upstream: `Metadata::SetLabel` (NaN/inf are clamped by `AvoidInf`).
    pub fn set_label(&mut self, num_data: usize, label: &[f32]) -> Result<()> {
        if label.len() != num_data {
            return Err(LgbmError::InvalidData(format!(
                "Length of label ({}) is not same with #data ({num_data})",
                label.len()
            )));
        }
        self.label = label.iter().map(|&v| avoid_inf_f32(v)).collect();
        Ok(())
    }

    /// upstream: `Metadata::SetWeights`.
    pub fn set_weight(&mut self, num_data: usize, weight: Option<&[f32]>) -> Result<()> {
        self.weight = match weight {
            None => None,
            Some(w) if w.len() != num_data => {
                return Err(LgbmError::InvalidData(format!(
                    "Length of weights ({}) is not same with #data ({num_data})",
                    w.len()
                )))
            }
            Some(w) => Some(w.iter().map(|&v| avoid_inf_f32(v)).collect()),
        };
        Ok(())
    }

    /// upstream: `Metadata::SetInitScore`.
    pub fn set_init_score(&mut self, num_data: usize, init_score: Option<&[f64]>) -> Result<()> {
        self.init_score = match init_score {
            None => None,
            Some(s) if s.is_empty() || s.len() % num_data != 0 => {
                return Err(LgbmError::InvalidData(format!(
                    "Initial score size ({}) doesn't match data size ({num_data})",
                    s.len()
                )))
            }
            Some(s) => Some(s.iter().map(|&v| avoid_inf_f64(v)).collect()),
        };
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    num_data: usize,
    bin_mappers: Vec<BinMapper>,
    /// inner feature index -> real (column) index; only non-trivial features
    used_features: Vec<usize>,
    /// Upstream's inner index of each inner feature (see `feature_groups`).
    upstream_inner: Vec<usize>,
    real_to_inner: Vec<Option<usize>>,
    bins: Vec<BinColumn>,
    pub metadata: Metadata,
    feature_names: Vec<String>,
    warnings: Vec<String>,
}

/// Optional per-row fields supplied with the features.
#[derive(Debug, Clone, Default)]
pub struct DatasetFields<'a> {
    pub label: &'a [f32],
    pub weight: Option<&'a [f32]>,
    pub init_score: Option<&'a [f64]>,
    pub feature_names: Option<Vec<String>>,
    pub categorical_features: Vec<usize>,
}

impl Dataset {
    /// Construct a training dataset, building bin mappers from the data.
    /// Uses `cfg.num_threads` threads (all cores when <= 0).
    pub fn from_dense(mat: &DenseMatrix<'_>, fields: DatasetFields<'_>, cfg: &Config) -> Result<Self> {
        with_num_threads(cfg.num_threads, || Self::from_dense_impl(mat, fields, cfg))?
    }

    fn from_dense_impl(mat: &DenseMatrix<'_>, fields: DatasetFields<'_>, cfg: &Config) -> Result<Self> {
        let n = mat.nrows();
        let ncol = mat.ncols();
        Self::validate_fields(n, ncol, &fields)?;
        if !fields.categorical_features.is_empty() {
            return Err(LgbmError::Unsupported("categorical features".into()));
        }
        if n > i32::MAX as usize {
            return Err(LgbmError::InvalidData("more than 2^31-1 rows".into()));
        }

        // upstream: c_api.cpp CreateSampleIndices / SampleCount
        let sample_cnt = n.min(cfg.bin_construct_sample_cnt.max(0) as usize) as i32;
        let mut rand = Random::new(cfg.data_random_seed);
        let sample_indices = rand.sample(n as i32, sample_cnt);
        let total_sample_size = sample_indices.len();

        // upstream: dataset_loader.cpp ConstructFromSampleData
        let filter_cnt =
            (cfg.min_data_in_leaf as f64 * total_sample_size as f64 / n as f64) as i32;
        let params = BinParams {
            max_bin: cfg.max_bin,
            min_data_in_bin: cfg.min_data_in_bin,
            min_split_data: filter_cnt,
            pre_filter: cfg.feature_pre_filter,
            use_missing: cfg.use_missing,
            zero_as_missing: cfg.zero_as_missing,
        };
        let columns: Vec<SampleColumn> = (0..ncol)
            .into_par_iter()
            .map(|c| {
                let mut col = SampleColumn { indices: Vec::new(), values: Vec::new() };
                for (i, &r) in sample_indices.iter().enumerate() {
                    let v = mat.get(r as usize, c);
                    if v.abs() > K_ZERO_THRESHOLD || v.is_nan() {
                        col.indices.push(i as i32);
                        col.values.push(v);
                    }
                }
                col
            })
            .collect();
        let bin_mappers: Vec<BinMapper> = columns
            .par_iter()
            .map(|col| BinMapper::find_bin(&col.values, total_sample_size, BinType::Numerical, &params))
            .collect::<Result<_>>()?;

        let (feature_names, replaced) = match fields.feature_names.clone() {
            Some(names) => sanitize_feature_names(names)?,
            None => ((0..ncol).map(|i| format!("Column_{i}")).collect(), false),
        };
        let mut ds = Self::assemble(mat, fields, bin_mappers, feature_names)?;
        let explicit_bool = |k: &str| cfg.explicit.get(k).map(|v| matches!(v.to_ascii_lowercase().as_str(), "true" | "+"));
        ds.upstream_inner = upstream_inner_order(
            &ds.bin_mappers,
            &ds.used_features,
            &columns,
            total_sample_size,
            n,
            explicit_bool("enable_bundle").unwrap_or(true),
            explicit_bool("is_enable_sparse").unwrap_or(true),
        );
        if replaced {
            ds.warnings.push(FEATURE_NAME_SPACE_WARNING.into());
        }
        // upstream: Dataset::Construct
        if ds.used_features.is_empty() {
            ds.warnings.push(
                "There are no meaningful features which satisfy the provided configuration. \
                 Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing \
                 Dataset might resolve this warning."
                    .into(),
            );
        }
        Ok(ds)
    }

    /// Non-fatal diagnostics from construction (upstream `Log::Warning`).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Construct a validation dataset that reuses `reference`'s bin mappers.
    /// Uses `num_threads` threads (all cores when <= 0).
    ///
    /// upstream: `Dataset::CreateValid`.
    pub fn from_dense_with_reference(
        mat: &DenseMatrix<'_>,
        fields: DatasetFields<'_>,
        reference: &Dataset,
        num_threads: i32,
    ) -> Result<Self> {
        with_num_threads(num_threads, || Self::from_dense_with_reference_impl(mat, fields, reference))?
    }

    fn from_dense_with_reference_impl(
        mat: &DenseMatrix<'_>,
        fields: DatasetFields<'_>,
        reference: &Dataset,
    ) -> Result<Self> {
        Self::validate_fields(mat.nrows(), mat.ncols(), &fields)?;
        if mat.ncols() != reference.num_total_features() {
            return Err(LgbmError::InvalidData(format!(
                "validation data has {} features, training data has {}",
                mat.ncols(),
                reference.num_total_features()
            )));
        }
        let mut ds = Self::assemble(mat, fields, reference.bin_mappers.clone(), reference.feature_names.clone())?;
        ds.upstream_inner = reference.upstream_inner.clone();
        Ok(ds)
    }

    fn validate_fields(n: usize, ncol: usize, f: &DatasetFields<'_>) -> Result<()> {
        if n == 0 {
            return Err(LgbmError::InvalidData("dataset has no rows".into()));
        }
        if ncol == 0 {
            return Err(LgbmError::InvalidData("dataset has no feature columns".into()));
        }
        if f.label.len() != n {
            return Err(LgbmError::InvalidData(format!(
                "Length of label ({}) is not same with #data ({n})",
                f.label.len()
            )));
        }
        if let Some(w) = f.weight {
            if w.len() != n {
                return Err(LgbmError::InvalidData(format!(
                    "Length of weights ({}) is not same with #data ({n})",
                    w.len()
                )));
            }
        }
        if let Some(s) = f.init_score {
            if s.len() % n != 0 || s.is_empty() {
                return Err(LgbmError::InvalidData(format!(
                    "Initial score size ({}) doesn't match data size ({n})",
                    s.len()
                )));
            }
        }
        if let Some(names) = &f.feature_names {
            if names.len() != ncol {
                return Err(LgbmError::InvalidData(
                    "Size of feature_names error, should equal with total number of features".into(),
                ));
            }
        }
        Ok(())
    }

    fn assemble(
        mat: &DenseMatrix<'_>,
        fields: DatasetFields<'_>,
        bin_mappers: Vec<BinMapper>,
        feature_names: Vec<String>,
    ) -> Result<Self> {
        let n = mat.nrows();
        let used_features: Vec<usize> =
            (0..bin_mappers.len()).filter(|&c| !bin_mappers[c].is_trivial).collect();
        let mut real_to_inner = vec![None; bin_mappers.len()];
        for (inner, &real) in used_features.iter().enumerate() {
            real_to_inner[real] = Some(inner);
        }
        let bins: Vec<BinColumn> = used_features
            .par_iter()
            .map(|&c| {
                let m = &bin_mappers[c];
                BinColumn::build(m.num_bin, n, |r| m.value_to_bin(mat.get(r, c)))
            })
            .collect();
        let mut metadata = Metadata::default();
        metadata.set_label(n, fields.label)?;
        metadata.set_weight(n, fields.weight)?;
        metadata.set_init_score(n, fields.init_score)?;
        Ok(Self {
            num_data: n,
            bin_mappers,
            upstream_inner: (0..used_features.len()).collect(),
            used_features,
            real_to_inner,
            bins,
            metadata,
            feature_names,
            warnings: Vec::new(),
        })
    }

    pub fn num_data(&self) -> usize {
        self.num_data
    }

    /// Number of non-trivial (trainable) features.
    pub fn num_features(&self) -> usize {
        self.used_features.len()
    }

    pub fn num_total_features(&self) -> usize {
        self.bin_mappers.len()
    }

    pub fn real_feature_index(&self, inner: usize) -> usize {
        self.used_features[inner]
    }

    pub fn inner_feature_index(&self, real: usize) -> Option<usize> {
        self.real_to_inner[real]
    }

    /// The index upstream gives inner feature `inner` (its features are
    /// numbered in shuffled feature-group order).
    pub fn upstream_inner_index(&self, inner: usize) -> usize {
        self.upstream_inner[inner]
    }

    pub fn feature_bin_mapper(&self, inner: usize) -> &BinMapper {
        &self.bin_mappers[self.used_features[inner]]
    }

    pub fn bin_mapper_by_real(&self, real: usize) -> &BinMapper {
        &self.bin_mappers[real]
    }

    pub fn feature_bins(&self, inner: usize) -> &BinColumn {
        &self.bins[inner]
    }

    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns `true` when spaces were replaced (upstream logs a warning).
    pub fn set_feature_names(&mut self, names: Vec<String>) -> Result<bool> {
        if names.len() != self.num_total_features() {
            return Err(LgbmError::InvalidData(
                "Size of feature_names error, should equal with total number of features".into(),
            ));
        }
        let (names, replaced) = sanitize_feature_names(names)?;
        self.feature_names = names;
        Ok(replaced)
    }

    /// upstream: `Dataset::feature_infos` — `[min:max]` per feature, `none` if unused.
    pub fn feature_infos(&self) -> Vec<String> {
        self.bin_mappers
            .iter()
            .map(|m| if m.is_trivial { "none".to_string() } else { m.bin_info_string() })
            .collect()
    }

    pub fn label(&self) -> &[f32] {
        &self.metadata.label
    }

    pub fn weight(&self) -> Option<&[f32]> {
        self.metadata.weight.as_deref()
    }

    pub fn init_score(&self) -> Option<&[f64]> {
        self.metadata.init_score.as_deref()
    }

    /// Bin index of each row for the given inner feature (for diagnostics/tests).
    pub fn bin_indices(&self, inner: usize) -> Vec<u32> {
        (0..self.num_data).map(|i| self.bins[inner].get(i)).collect()
    }

    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }

    /// Whether `other` was binned with identical mappers (NaN bounds compare equal).
    pub fn same_bins_as(&self, other: &Dataset) -> bool {
        self.bin_mappers.len() == other.bin_mappers.len()
            && self.bin_mappers.iter().zip(&other.bin_mappers).all(|(a, b)| {
                a.num_bin == b.num_bin
                    && a.missing_type == b.missing_type
                    && a.is_trivial == b.is_trivial
                    && a.default_bin == b.default_bin
                    && a.most_freq_bin == b.most_freq_bin
                    && a.bin_upper_bound.len() == b.bin_upper_bound.len()
                    && a.bin_upper_bound.iter().zip(&b.bin_upper_bound).all(|(x, y)| x.to_bits() == y.to_bits())
            })
    }
}

/// Message upstream logs when [`sanitize_feature_names`] replaced spaces.
pub const FEATURE_NAME_SPACE_WARNING: &str = "Found whitespace in feature_names, replace with underlines";

/// upstream: include/LightGBM/dataset.h `Dataset::set_feature_names`.
/// Spaces become `_`; JSON special characters and duplicates are rejected.
pub fn sanitize_feature_names(mut names: Vec<String>) -> Result<(Vec<String>, bool)> {
    let mut seen = std::collections::HashSet::new();
    let mut replaced = false;
    for name in names.iter_mut() {
        if name.chars().any(|c| matches!(c, '"' | ',' | ':' | '[' | ']' | '{' | '}')) {
            return Err(LgbmError::InvalidData("Do not support special JSON characters in feature name.".into()));
        }
        if name.contains(' ') {
            replaced = true;
            *name = name.replace(' ', "_");
        }
        if !seen.insert(name.clone()) {
            return Err(LgbmError::InvalidData(format!("Feature ({name}) appears more than one time.")));
        }
    }
    Ok((names, replaced))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivial_columns_are_dropped() {
        let n = 100;
        let mut data = Vec::new();
        for i in 0..n {
            data.push(i as f64);
            data.push(1.0); // constant
        }
        let label = vec![0.0f32; n];
        let mat = DenseMatrix::from_f64_row_major(&data, n, 2).unwrap();
        let ds = Dataset::from_dense(
            &mat,
            DatasetFields { label: &label, ..Default::default() },
            &Config::default(),
        )
        .unwrap();
        assert_eq!(ds.num_total_features(), 2);
        assert_eq!(ds.num_features(), 1);
        assert_eq!(ds.real_feature_index(0), 0);
        assert_eq!(ds.feature_infos()[1], "none");
    }

    #[test]
    fn column_major_matches_row_major() {
        let n = 50;
        let rm: Vec<f64> = (0..n * 3).map(|i| ((i * 7919) % 101) as f64).collect();
        let mut cm = vec![0.0; n * 3];
        for r in 0..n {
            for c in 0..3 {
                cm[c * n + r] = rm[r * 3 + c];
            }
        }
        let label = vec![0.0f32; n];
        let cfg = Config::from_pairs([("min_data_in_leaf", "1"), ("min_data_in_bin", "1")]).unwrap();
        let a = Dataset::from_dense(
            &DenseMatrix::new(DenseValues::F64(&rm), n, 3, true).unwrap(),
            DatasetFields { label: &label, ..Default::default() },
            &cfg,
        )
        .unwrap();
        let b = Dataset::from_dense(
            &DenseMatrix::new(DenseValues::F64(&cm), n, 3, false).unwrap(),
            DatasetFields { label: &label, ..Default::default() },
            &cfg,
        )
        .unwrap();
        assert!(a.same_bins_as(&b));
        for f in 0..a.num_features() {
            assert_eq!(a.bin_indices(f), b.bin_indices(f));
        }
    }

    #[test]
    fn label_length_is_validated() {
        let data = vec![0.0; 10];
        let mat = DenseMatrix::from_f64_row_major(&data, 5, 2).unwrap();
        let e = Dataset::from_dense(
            &mat,
            DatasetFields { label: &[0.0; 4], ..Default::default() },
            &Config::default(),
        )
        .unwrap_err();
        assert!(matches!(e, LgbmError::InvalidData(_)));
    }
}
