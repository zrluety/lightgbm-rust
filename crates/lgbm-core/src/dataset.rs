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

use crate::bin::Bin;
use crate::binning::{BinMapper, BinParams, BinType};
use crate::config::Config;
use crate::error::{LgbmError, Result};
use crate::feature_group::{FeatureBins, FeatureGroup, GroupsBuilder, SubFeature};
use crate::feature_groups::{upstream_groups, GroupSpec, SampleColumn};
use crate::matrix::Matrix;
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
    Ok(crate::log::install(&pool, f))
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

/// How a new dataset's features are grouped.
pub(crate) enum Layout<'a> {
    /// A training dataset's groups (upstream `Dataset::Construct`).
    Train(Vec<GroupSpec>),
    /// One group per feature in the reference's upstream inner order, given
    /// as each inner feature's upstream index (upstream `CreateValid`).
    Valid(&'a [usize]),
}

/// Labels, weights, initial scores, and ranking query/position data.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    pub label: Vec<f32>,
    pub weight: Option<Vec<f32>>,
    /// Class-major (`k * num_data + i`), like upstream.
    pub init_score: Option<Vec<f64>>,
    /// `num_queries + 1` row offsets (upstream `query_boundaries_`).
    pub query_boundaries: Option<Vec<i32>>,
    /// Mean row weight per query. Like upstream, these are only recomputed
    /// when both weights and queries are set, so they can be stale.
    pub query_weights: Option<Vec<f32>>,
    /// Dense position id of each row (first-seen order).
    pub positions: Option<Vec<i32>>,
    /// Raw position value of each dense id, as a string.
    pub position_ids: Vec<String>,
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
        if self.weight.is_some() {
            self.calculate_query_weights();
        }
        Ok(())
    }

    /// upstream: `Metadata::SetQuery`; `counts` are the query sizes.
    pub fn set_query(&mut self, num_data: usize, counts: Option<&[i32]>) -> Result<()> {
        let counts = match counts {
            Some(c) if !c.is_empty() => c,
            _ => {
                self.query_boundaries = None;
                return Ok(());
            }
        };
        let sum: i64 = counts.iter().map(|&c| c as i64).sum();
        if sum != num_data as i64 {
            // upstream passes the arguments in this order
            return Err(LgbmError::InvalidData(format!(
                "Sum of query counts ({num_data}) differs from the length of #data ({sum})"
            )));
        }
        let mut b = Vec::with_capacity(counts.len() + 1);
        b.push(0i32);
        for &c in counts {
            b.push(b[b.len() - 1] + c);
        }
        self.query_boundaries = Some(b);
        self.calculate_query_weights();
        Ok(())
    }

    /// upstream: `Metadata::SetPosition`.
    pub fn set_position(&mut self, num_data: usize, positions: Option<&[i32]>) -> Result<()> {
        let positions = match positions {
            Some(p) if !p.is_empty() => p,
            _ => {
                self.positions = None;
                // upstream keeps the ids, which would make the ranking
                // objectives read a missing position array
                self.position_ids.clear();
                return Ok(());
            }
        };
        if positions.len() != num_data {
            return Err(LgbmError::InvalidData(format!(
                "Positions size ({}) doesn't match data size ({num_data})",
                positions.len()
            )));
        }
        if self.positions.is_some() {
            crate::log::warning("Overwriting positions in dataset.");
        }
        let mut ids = std::collections::HashMap::new();
        self.position_ids.clear();
        let mut dense = Vec::with_capacity(num_data);
        for &p in positions {
            let next = ids.len() as i32;
            let id = *ids.entry(p).or_insert_with(|| {
                self.position_ids.push(p.to_string());
                next
            });
            dense.push(id);
        }
        crate::log::debug(&format!("number of unique positions found = {}", self.position_ids.len()));
        self.positions = Some(dense);
        Ok(())
    }

    /// upstream: `Metadata::CalculateQueryWeights`.
    fn calculate_query_weights(&mut self) {
        let (Some(w), Some(b)) = (&self.weight, &self.query_boundaries) else { return };
        let qw = b
            .windows(2)
            .map(|q| {
                let mut s = 0.0f32;
                for &x in &w[q[0] as usize..q[1] as usize] {
                    s += x;
                }
                s / (q[1] - q[0]) as f32
            })
            .collect();
        self.query_weights = Some(qw);
    }

    pub fn num_queries(&self) -> usize {
        self.query_boundaries.as_ref().map_or(0, |b| b.len() - 1)
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

/// The construction parameters upstream records in a dataset
/// (`max_bin_`, `min_data_in_bin_`, ...) and checks when a binary file is
/// loaded (`DatasetLoader::CheckDataset`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinConstructConfig {
    pub max_bin: i32,
    pub min_data_in_bin: i32,
    pub bin_construct_sample_cnt: i32,
    pub use_missing: bool,
    pub zero_as_missing: bool,
}

impl BinConstructConfig {
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            max_bin: cfg.max_bin,
            min_data_in_bin: cfg.min_data_in_bin,
            bin_construct_sample_cnt: cfg.bin_construct_sample_cnt,
            use_missing: cfg.use_missing,
            zero_as_missing: cfg.zero_as_missing,
        }
    }
}

impl Default for BinConstructConfig {
    fn default() -> Self {
        Self::from_config(&Config::default())
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub(crate) num_data: usize,
    pub(crate) bin_mappers: Vec<BinMapper>,
    /// inner feature index -> real (column) index; only non-trivial features
    pub(crate) used_features: Vec<usize>,
    /// Upstream's inner index of each inner feature (see `feature_groups`).
    pub(crate) upstream_inner: Vec<usize>,
    pub(crate) real_to_inner: Vec<Option<usize>>,
    /// upstream `feature_groups_`, in upstream group order.
    pub(crate) groups: Vec<FeatureGroup>,
    /// (group, sub-feature) of each inner feature.
    pub(crate) feature_loc: Vec<(u32, u32)>,
    pub metadata: Metadata,
    pub(crate) feature_names: Vec<String>,
    pub(crate) bin_config: BinConstructConfig,
    /// upstream `max_bin_by_feature_` (empty unless given at construction).
    pub(crate) max_bin_by_feature: Vec<i32>,
    /// upstream `forced_bin_bounds_`: one (possibly empty) list per column.
    pub(crate) forced_bin_bounds: Vec<Vec<f64>>,
    /// upstream `data_filename_`: the file this dataset was loaded from.
    pub(crate) data_filename: Option<String>,
    /// upstream `label_idx_`: the label's column in the text file (0 otherwise).
    pub(crate) label_idx: i32,
    /// upstream `raw_data_` (kept iff `has_raw_`, i.e. built with
    /// `linear_tree`): `f32` feature values per inner feature, empty for
    /// categorical features.
    pub(crate) raw: Option<Vec<Vec<f32>>>,
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
        Self::from_matrix(&Matrix::Dense(*mat), fields, cfg)
    }

    /// Like [`Dataset::from_dense`], for dense or sparse (CSR/CSC) input.
    pub fn from_matrix(mat: &Matrix<'_>, fields: DatasetFields<'_>, cfg: &Config) -> Result<Self> {
        with_num_threads(cfg.num_threads, || Self::from_matrix_impl(mat, fields, cfg))?
    }

    fn from_matrix_impl(mat: &Matrix<'_>, fields: DatasetFields<'_>, cfg: &Config) -> Result<Self> {
        let n = mat.nrows();
        let ncol = mat.ncols();
        Self::validate_fields(n, ncol, &fields)?;
        let mut is_categorical = vec![false; ncol];
        for c in cfg.categorical_indices()? {
            if c >= 0 && (c as usize) < ncol {
                is_categorical[c as usize] = true;
            }
        }
        for &c in &fields.categorical_features {
            if c < ncol {
                is_categorical[c] = true;
            }
        }
        if n > i32::MAX as usize {
            return Err(LgbmError::InvalidData("more than 2^31-1 rows".into()));
        }
        // upstream: dataset_loader.cpp ConstructFromSampleData
        if is_categorical
            .iter()
            .enumerate()
            .any(|(c, &cat)| cat && cfg.monotone_constraints.get(c).is_some_and(|&m| m != 0))
        {
            return Err(LgbmError::InvalidParameter(
                "The output cannot be monotone with respect to categorical features".into(),
            ));
        }

        // upstream: c_api.cpp CreateSampleIndices / SampleCount
        let sample_cnt = n.min(cfg.bin_construct_sample_cnt.max(0) as usize) as i32;
        let mut rand = Random::new(cfg.data_random_seed);
        let sample_indices = rand.sample(n as i32, sample_cnt);
        let total_sample_size = sample_indices.len();

        let columns: Vec<SampleColumn> = mat.sample_columns(&sample_indices);
        let found = find_bin_mappers(
            &columns,
            total_sample_size,
            n,
            &is_categorical,
            &vec![false; ncol],
            cfg,
            "static_cast<size_t>(num_col)",
        )?;

        let (feature_names, replaced) = match fields.feature_names.clone() {
            Some(names) => sanitize_feature_names(names)?,
            None => ((0..ncol).map(|i| format!("Column_{i}")).collect(), false),
        };
        let layout = train_layout(&found.bin_mappers, &columns, total_sample_size, n, cfg);
        let mut ds = Self::assemble(mat, fields, found.bin_mappers, feature_names, layout)?;
        if cfg.linear_tree {
            ds.set_raw_from(mat);
        }
        ds.forced_bin_bounds = found.forced_bin_bounds;
        ds.finish_construct(cfg, replaced);
        Ok(ds)
    }

    /// Recorded parameters and warnings of a freshly binned dataset.
    ///
    /// upstream: the end of `Dataset::Construct`.
    pub(crate) fn finish_construct(&mut self, cfg: &Config, names_replaced: bool) {
        self.bin_config = BinConstructConfig::from_config(cfg);
        self.max_bin_by_feature = cfg.max_bin_by_feature.clone();
        if names_replaced {
            crate::log::warning(FEATURE_NAME_SPACE_WARNING);
        }
        if self.used_features.is_empty() {
            crate::log::warning(
                "There are no meaningful features which satisfy the provided configuration. \
                 Decreasing Dataset parameters min_data_in_bin or min_data_in_leaf and re-constructing \
                 Dataset might resolve this warning.",
            );
        }
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
        Self::from_matrix_with_reference(&Matrix::Dense(*mat), fields, reference, num_threads)
    }

    /// Like [`Dataset::from_dense_with_reference`], for dense or sparse input.
    pub fn from_matrix_with_reference(
        mat: &Matrix<'_>,
        fields: DatasetFields<'_>,
        reference: &Dataset,
        num_threads: i32,
    ) -> Result<Self> {
        with_num_threads(num_threads, || Self::from_matrix_with_reference_impl(mat, fields, reference))?
    }

    fn from_matrix_with_reference_impl(
        mat: &Matrix<'_>,
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
        if reference.has_raw() && mat.is_csc() {
            return Err(LgbmError::Unsupported(
                "CSC validation data for a Dataset with linear_tree (upstream writes its raw values without \
                 allocating them)"
                    .into(),
            ));
        }
        let mut ds = Self::assemble(
            mat,
            fields,
            reference.bin_mappers.clone(),
            reference.feature_names.clone(),
            Layout::Valid(&reference.upstream_inner),
        )?;
        if reference.has_raw() {
            ds.set_raw_from(mat);
        }
        ds.bin_config = reference.bin_config;
        ds.forced_bin_bounds = reference.forced_bin_bounds.clone();
        ds.label_idx = reference.label_idx;
        Ok(ds)
    }

    /// Rows `used` (sorted, in range) of this dataset, sharing its bin mappers.
    ///
    /// upstream: c_api.cpp `LGBM_DatasetGetSubset`, `Dataset::CopySubrow`,
    /// `Metadata::Init(const Metadata&, const data_size_t*, data_size_t)`.
    pub fn subset(&self, used: &[i32]) -> Result<Self> {
        if used.is_empty() {
            return Err(LgbmError::InvalidParameter("Check failed: (num_used_row_indices) > (0)".into()));
        }
        check_elements_interval_closed(used, 0, self.num_data as i32 - 1, "Used indices of subset", |v| v.to_string())?;
        if used.windows(2).any(|w| w[0] > w[1]) {
            return Err(LgbmError::InvalidData("used_row_indices should be sorted in Subset".into()));
        }
        let n = used.len();
        let used_u32: Vec<u32> = used.iter().map(|&i| i as u32).collect();
        let groups = self.groups.par_iter().map(|g| g.copy_subrow(&used_u32)).collect();
        let pick_f32 = |v: &[f32]| used.iter().map(|&i| v[i as usize]).collect::<Vec<_>>();
        let init_score = self.metadata.init_score.as_ref().map(|s| {
            let k = s.len() / self.num_data;
            let mut out = Vec::with_capacity(n * k);
            for c in 0..k {
                out.extend(used.iter().map(|&i| s[c * self.num_data + i as usize]));
            }
            out
        });
        Ok(Self {
            num_data: n,
            bin_mappers: self.bin_mappers.clone(),
            used_features: self.used_features.clone(),
            upstream_inner: self.upstream_inner.clone(),
            real_to_inner: self.real_to_inner.clone(),
            groups,
            feature_loc: self.feature_loc.clone(),
            metadata: Metadata {
                label: pick_f32(&self.metadata.label),
                weight: self.metadata.weight.as_deref().map(pick_f32),
                init_score,
                // upstream iterates over the new Metadata's `num_queries_`,
                // which is still 0, so a grouped fullset yields `[0]` until
                // the caller sets the groups (the Python package does)
                query_boundaries: self.metadata.query_boundaries.as_ref().map(|_| vec![0]),
                ..Default::default()
            },
            feature_names: self.feature_names.clone(),
            bin_config: self.bin_config,
            max_bin_by_feature: Vec::new(),
            forced_bin_bounds: self.forced_bin_bounds.clone(),
            data_filename: None,
            label_idx: self.label_idx,
            raw: self.raw.as_ref().map(|r| {
                r.iter().map(|col| if col.is_empty() { Vec::new() } else { pick_f32(col) }).collect()
            }),
        })
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

    pub(crate) fn assemble(
        mat: &Matrix<'_>,
        fields: DatasetFields<'_>,
        bin_mappers: Vec<BinMapper>,
        feature_names: Vec<String>,
        layout: Layout<'_>,
    ) -> Result<Self> {
        let n = mat.nrows();
        let mut ds = Self::unfilled(n, bin_mappers, feature_names, layout);
        ds.fill_groups(|b, d| mat.push_into(b, &d.used_features, &d.bin_mappers, &d.real_to_inner));
        ds.metadata.set_label(n, fields.label)?;
        ds.metadata.set_weight(n, fields.weight)?;
        ds.metadata.set_init_score(n, fields.init_score)?;
        Ok(ds)
    }

    /// A dataset of `n` rows whose feature groups hold no bins yet (see
    /// [`Dataset::fill_groups`]), with empty metadata.
    pub(crate) fn unfilled(n: usize, bin_mappers: Vec<BinMapper>, feature_names: Vec<String>, layout: Layout<'_>) -> Self {
        let used_features: Vec<usize> = (0..bin_mappers.len()).filter(|&c| !bin_mappers[c].is_trivial).collect();
        let mut real_to_inner = vec![None; bin_mappers.len()];
        for (inner, &real) in used_features.iter().enumerate() {
            real_to_inner[real] = Some(inner);
        }
        let sub = |inner: usize| SubFeature::of(inner, &bin_mappers[used_features[inner]]);
        let groups: Vec<FeatureGroup> = match layout {
            Layout::Train(specs) => specs
                .iter()
                .enumerate()
                .map(|(g, s)| {
                    let subs = s.features.iter().map(|&r| sub(real_to_inner[r].expect("grouped features are used"))).collect();
                    FeatureGroup::new(subs, s.is_multi_val, g)
                })
                .collect(),
            Layout::Valid(upstream_inner) => {
                let mut by_upstream = vec![0usize; upstream_inner.len()];
                for (inner, &u) in upstream_inner.iter().enumerate() {
                    by_upstream[u] = inner;
                }
                by_upstream.into_iter().map(|inner| FeatureGroup::new_single(sub(inner))).collect()
            }
        };
        let mut feature_loc = vec![(0u32, 0u32); used_features.len()];
        let mut upstream_inner = vec![0usize; used_features.len()];
        let mut k = 0;
        for (g, grp) in groups.iter().enumerate() {
            for (s, sf) in grp.subs.iter().enumerate() {
                feature_loc[sf.inner] = (g as u32, s as u32);
                upstream_inner[sf.inner] = k;
                k += 1;
            }
        }
        Self {
            num_data: n,
            forced_bin_bounds: vec![Vec::new(); bin_mappers.len()],
            bin_mappers,
            upstream_inner,
            used_features,
            real_to_inner,
            groups,
            feature_loc,
            metadata: Metadata::default(),
            feature_names,
            bin_config: BinConstructConfig::default(),
            max_bin_by_feature: Vec::new(),
            data_filename: None,
            label_idx: 0,
            raw: None,
        }
    }

    /// upstream `Dataset::has_raw`.
    pub fn has_raw(&self) -> bool {
        self.raw.is_some()
    }

    /// Raw values of inner feature `inner` (upstream `raw_index`); empty
    /// without raw storage or for a categorical feature.
    pub fn raw(&self, inner: usize) -> &[f32] {
        self.raw.as_ref().map_or(&[][..], |r| r[inner].as_slice())
    }

    /// Store the raw values of `mat` (see [`Matrix::raw_values`]).
    fn set_raw_from(&mut self, mat: &Matrix<'_>) {
        self.raw = Some(mat.raw_values(&self.used_features, &self.bin_mappers, &self.real_to_inner));
    }

    /// Fill the bins of an [`Dataset::unfilled`] dataset: `push(builder, self)`
    /// pushes every row.
    pub(crate) fn fill_groups(&mut self, push: impl FnOnce(&mut GroupsBuilder<'_>, &Self)) {
        let groups = std::mem::take(&mut self.groups);
        let this: &Self = self;
        let mappers = this.used_features.iter().map(|&r| &this.bin_mappers[r]).collect();
        let mut b = GroupsBuilder::new(groups, mappers, this.num_data);
        push(&mut b, this);
        let groups = b.finish();
        self.groups = groups;
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

    pub fn num_feature_groups(&self) -> usize {
        self.groups.len()
    }

    /// upstream `feature_groups_`, in upstream group order.
    pub fn feature_groups(&self) -> &[FeatureGroup] {
        &self.groups
    }

    /// The multi-value group, if any.
    pub fn multi_val_group(&self) -> Option<usize> {
        self.groups.iter().position(|g| g.is_multi_val)
    }

    /// (group, sub-feature) of inner feature `inner`.
    pub fn feature_location(&self, inner: usize) -> (usize, usize) {
        let (g, s) = self.feature_loc[inner];
        (g as usize, s as usize)
    }

    pub fn feature_bin_mapper(&self, inner: usize) -> &BinMapper {
        &self.bin_mappers[self.used_features[inner]]
    }

    pub fn bin_mapper_by_real(&self, real: usize) -> &BinMapper {
        &self.bin_mappers[real]
    }

    /// The stored bins of inner feature `inner`.
    pub fn feature_bins(&self, inner: usize) -> FeatureBins<'_> {
        let (g, s) = self.feature_location(inner);
        self.groups[g].feature_bins(s)
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
        let mut c = self.feature_bins(inner).cursor(0);
        (0..self.num_data).map(|i| c.get(i)).collect()
    }

    /// Heap bytes of the stored bins.
    pub fn bin_heap_bytes(&self) -> usize {
        self.groups.iter().flat_map(|g| &g.bins).map(Bin::heap_bytes).sum()
    }

    pub fn bin_mappers(&self) -> &[BinMapper] {
        &self.bin_mappers
    }

    /// upstream `max_bin_by_feature_` (empty when not set at construction).
    pub fn max_bin_by_feature(&self) -> &[i32] {
        &self.max_bin_by_feature
    }

    /// upstream `forced_bin_bounds_`, one list per column.
    pub fn forced_bin_bounds(&self) -> &[Vec<f64>] {
        &self.forced_bin_bounds
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
                    && a.bin_type == b.bin_type
                    && a.bin_2_categorical == b.bin_2_categorical
                    && a.bin_upper_bound.len() == b.bin_upper_bound.len()
                    && a.bin_upper_bound.iter().zip(&b.bin_upper_bound).all(|(x, y)| x.to_bits() == y.to_bits())
            })
    }
}

/// The feature groups of a training dataset with these mappers.
///
/// upstream: `Dataset::Construct` (`FastFeatureBundling`).
pub(crate) fn train_layout(
    bin_mappers: &[BinMapper],
    columns: &[SampleColumn],
    total_sample_size: usize,
    num_data: usize,
    cfg: &Config,
) -> Layout<'static> {
    let used: Vec<usize> = (0..bin_mappers.len()).filter(|&c| !bin_mappers[c].is_trivial).collect();
    let explicit_bool = |k: &str| cfg.explicit.get(k).map(|v| matches!(v.to_ascii_lowercase().as_str(), "true" | "+"));
    Layout::Train(upstream_groups(
        bin_mappers,
        &used,
        columns,
        total_sample_size,
        num_data,
        explicit_bool("enable_bundle").unwrap_or(true),
        explicit_bool("is_enable_sparse").unwrap_or(true),
    ))
}

/// One bin mapper per column from the sampled values; `skip` columns get a
/// trivial mapper (upstream leaves them null).
///
/// upstream: dataset_loader.cpp `ConstructFromSampleData` /
/// `ConstructBinMappersFromTextData` and `CheckCategoricalFeatureNumBin`.
pub(crate) fn find_bin_mappers(
    columns: &[SampleColumn],
    total_sample_size: usize,
    num_data: usize,
    is_categorical: &[bool],
    skip: &[bool],
    cfg: &Config,
    num_total_features_expr: &str,
) -> Result<FoundBins> {
    check_max_bin_by_feature(cfg, columns.len(), num_total_features_expr)?;
    let forced_bin_bounds = get_forced_bins(&cfg.forcedbins_filename, columns.len(), is_categorical)?;
    let filter_cnt = (cfg.min_data_in_leaf as f64 * total_sample_size as f64 / num_data as f64) as i32;
    let max_bin_of = |c: usize| if cfg.max_bin_by_feature.is_empty() { cfg.max_bin } else { cfg.max_bin_by_feature[c] };
    let found: Vec<(BinMapper, Vec<String>)> = columns
        .par_iter()
        .enumerate()
        .map(|(c, col)| {
            if skip[c] {
                return Ok((BinMapper::default(), Vec::new()));
            }
            let bin_type = if is_categorical[c] { BinType::Categorical } else { BinType::Numerical };
            let params = BinParams {
                max_bin: max_bin_of(c),
                min_data_in_bin: cfg.min_data_in_bin,
                min_split_data: filter_cnt,
                pre_filter: cfg.feature_pre_filter,
                use_missing: cfg.use_missing,
                zero_as_missing: cfg.zero_as_missing,
            };
            let mut w = Vec::new();
            BinMapper::find_bin_logged(&col.values, total_sample_size, bin_type, &params, &forced_bin_bounds[c], &mut w)
                .map(|m| (m, w))
        })
        .collect::<Result<_>>()?;
    let mut bin_mappers = Vec::with_capacity(found.len());
    for (m, w) in found {
        crate::log::warnings(w);
        bin_mappers.push(m);
    }
    if bin_mappers
        .iter()
        .enumerate()
        .any(|(c, m)| m.bin_type == BinType::Categorical && m.num_bin > max_bin_of(c))
    {
        crate::log::warning("Categorical features with more bins than the configured maximum bin number found.");
        crate::log::warning(
            "For categorical features, max_bin and max_bin_by_feature may be ignored with a large number of categories.",
        );
    }
    Ok(FoundBins { bin_mappers, forced_bin_bounds })
}

/// Output of [`find_bin_mappers`].
pub(crate) struct FoundBins {
    pub bin_mappers: Vec<BinMapper>,
    pub forced_bin_bounds: Vec<Vec<f64>>,
}

/// upstream's `CHECK_EQ`/`CHECK_GT` on `max_bin_by_feature`;
/// `num_total_features_expr` is the left operand as upstream spells it.
pub(crate) fn check_max_bin_by_feature(cfg: &Config, num_total_features: usize, num_total_features_expr: &str) -> Result<()> {
    let m = &cfg.max_bin_by_feature;
    if m.is_empty() {
        return Ok(());
    }
    if m.len() != num_total_features {
        return Err(LgbmError::InvalidParameter(format!(
            "Check failed: ({num_total_features_expr}) == (config_.max_bin_by_feature.size())"
        )));
    }
    if m.iter().min().is_some_and(|&v| v <= 1) {
        return Err(LgbmError::InvalidParameter(
            "Check failed: (*(std::min_element(config_.max_bin_by_feature.begin(), config_.max_bin_by_feature.end()))) > (1)"
                .into(),
        ));
    }
    Ok(())
}

/// upstream: dataset_loader.cpp `DatasetLoader::GetForcedBins` (json11
/// semantics: missing or mistyped keys read as 0 / empty). A negative
/// `feature` is undefined behavior upstream; it is ignored here.
pub(crate) fn get_forced_bins(
    path: &str,
    num_total_features: usize,
    is_categorical: &[bool],
) -> Result<Vec<Vec<f64>>> {
    let mut forced_bins = vec![Vec::new(); num_total_features];
    if path.is_empty() {
        return Ok(forced_bins);
    }
    let Ok(text) = std::fs::read_to_string(path) else {
        crate::log::warning(&format!("Could not open {path}. Will ignore."));
        return Ok(forced_bins);
    };
    let json: serde_json::Value = serde_json::from_str(&text).unwrap_or(serde_json::Value::Null);
    let serde_json::Value::Array(items) = json else {
        return Err(LgbmError::InvalidParameter("Check failed: forced_bins_json.is_array()".into()));
    };
    for item in &items {
        // json11 int_value(): static_cast<int> of the double
        let feature_num = item.get("feature").and_then(|v| v.as_f64()).map_or(0, |v| v as i32);
        if feature_num as i64 >= num_total_features as i64 {
            return Err(LgbmError::InvalidParameter("Check failed: (feature_num) < (num_total_features)".into()));
        }
        if feature_num < 0 {
            continue;
        }
        let f = feature_num as usize;
        if is_categorical.get(f).copied().unwrap_or(false) {
            crate::log::warning(&format!("Feature {feature_num} is categorical. Will ignore forced bins for this feature."));
        } else if let Some(serde_json::Value::Array(bounds)) = item.get("bin_upper_bound") {
            forced_bins[f].extend(bounds.iter().map(|b| b.as_f64().unwrap_or(0.0)));
        }
    }
    for b in &mut forced_bins {
        b.dedup();
    }
    Ok(forced_bins)
}

/// upstream: utils/common.h `CheckElementsIntervalClosed` (same pairwise scan,
/// so the reported element matches). `show` formats a value as upstream's
/// `std::ostream` does.
pub(crate) fn check_elements_interval_closed<T: PartialOrd + Copy>(
    y: &[T],
    ymin: T,
    ymax: T,
    caller: &str,
    show: impl Fn(T) -> String,
) -> Result<()> {
    let fatal = |i: usize| {
        Err(LgbmError::InvalidData(format!(
            "[{caller}]: does not tolerate element [#{i} = {}] outside [{}, {}]",
            show(y[i]),
            show(ymin),
            show(ymax)
        )))
    };
    let mut i = 1;
    while i < y.len() {
        let (a, b) = (y[i - 1], y[i]);
        if a < b {
            if a < ymin {
                return fatal(i - 1);
            } else if b > ymax {
                return fatal(i);
            }
        } else if a > ymax {
            return fatal(i - 1);
        } else if b < ymin {
            return fatal(i);
        }
        i += 2;
    }
    if y.len() % 2 == 1 {
        let last = y.len() - 1;
        if y[last] < ymin || y[last] > ymax {
            return fatal(last);
        }
    }
    Ok(())
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
    fn subset_copies_rows_and_metadata() {
        let n = 40;
        let data: Vec<f64> = (0..n * 2).map(|i| ((i * 37) % 23) as f64).collect();
        let label: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let weight: Vec<f32> = (0..n).map(|i| 1.0 + i as f32).collect();
        let init: Vec<f64> = (0..2 * n).map(|i| i as f64).collect();
        let cfg = Config::from_pairs([("min_data_in_leaf", "1"), ("min_data_in_bin", "1")]).unwrap();
        let mat = DenseMatrix::from_f64_row_major(&data, n, 2).unwrap();
        let fields =
            DatasetFields { label: &label, weight: Some(&weight), init_score: Some(&init), ..Default::default() };
        let ds = Dataset::from_dense(&mat, fields, &cfg).unwrap();
        let used = [1, 5, 6, 39];
        let sub = ds.subset(&used).unwrap();
        assert_eq!(sub.num_data(), 4);
        assert!(sub.same_bins_as(&ds));
        assert_eq!(sub.label(), &[1.0, 5.0, 6.0, 39.0]);
        assert_eq!(sub.weight().unwrap(), &[2.0, 6.0, 7.0, 40.0]);
        assert_eq!(sub.init_score().unwrap(), &[1.0, 5.0, 6.0, 39.0, 41.0, 45.0, 46.0, 79.0]);
        for f in 0..ds.num_features() {
            let full = ds.bin_indices(f);
            assert_eq!(sub.bin_indices(f), used.iter().map(|&i| full[i as usize]).collect::<Vec<_>>());
        }
        let e = ds.subset(&[3, 40]).unwrap_err().to_string();
        assert!(e.contains("[Used indices of subset]: does not tolerate element [#1 = 40] outside [0, 39]"), "{e}");
        assert!(ds.subset(&[5, 3]).unwrap_err().to_string().contains("should be sorted"));
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
