//! PyO3 bindings for `lgbm-core`.
//!
//! This layer only converts between Python objects and core types; all
//! modeling logic lives in `lgbm-core`. The public Python API
//! (`Dataset`, `Booster`, `train`, callbacks) is implemented in
//! `python/lightgbm_rust/` on top of the two classes exported here.
//!
//! Conventions:
//! * every long-running call releases the GIL (`Python::detach`);
//! * core errors become `lightgbm_rust.LightGBMError`;
//! * Rust panics are caught and also surface as `LightGBMError`, so a bug in
//!   the engine never aborts the interpreter.
//! * dense NumPy input that is float32/float64 and C- or F-contiguous is
//!   borrowed without copying for the duration of the call.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use lgbm_core::arrow::{ArrowArrayStream, ArrowChunkedArray};
use lgbm_core::boosting::PredictKind;
use lgbm_core::tree::find_in_bitset;
use lgbm_core::{
    Config, Dataset, DatasetFields, DenseMatrix, DenseValues, Gbdt, LgbmError, Matrix as Matrix2, SparseIndptr,
    SparseMatrix,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyUntypedArrayMethods,
};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods, PyDict};

create_exception!(_lightgbm_rust, LightGBMError, PyException, "Error thrown by lightgbm-rust.");

fn panic_message(p: &(dyn Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

fn core_err(e: LgbmError) -> PyErr {
    LightGBMError::new_err(e.to_string())
}

/// Run `f`, mapping core errors and panics to `LightGBMError`.
fn guarded<R>(f: impl FnOnce() -> lgbm_core::Result<R>) -> PyResult<R> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(core_err(e)),
        Err(p) => Err(LightGBMError::new_err(format!(
            "internal error (panic) in lightgbm-rust: {}",
            panic_message(p.as_ref())
        ))),
    }
}

/// Release the GIL and run `f` under [`guarded`].
fn detached<R: Send>(py: Python<'_>, f: impl FnOnce() -> lgbm_core::Result<R> + Send) -> PyResult<R> {
    py.detach(|| guarded(f))
}

enum SparseIndptrArray<'py> {
    I32(PyReadonlyArray1<'py, i32>),
    I64(PyReadonlyArray1<'py, i64>),
}

enum ValuesArray<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
}

/// A borrowed 2-D float array, or a CSR/CSC matrix passed as
/// `(is_csr, indptr, indices, data, nrows, ncols)`.
enum Matrix<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
    Sparse {
        is_csr: bool,
        indptr: SparseIndptrArray<'py>,
        indices: PyReadonlyArray1<'py, i32>,
        values: ValuesArray<'py>,
        nrows: usize,
        ncols: usize,
    },
}

impl<'py> Matrix<'py> {
    fn extract(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(a) = obj.extract::<PyReadonlyArray2<'py, f64>>() {
            return Ok(Self::F64(a));
        }
        if let Ok(a) = obj.extract::<PyReadonlyArray2<'py, f32>>() {
            return Ok(Self::F32(a));
        }
        type SparseParts<'py> = (bool, Bound<'py, PyAny>, PyReadonlyArray1<'py, i32>, Bound<'py, PyAny>, usize, usize);
        if let Ok((is_csr, indptr, indices, values, nrows, ncols)) = obj.extract::<SparseParts<'py>>() {
            let indptr = if let Ok(a) = indptr.extract::<PyReadonlyArray1<'py, i32>>() {
                SparseIndptrArray::I32(a)
            } else {
                SparseIndptrArray::I64(indptr.extract().map_err(|_| PyTypeError::new_err("indptr must be int32 or int64"))?)
            };
            let values = if let Ok(a) = values.extract::<PyReadonlyArray1<'py, f64>>() {
                ValuesArray::F64(a)
            } else {
                ValuesArray::F32(values.extract().map_err(|_| PyTypeError::new_err("data must be float32 or float64"))?)
            };
            return Ok(Self::Sparse { is_csr, indptr, indices, values, nrows, ncols });
        }
        Err(PyTypeError::new_err("expected a 2-D numpy array of dtype float32 or float64"))
    }

    fn view(&self) -> PyResult<Matrix2<'_>> {
        let Self::Sparse { is_csr, indptr, indices, values, nrows, ncols } = self else {
            return self.dense_view().map(Matrix2::Dense);
        };
        let indptr = match indptr {
            SparseIndptrArray::I32(a) => SparseIndptr::I32(slice_of(a, "indptr")?),
            SparseIndptrArray::I64(a) => SparseIndptr::I64(slice_of(a, "indptr")?),
        };
        let values = match values {
            ValuesArray::F32(a) => DenseValues::F32(slice_of(a, "data")?),
            ValuesArray::F64(a) => DenseValues::F64(slice_of(a, "data")?),
        };
        SparseMatrix::new(indptr, slice_of(indices, "indices")?, values, *nrows, *ncols, *is_csr)
            .map(Matrix2::Sparse)
            .map_err(core_err)
    }

    fn dense_view(&self) -> PyResult<DenseMatrix<'_>> {
        fn layout<T: numpy::Element>(a: &PyReadonlyArray2<'_, T>) -> PyResult<(usize, usize, bool)> {
            let s = a.shape();
            let row_major = if a.is_c_contiguous() {
                true
            } else if a.is_fortran_contiguous() {
                false
            } else {
                return Err(PyValueError::new_err("array must be C- or F-contiguous"));
            };
            Ok((s[0], s[1], row_major))
        }
        let not_contig = |_| PyValueError::new_err("array must be C- or F-contiguous");
        let r = match self {
            Self::F32(a) => {
                let (n, m, rm) = layout(a)?;
                DenseMatrix::new(DenseValues::F32(a.as_slice().map_err(not_contig)?), n, m, rm)
            }
            Self::F64(a) => {
                let (n, m, rm) = layout(a)?;
                DenseMatrix::new(DenseValues::F64(a.as_slice().map_err(not_contig)?), n, m, rm)
            }
            Self::Sparse { .. } => unreachable!("sparse input has no dense view"),
        };
        r.map_err(core_err)
    }
}

fn slice_of<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice().map_err(|_| PyValueError::new_err(format!("{what} must be a contiguous 1-D array")))
}

fn config_from(params: Vec<(String, String)>) -> PyResult<Config> {
    guarded(|| Config::from_pairs(params))
}

/// Binned dataset (`lgbm_core::Dataset`).
#[pyclass(name = "RsDataset", module = "lightgbm_rust._lightgbm_rust")]
struct RsDataset {
    inner: Arc<Dataset>,
    warnings: Vec<String>,
}

#[pymethods]
impl RsDataset {
    #[new]
    #[pyo3(signature = (data, label, params, weight=None, init_score=None, feature_names=None, reference=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        label: PyReadonlyArray1<'_, f32>,
        params: Vec<(String, String)>,
        weight: Option<PyReadonlyArray1<'_, f32>>,
        init_score: Option<PyReadonlyArray1<'_, f64>>,
        feature_names: Option<Vec<String>>,
        reference: Option<PyRef<'_, RsDataset>>,
    ) -> PyResult<Self> {
        let cfg = config_from(params)?;
        let mat = Matrix::extract(data)?;
        let view = mat.view()?;
        let label = slice_of(&label, "label")?;
        let weight = weight.as_ref().map(|w| slice_of(w, "weight")).transpose()?;
        let init_score = init_score.as_ref().map(|s| slice_of(s, "init_score")).transpose()?;
        let reference = reference.map(|r| r.inner.clone());
        let fields = DatasetFields {
            label,
            weight,
            init_score,
            feature_names,
            categorical_features: Vec::new(),
        };
        let ds = detached(py, || match &reference {
            Some(r) => Dataset::from_matrix_with_reference(&view, fields, r, cfg.num_threads),
            None => Dataset::from_matrix(&view, fields, &cfg),
        })?;
        let mut warnings = cfg.warnings;
        warnings.extend(ds.warnings().iter().cloned());
        Ok(Self { inner: Arc::new(ds), warnings })
    }

    /// upstream: `LGBM_DatasetGetSubset`.
    fn subset(&self, py: Python<'_>, used: PyReadonlyArray1<'_, i32>, params: Vec<(String, String)>) -> PyResult<Self> {
        let cfg = config_from(params)?;
        let used = slice_of(&used, "used_indices")?;
        let full = &self.inner;
        let ds = detached(py, || lgbm_core::dataset::with_num_threads(cfg.num_threads, || full.subset(used))?)?;
        Ok(Self { inner: Arc::new(ds), warnings: cfg.warnings })
    }

    fn config_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    fn num_data(&self) -> usize {
        self.inner.num_data()
    }

    /// Number of input columns (upstream `num_total_features`).
    fn num_feature(&self) -> usize {
        self.inner.num_total_features()
    }

    /// Number of non-trivial (trainable) features.
    fn num_used_features(&self) -> usize {
        self.inner.num_features()
    }

    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names().to_vec()
    }

    /// Returns warnings (e.g. spaces replaced), like upstream's log output.
    fn set_feature_names(&mut self, names: Vec<String>) -> PyResult<Vec<String>> {
        let replaced = guarded(|| Arc::make_mut(&mut self.inner).set_feature_names(names))?;
        Ok(if replaced { vec![lgbm_core::dataset::FEATURE_NAME_SPACE_WARNING.to_string()] } else { Vec::new() })
    }

    fn feature_infos(&self) -> Vec<String> {
        self.inner.feature_infos()
    }

    /// Bin upper bounds of input column `col`.
    fn bin_upper_bounds(&self, col: usize) -> PyResult<Vec<f64>> {
        if col >= self.inner.num_total_features() {
            return Err(PyValueError::new_err("column index out of range"));
        }
        Ok(self.inner.bin_mapper_by_real(col).bin_upper_bound.clone())
    }

    /// upstream: `LGBM_DatasetGetFeatureNumBin` (0 for unused features).
    fn feature_num_bin(&self, feature: i64) -> PyResult<usize> {
        let n = self.inner.num_total_features();
        if feature < 0 || feature as usize >= n {
            return Err(LightGBMError::new_err(format!(
                "Tried to retrieve number of bins for feature index {feature}, but the valid feature indices are [0, {}].",
                n as i64 - 1
            )));
        }
        let col = feature as usize;
        Ok(match self.inner.inner_feature_index(col) {
            Some(_) => self.inner.bin_mapper_by_real(col).num_bin as usize,
            None => 0,
        })
    }

    /// Per-row bin indices of input column `col`, or `None` for trivial features.
    fn bin_indices<'py>(&self, py: Python<'py>, col: usize) -> PyResult<Option<Bound<'py, PyArray1<u32>>>> {
        if col >= self.inner.num_total_features() {
            return Err(PyValueError::new_err("column index out of range"));
        }
        Ok(self.inner.inner_feature_index(col).map(|i| self.inner.bin_indices(i).into_pyarray(py)))
    }

    fn get_label<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        self.inner.label().to_vec().into_pyarray(py)
    }

    fn get_weight<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.inner.weight().map(|w| w.to_vec().into_pyarray(py))
    }

    fn get_init_score<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.init_score().map(|s| s.to_vec().into_pyarray(py))
    }

    /// Query boundaries (`num_queries + 1` offsets), like upstream's `group` field.
    fn get_group<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.inner.metadata.query_boundaries.as_ref().map(|b| b.clone().into_pyarray(py))
    }

    /// Dense position ids, like upstream's `position` field.
    fn get_position<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<i32>>> {
        self.inner.metadata.positions.as_ref().map(|p| p.clone().into_pyarray(py))
    }

    /// Replace a metadata field and return upstream's warnings. Boosters
    /// already created from this dataset keep the previous values.
    #[pyo3(signature = (name, values=None))]
    fn set_field(&mut self, name: &str, values: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<String>> {
        let n = self.inner.num_data();
        let ds = Arc::make_mut(&mut self.inner);
        let mut warnings = Vec::new();
        match name {
            "group" => {
                let a: Option<PyReadonlyArray1<'_, i32>> = values.map(|o| o.extract()).transpose()?;
                let v = a.as_ref().map(|a| slice_of(a, "group")).transpose()?;
                ds.metadata.set_query(n, v).map_err(core_err)?;
            }
            "position" => {
                let a: Option<PyReadonlyArray1<'_, i32>> = values.map(|o| o.extract()).transpose()?;
                let v = a.as_ref().map(|a| slice_of(a, "position")).transpose()?;
                warnings = ds.metadata.set_position(n, v).map_err(core_err)?;
            }
            "label" => {
                let a: PyReadonlyArray1<'_, f32> = values
                    .ok_or_else(|| PyValueError::new_err("label cannot be None"))?
                    .extract()?;
                let v = slice_of(&a, "label")?;
                ds.metadata.set_label(n, v).map_err(core_err)?;
            }
            "weight" => {
                let a: Option<PyReadonlyArray1<'_, f32>> = values.map(|o| o.extract()).transpose()?;
                let v = a.as_ref().map(|a| slice_of(a, "weight")).transpose()?;
                ds.metadata.set_weight(n, v).map_err(core_err)?;
            }
            "init_score" => {
                let a: Option<PyReadonlyArray1<'_, f64>> = values.map(|o| o.extract()).transpose()?;
                let v = a.as_ref().map(|a| slice_of(a, "init_score")).transpose()?;
                ds.metadata.set_init_score(n, v).map_err(core_err)?;
            }
            other => {
                return Err(LightGBMError::new_err(format!(
                    "field `{other}` is not supported by lightgbm-rust yet"
                )));
            }
        }
        Ok(warnings)
    }
}

/// Gradient boosting model (`lgbm_core::Gbdt`).
#[pyclass(name = "RsBooster", module = "lightgbm_rust._lightgbm_rust")]
struct RsBooster {
    inner: Gbdt,
    warnings: Vec<String>,
}

fn predict_kind(kind: &str) -> PyResult<PredictKind> {
    match kind {
        "normal" => Ok(PredictKind::Normal),
        "raw" => Ok(PredictKind::Raw),
        "leaf" => Ok(PredictKind::LeafIndex),
        "contrib" => Ok(PredictKind::Contrib),
        _ => Err(PyValueError::new_err(format!("unknown prediction kind `{kind}`"))),
    }
}

#[pymethods]
impl RsBooster {
    /// Create a booster for training on `train`.
    #[staticmethod]
    fn for_training(py: Python<'_>, train: PyRef<'_, RsDataset>, params: Vec<(String, String)>) -> PyResult<Self> {
        let cfg = config_from(params)?;
        let mut warnings = cfg.warnings.clone();
        let data = train.inner.clone();
        let mut inner = detached(py, || Gbdt::new(cfg, data, None))?;
        warnings.extend(inner.take_warnings());
        Ok(Self { inner, warnings })
    }

    #[staticmethod]
    fn from_model_string(py: Python<'_>, text: String) -> PyResult<Self> {
        let inner = detached(py, || Gbdt::load_model_from_string(&text))?;
        Ok(Self { inner, warnings: Vec::new() })
    }

    fn config_warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }

    /// upstream: `LGBM_BoosterMerge`.
    fn merge_from(&mut self, other: PyRef<'_, RsBooster>) {
        self.inner.merge_from(&other.inner);
    }

    fn add_valid(&mut self, data: PyRef<'_, RsDataset>, name: &str) -> PyResult<()> {
        let d = data.inner.clone();
        guarded(|| self.inner.add_valid(d, name))
    }

    /// One iteration with the built-in objective. Returns `True` when no
    /// further split is possible.
    fn update(&mut self, py: Python<'_>) -> PyResult<bool> {
        let g = &mut self.inner;
        detached(py, || g.train_one_iter(None))
    }

    /// One iteration with caller-supplied gradients/Hessians (class-major).
    fn update_custom(
        &mut self,
        py: Python<'_>,
        grad: PyReadonlyArray1<'_, f32>,
        hess: PyReadonlyArray1<'_, f32>,
    ) -> PyResult<bool> {
        let g = slice_of(&grad, "grad")?;
        let h = slice_of(&hess, "hess")?;
        let b = &mut self.inner;
        detached(py, || b.train_one_iter(Some((g, h))))
    }

    fn rollback_one_iter(&mut self) -> PyResult<()> {
        guarded(|| self.inner.rollback_one_iter())
    }

    fn eval_train(&self, py: Python<'_>) -> PyResult<Vec<(String, String, f64, bool)>> {
        let g = &self.inner;
        detached(py, || Ok(g.eval_train()))
    }

    fn eval_valid(&self, py: Python<'_>) -> PyResult<Vec<(String, String, f64, bool)>> {
        let g = &self.inner;
        detached(py, || Ok(g.eval_valid()))
    }

    fn num_valid(&self) -> usize {
        self.inner.num_valid()
    }

    /// Current scores of training data (`data_idx == 0`) or validation set
    /// `data_idx - 1`, class-major. With `transform`, the objective's output
    /// transformation is applied (upstream `__inner_predict`).
    fn inner_predict<'py>(&self, py: Python<'py>, data_idx: usize, transform: bool) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let scores = if data_idx == 0 {
            self.inner.train_scores()
        } else {
            self.inner.valid_scores(data_idx - 1)
        }
        .ok_or_else(|| LightGBMError::new_err("data_idx out of range or booster is not training"))?;
        let ntpi = self.inner.num_tree_per_iteration();
        let mut out = scores.to_vec();
        if let (true, Some(obj)) = (transform, self.inner.objective()) {
            let n = scores.len() / ntpi;
            let mut raw = vec![0.0; ntpi];
            let mut conv = vec![0.0; ntpi];
            for i in 0..n {
                for k in 0..ntpi {
                    raw[k] = scores[k * n + i];
                }
                obj.convert_output(&raw, &mut conv);
                for k in 0..ntpi {
                    out[k * n + i] = conv[k];
                }
            }
        }
        Ok(out.into_pyarray(py))
    }

    /// Predict a dense matrix. Returns a 2-D array `(nrows, width)`.
    /// `params` are the prediction parameters (upstream's predict `parameter` string).
    #[pyo3(signature = (data, kind="normal", start_iteration=0, num_iteration=-1, params=Vec::new()))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        kind: &str,
        start_iteration: i32,
        num_iteration: i32,
        params: Vec<(String, String)>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let kind = predict_kind(kind)?;
        let cfg = config_from(params)?;
        let g = &self.inner;
        let early_stop = guarded(|| {
            g.prediction_early_stop(cfg.pred_early_stop, cfg.pred_early_stop_freq, cfg.pred_early_stop_margin)
        })?;
        let mat = Matrix::extract(data)?;
        let view = mat.view()?;
        let nrows = view.nrows();
        let out =
            detached(py, || g.predict_matrix_early_stop(&view, kind, start_iteration, num_iteration, early_stop))?;
        let width = if nrows == 0 { 0 } else { out.len() / nrows };
        out.into_pyarray(py).reshape([nrows, width])
    }

    /// SHAP values of a CSR/CSC matrix as one `(indptr, indices, data)` per
    /// class, in the input's layout.
    #[pyo3(signature = (data, start_iteration=0, num_iteration=-1))]
    #[allow(clippy::type_complexity)]
    fn predict_contrib_sparse<'py>(
        &self,
        py: Python<'py>,
        data: &Bound<'py, PyAny>,
        start_iteration: i32,
        num_iteration: i32,
    ) -> PyResult<Vec<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<f64>>)>> {
        let mat = Matrix::extract(data)?;
        let view = mat.view()?;
        let Matrix2::Sparse(sp) = view else {
            return Err(PyTypeError::new_err("expected a CSR or CSC matrix"));
        };
        let g = &self.inner;
        let out = detached(py, || g.predict_contrib_sparse(&view, start_iteration, num_iteration, sp.is_csr()))?;
        Ok(out
            .into_iter()
            .map(|m| (m.indptr.into_pyarray(py), m.indices.into_pyarray(py), m.values.into_pyarray(py)))
            .collect())
    }

    #[pyo3(signature = (start_iteration=0, num_iteration=-1, importance_type=0))]
    fn model_to_string(&self, py: Python<'_>, start_iteration: i32, num_iteration: i32, importance_type: i32) -> PyResult<String> {
        let g = &self.inner;
        detached(py, || g.save_model_to_string(start_iteration, num_iteration, importance_type))
    }

    /// upstream: `LGBM_BoosterDumpModel` (JSON text).
    #[pyo3(signature = (start_iteration=0, num_iteration=-1, importance_type=0))]
    fn dump_model(&self, py: Python<'_>, start_iteration: i32, num_iteration: i32, importance_type: i32) -> PyResult<String> {
        let g = &self.inner;
        detached(py, || g.dump_model(start_iteration, num_iteration, importance_type))
    }

    #[pyo3(signature = (num_iteration=-1, importance_type=0))]
    fn feature_importance<'py>(&self, py: Python<'py>, num_iteration: i32, importance_type: i32) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let v = guarded(|| self.inner.feature_importance(num_iteration, importance_type))?;
        Ok(v.into_pyarray(py))
    }

    fn num_trees(&self) -> usize {
        self.inner.num_trees()
    }

    fn current_iteration(&self) -> usize {
        self.inner.current_iteration()
    }

    fn num_feature(&self) -> usize {
        self.inner.num_feature()
    }

    fn num_tree_per_iteration(&self) -> usize {
        self.inner.num_tree_per_iteration()
    }

    fn feature_names(&self) -> Vec<String> {
        self.inner.feature_names().to_vec()
    }

    fn objective_name(&self) -> Option<String> {
        self.inner.objective().map(|o| o.name().to_string())
    }

    fn loaded_parameters(&self) -> Option<String> {
        self.inner.loaded_parameters().map(str::to_string)
    }

    fn clear_objective(&mut self) {
        self.inner.clear_objective();
    }

    fn free_training_state(&mut self) {
        self.inner.free_training_state();
    }

    fn set_num_threads(&mut self, n: i32) -> PyResult<()> {
        guarded(|| self.inner.set_num_threads(n))
    }

    /// Full-precision tree arrays, one dict per tree (for differential tests).
    fn tree_arrays<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .trees()
            .iter()
            .map(|t| {
                let d = PyDict::new(py);
                let ni = t.num_leaves.saturating_sub(1);
                d.set_item("num_leaves", t.num_leaves)?;
                d.set_item("shrinkage", t.shrinkage)?;
                d.set_item("split_feature", t.split_feature[..ni].to_vec())?;
                d.set_item("split_gain", t.split_gain[..ni].iter().map(|&g| g as f64).collect::<Vec<_>>())?;
                d.set_item("threshold", t.threshold[..ni].to_vec())?;
                let cats: Vec<Option<Vec<i32>>> = (0..ni)
                    .map(|i| {
                        t.is_categorical(i).then(|| {
                            let bits = t.cat_bitset(i);
                            (0..bits.len() as i32 * 32).filter(|&c| find_in_bitset(bits, c)).collect()
                        })
                    })
                    .collect();
                d.set_item("cat_threshold", cats)?;
                d.set_item("decision_type", t.decision_type[..ni].to_vec())?;
                d.set_item("left_child", t.left_child[..ni].to_vec())?;
                d.set_item("right_child", t.right_child[..ni].to_vec())?;
                d.set_item("internal_value", t.internal_value[..ni].to_vec())?;
                d.set_item("internal_weight", t.internal_weight[..ni].to_vec())?;
                d.set_item("internal_count", t.internal_count[..ni].to_vec())?;
                d.set_item("leaf_value", t.leaf_value[..t.num_leaves].to_vec())?;
                d.set_item("leaf_weight", t.leaf_weight[..t.num_leaves.min(t.leaf_weight.len())].to_vec())?;
                d.set_item("leaf_count", t.leaf_count[..t.num_leaves].to_vec())?;
                Ok(d)
            })
            .collect()
    }

    /// Gradients and Hessians of the most recent iteration (class-major).
    #[allow(clippy::type_complexity)]
    fn last_gradients<'py>(&self, py: Python<'py>) -> Option<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
        self.inner
            .last_gradients()
            .map(|(g, h)| (g.to_vec().into_pyarray(py), h.to_vec().into_pyarray(py)))
    }
}

/// Upstream parameter table: `[(name, cpp_type, [aliases...])]` in `config.h` order.
#[pyfunction]
fn param_specs() -> Vec<(String, String, Vec<String>)> {
    lgbm_core::config::param_specs()
        .iter()
        .map(|s| (s.name.clone(), s.cpp_type.clone(), s.aliases.clone()))
        .collect()
}

/// Parse and validate parameters; returns the non-fatal warnings.
#[pyfunction]
fn validate_params(params: Vec<(String, String)>) -> PyResult<Vec<String>> {
    Ok(config_from(params)?.warnings)
}

/// upstream: `LGBM_DatasetUpdateParamChecking`.
#[pyfunction]
fn dataset_update_param_checking(old: Vec<(String, String)>, new: Vec<(String, String)>) -> PyResult<()> {
    let old = config_from(old)?;
    let new = config_from(new)?;
    guarded(|| lgbm_core::config::dataset_update_param_checking(&old, &new))
}

/// Consume the `ArrowArrayStream` in an `__arrow_c_stream__` capsule.
fn arrow_stream(capsule: &Bound<'_, PyAny>) -> PyResult<ArrowChunkedArray> {
    let capsule = capsule
        .cast::<PyCapsule>()
        .map_err(|_| PyTypeError::new_err("expected the PyCapsule returned by __arrow_c_stream__()"))?;
    let ptr = capsule.pointer_checked(Some(c"arrow_array_stream"))?;
    guarded(|| unsafe { ArrowChunkedArray::from_stream(ptr.as_ptr().cast::<ArrowArrayStream>()) })
}

/// A table stream as column-major float64 values: `(values, num_rows, num_columns)`.
///
/// upstream: c_api.cpp `DatasetCreateFromArrowChunkedArray` and
/// `LGBM_BoosterPredictForArrowChunkedArray` read every field as double.
#[pyfunction]
fn arrow_table_to_columns<'py>(
    py: Python<'py>,
    capsule: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, usize)> {
    let table = arrow_stream(capsule)?;
    let (values, n, ncol) = guarded(|| {
        let ncol = table.num_fields()?;
        let n = table.len();
        let mut values = Vec::with_capacity(n * ncol);
        for j in 0..ncol {
            values.extend(table.field_values::<f64>(j)?);
        }
        Ok((values, n, ncol))
    })?;
    Ok((values.into_pyarray(py), n, ncol))
}

/// Values of a field stream as `dtype` ("float32", "float64" or "int32"); the
/// fields of a table are concatenated (multiclass init scores).
///
/// upstream: metadata.cpp `SetLabel` / `SetWeights` (label_t), `SetInitScore`
/// (double, `InitScoreView`), `SetQuery` / `SetPosition` (data_size_t).
#[pyfunction]
fn arrow_field_values<'py>(py: Python<'py>, capsule: &Bound<'py, PyAny>, dtype: &str) -> PyResult<Bound<'py, PyAny>> {
    let arr = arrow_stream(capsule)?;
    Ok(match dtype {
        "float32" => guarded(|| arr.values::<f32>())?.into_pyarray(py).into_any(),
        "float64" => guarded(|| arr.concatenated_values::<f64>())?.into_pyarray(py).into_any(),
        "int32" => guarded(|| arr.values::<i32>())?.into_pyarray(py).into_any(),
        other => return Err(PyValueError::new_err(format!("unsupported dtype {other}"))),
    })
}

#[pymodule]
fn _lightgbm_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("LightGBMError", m.py().get_type::<LightGBMError>())?;
    m.add("UPSTREAM_VERSION", lgbm_core::UPSTREAM_VERSION)?;
    m.add("UPSTREAM_COMMIT", lgbm_core::UPSTREAM_COMMIT)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<RsDataset>()?;
    m.add_class::<RsBooster>()?;
    m.add_function(wrap_pyfunction!(param_specs, m)?)?;
    m.add_function(wrap_pyfunction!(validate_params, m)?)?;
    m.add_function(wrap_pyfunction!(dataset_update_param_checking, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_table_to_columns, m)?)?;
    m.add_function(wrap_pyfunction!(arrow_field_values, m)?)?;
    Ok(())
}
