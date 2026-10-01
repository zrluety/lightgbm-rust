"""Dataset and Booster, mirroring ``lightgbm.basic`` (LightGBM 4.7.0).

The engine is the pure-Rust ``lgbm-core`` crate exposed through
``lightgbm_rust._lightgbm_rust``. This module reproduces the Python-side
behavior of upstream ``lightgbm/basic.py`` (parameter handling, lazy dataset
construction, input conversion, evaluation naming, prediction defaults) so
that code written against ``lightgbm`` behaves the same. Features that are
not implemented raise ``LightGBMError`` with a "not supported by
lightgbm-rust yet" message instead of being silently ignored.

Portions of the logic follow upstream ``python-package/lightgbm/basic.py``
(Copyright Microsoft Corporation, MIT License; see NOTICE).
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np

from . import _lightgbm_rust as _rs
from ._lightgbm_rust import LightGBMError

__all__ = ["Booster", "Dataset", "EvalResult", "LightGBMError", "register_logger"]

_NOT_SUPPORTED = "not supported by lightgbm-rust yet"

_LGBM_EvalFunctionResultType = Tuple[str, float, bool]


def _unsupported(what: str) -> LightGBMError:
    return LightGBMError(f"{_NOT_SUPPORTED}: {what}")


# --------------------------------------------------------------------------- logging


class _DummyLogger:
    def info(self, msg: str) -> None:
        print(msg)  # noqa: T201

    def warning(self, msg: str) -> None:
        import warnings

        warnings.warn(msg, stacklevel=3)


_LOGGER: Any = _DummyLogger()
_INFO_METHOD_NAME = "info"
_WARNING_METHOD_NAME = "warning"


def register_logger(
    logger: Any, info_method_name: str = "info", warning_method_name: str = "warning"
) -> None:
    """Register a custom logger (same contract as ``lightgbm.register_logger``)."""
    for name in (info_method_name, warning_method_name):
        if not callable(getattr(logger, name, None)):
            raise TypeError(f"Logger must provide '{info_method_name}' and '{warning_method_name}' method")
    global _LOGGER, _INFO_METHOD_NAME, _WARNING_METHOD_NAME
    _LOGGER = logger
    _INFO_METHOD_NAME = info_method_name
    _WARNING_METHOD_NAME = warning_method_name


def _log_info(msg: str) -> None:
    getattr(_LOGGER, _INFO_METHOD_NAME)(msg)


def _log_warning(msg: str) -> None:
    getattr(_LOGGER, _WARNING_METHOD_NAME)(msg)


def _log_native(msg: str) -> None:
    """Engine log lines go to the logger's info method, like upstream's C++ log callback."""
    getattr(_LOGGER, _INFO_METHOD_NAME)(msg)


# upstream: the C++ log level is process-global and only changes when a
# parameter string contains `verbosity` (preferred) or `verbose`
# (Config::SetVerbosity). 1 = Info is the initial level.
_ENGINE_LOG_LEVEL = 1


def _emit_engine_warnings(warnings: List[str], params: Optional[Dict[str, Any]]) -> None:
    global _ENGINE_LOG_LEVEL
    params = params or {}
    for key in ("verbosity", "verbose"):
        if key in params:
            try:
                _ENGINE_LOG_LEVEL = int(params[key])
            except (TypeError, ValueError):
                pass
            break
    if _ENGINE_LOG_LEVEL >= 0:
        for w in warnings:
            _log_native(f"[LightGBM] [Warning] {w}")


class _TempFile:
    """Temporary file path that is removed on exit (upstream ``_TempFile``)."""

    def __enter__(self) -> "_TempFile":
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile(prefix="lightgbm_tmp_", delete=True) as f:
            self.name = f.name
            self.path = Path(self.name)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.path.is_file():
            self.path.unlink()


class LGBMDeprecationWarning(FutureWarning):
    """Custom deprecation warning."""


# --------------------------------------------------------------------------- params


class _ConfigAliases:
    aliases: Optional[Dict[str, List[str]]] = None
    types: Optional[Dict[str, str]] = None

    @classmethod
    def _load(cls) -> None:
        if cls.aliases is None:
            specs = _rs.param_specs()
            # upstream: Config::DumpAliases orders aliases with Config::SortAlias
            cls.aliases = {
                name: [name, *sorted(aliases, key=lambda a: (len(a), a))] for name, _, aliases in specs
            }
            cls.types = {name: cpp_type for name, cpp_type, _ in specs}

    @classmethod
    def get(cls, *args: str) -> Set[str]:
        ret: Set[str] = set()
        for i in args:
            ret.update(cls.get_sorted(i))
        return ret

    @classmethod
    def get_sorted(cls, name: str) -> List[str]:
        cls._load()
        assert cls.aliases is not None
        return cls.aliases.get(name, [name])

    @classmethod
    def get_by_alias(cls, *args: str) -> Set[str]:
        cls._load()
        assert cls.aliases is not None
        ret = set(args)
        for arg in args:
            for aliases in cls.aliases.values():
                if arg in aliases:
                    ret.update(aliases)
                    break
        return ret

    @classmethod
    def canonical(cls, key: str) -> str:
        cls._load()
        assert cls.aliases is not None
        for name, aliases in cls.aliases.items():
            if key in aliases:
                return name
        return key


def _choose_param_value(main_param_name: str, params: Dict[str, Any], default_value: Any) -> Dict[str, Any]:
    """Get a single parameter value, accounting for aliases (upstream semantics)."""
    params = copy.deepcopy(params)
    aliases = [a for a in _ConfigAliases.get_sorted(main_param_name) if a != main_param_name]
    if main_param_name in params:
        for param in aliases:
            params.pop(param, None)
        return params
    for param in aliases:
        if param in params:
            params[main_param_name] = params[param]
            break
    if main_param_name in params:
        for param in aliases:
            params.pop(param, None)
        return params
    params[main_param_name] = default_value
    return params


def _is_numeric(obj: Any) -> bool:
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        return False


def _to_string(x: Any) -> str:
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    return str(x)


def _param_dict_to_pairs(data: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """upstream: ``_param_dict_to_str``, returned as key/value pairs."""
    if not data:
        return []
    pairs = []
    for key, val in data.items():
        if isinstance(val, (list, tuple, set)) or (isinstance(val, np.ndarray) and val.ndim == 1):
            pairs.append((key, ",".join(map(_to_string, val))))
        elif isinstance(val, (str, Path, int, float, bool, np.number, np.bool_)) or _is_numeric(val):
            pairs.append((key, str(val)))
        elif val is not None:
            raise TypeError(f"Unknown type of parameter:{key}, got:{type(val).__name__}")
    return pairs


def _parse_loaded_params(text: Optional[str]) -> Dict[str, Any]:
    """upstream: ``Booster._get_loaded_param`` (``[name: value]`` lines -> typed dict)."""
    out: Dict[str, Any] = {}
    if not text:
        return out
    _ConfigAliases._load()
    types = _ConfigAliases.types or {}
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("[") and line.endswith("]")) or ": " not in line:
            continue
        key, value = line[1:-1].split(": ", 1)
        t = types.get(key, "std::string")
        try:
            if t == "int":
                out[key] = int(value)
            elif t == "double":
                out[key] = float(value)
            elif t == "bool":
                out[key] = value in ("1", "true")
            elif t.startswith("std::vector<"):
                inner = t[len("std::vector<") : -1]
                items = [v for v in value.split(",") if v != ""]
                if inner == "int":
                    out[key] = [int(v) for v in items]
                elif inner == "double":
                    out[key] = [float(v) for v in items]
                else:
                    out[key] = items
            else:
                out[key] = value
        except ValueError:
            out[key] = value
    return out


# --------------------------------------------------------------------------- inputs


def _is_pandas_df(data: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return False
    return isinstance(data, pd.DataFrame)


def _is_pandas_series(data: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return False
    return isinstance(data, pd.Series)


def _is_scipy_sparse(data: Any) -> bool:
    try:
        import scipy.sparse
    except ImportError:  # pragma: no cover
        return False
    return scipy.sparse.issparse(data)


def _is_allowed_numpy_dtype(dtype: type) -> bool:
    float128 = getattr(np, "float128", type(None))
    return issubclass(dtype, (np.integer, np.floating, np.bool_)) and not issubclass(
        dtype, (np.timedelta64, float128)
    )


def _pandas_to_numpy(data: Any, feature_name: Any, categorical_feature: Any) -> Tuple[np.ndarray, Optional[List[str]]]:
    """upstream: ``_data_from_pandas`` for frames without categorical columns."""
    import pandas as pd

    if len(data.shape) != 2 or data.shape[0] < 1:
        raise ValueError("Input data must be 2 dimensional and non empty.")
    names = [str(col) for col in data.columns] if feature_name == "auto" else None
    if any(isinstance(dtype, pd.CategoricalDtype) for dtype in data.dtypes):
        raise _unsupported("pandas categorical columns")
    bad = [f"{c}: {d}" for c, d in data.dtypes.items() if not _is_allowed_numpy_dtype(d.type)]
    if bad:
        raise ValueError(f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad)}")
    target_dtype = np.result_type(*[d.type for d in data.dtypes], np.float32)
    try:
        arr = data.to_numpy(dtype=target_dtype, copy=False)
    except ValueError:
        arr = data.to_numpy(dtype=target_dtype, na_value=np.nan)
    return arr, names


def _to_float_matrix(data: Any, feature_name: Any = "auto", categorical_feature: Any = "auto") -> Tuple[np.ndarray, Optional[List[str]]]:
    """Convert supported inputs to a 2-D float32/float64 array.

    float32/float64 arrays that are C- or F-contiguous are passed through
    unchanged (borrowed by the engine without copying). Other numeric dtypes
    are converted to float32, like upstream ``_np2d_to_np1d``.
    """
    names: Optional[List[str]] = None
    if isinstance(data, (str, Path)):
        raise _unsupported("training from files")
    if _is_scipy_sparse(data):
        raise _unsupported("scipy.sparse input")
    if isinstance(data, Sequence) or (isinstance(data, list) and data and isinstance(data[0], Sequence)):
        raise _unsupported("lightgbm.Sequence input")
    if _is_arrow_like_frame(data):
        if feature_name == "auto":
            names = list(nw.from_native(data).schema.names())
        values, nrow, ncol = _rs.arrow_table_to_columns(nw.from_native(data).__arrow_c_stream__())
        return values.reshape((ncol, nrow)).T, names
    if _is_pandas_df(data):
        data, names = _pandas_to_numpy(data, feature_name, categorical_feature)
    elif isinstance(data, list) and data and all(isinstance(m, np.ndarray) for m in data):
        # upstream: Dataset.__init_from_list_np2d (rows of all chunks, in order)
        chunks = []
        for m in data:
            if m.ndim != 2:
                raise ValueError("Input numpy.ndarray must be 2 dimensional")
            chunks.append(m if m.dtype in (np.float32, np.float64) else m.astype(np.float32))
        if len({c.shape[1] for c in chunks}) != 1:
            raise ValueError("Input arrays must have same number of columns")
        if len({c.dtype for c in chunks}) != 1:
            raise ValueError("Input chunks must have same type")
        data = np.vstack(chunks)
    elif isinstance(data, list):
        data = np.array(data)
    elif not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as err:  # pragma: no cover
            raise TypeError(f"Cannot initialize Dataset from {type(data).__name__}") from err
    if data.ndim != 2:
        raise ValueError("Input numpy.ndarray or list must be 2 dimensional")
    if data.dtype not in (np.float32, np.float64):
        data = data.astype(np.float32)
    if not (data.flags["C_CONTIGUOUS"] or data.flags["F_CONTIGUOUS"]):
        data = np.ascontiguousarray(data)
    return data, names


def _is_1d_list(data: Any) -> bool:
    return isinstance(data, list) and (not data or _is_numeric(data[0]))


def _is_numeric(obj: Any) -> bool:
    try:
        float(obj)
        return True
    except (TypeError, ValueError):
        return False


def _list_to_1d_numpy(*, data: Any, dtype: Any, name: str) -> np.ndarray:
    """Convert data to a 1-D numpy array (same accepted inputs and messages as upstream)."""
    if isinstance(data, np.ndarray) and data.ndim == 1:
        return np.ascontiguousarray(data, dtype=dtype)
    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[1] == 1:
        _log_warning("Converting column-vector to 1d array")
        return np.ascontiguousarray(data.ravel(), dtype=dtype)
    if _is_1d_list(data):
        return np.asarray(data, dtype=dtype)
    if _is_pandas_series(data):
        bad = [f"{c}: {d}" for c, d in data.to_frame().dtypes.items() if not _is_allowed_numpy_dtype(d.type)]
        if bad:
            raise ValueError(f"pandas dtypes must be int, float or bool.\nFields with bad pandas dtypes: {', '.join(bad)}")
        return np.asarray(data, dtype=dtype)
    raise TypeError(f"Wrong type({type(data).__name__}) for {name}.\nIt should be list, numpy 1-D array or pandas Series")


def _to_1d(data: Any, dtype: Any, name: str) -> np.ndarray:
    return _list_to_1d_numpy(data=data, dtype=dtype, name=name)


def _is_arrow_like_frame(data: Any) -> bool:
    """A non-pandas frame that narwhals exports through the Arrow C stream (pyarrow Table, polars DataFrame)."""
    return nwd.is_into_dataframe(data) and not _is_pandas_df(data)


def _is_arrow_like_field(data: Any) -> bool:
    """upstream: the Arrow branch of ``Dataset.set_field``."""
    return (nwd.is_into_dataframe(data) or nwd.is_into_series(data)) and not (
        _is_pandas_df(data) or _is_pandas_series(data)
    )


def _arrow_field_to_numpy(data: Any, dtype: str) -> np.ndarray:
    """upstream: ``LGBM_DatasetSetFieldFromArrowStream`` (tables are concatenated column by column)."""
    return _rs.arrow_field_values(nw.from_native(data, allow_series=True).__arrow_c_stream__(), dtype)


def _weight_is_all_ones(weight: Any) -> bool:
    """upstream: ``Dataset.set_weight`` drops weights that are all one."""
    if nwd.is_into_series(weight):
        return bool((nw.from_native(weight, series_only=True) == 1).all())
    return bool(np.all(weight == 1))


def _field_1d_to_numpy(data: Any, dtype: Any, name: str) -> np.ndarray:
    if _is_arrow_like_field(data):
        return _arrow_field_to_numpy(data, np.dtype(dtype).name)
    return _to_1d(data, dtype, name)


def _label_to_numpy(label: Any) -> np.ndarray:
    """upstream: ``Dataset.set_label`` (a one-column pandas DataFrame is accepted)."""
    if _is_pandas_df(label):
        if len(label.columns) > 1:
            raise ValueError("DataFrame for label cannot have multiple columns")
        return np.ravel(np.asarray(label, dtype=np.float32))
    return _field_1d_to_numpy(label, np.float32, "label")


def _init_score_to_numpy(init_score: Any) -> np.ndarray:
    if _is_arrow_like_field(init_score):
        return _arrow_field_to_numpy(init_score, "float64")
    s = np.asarray(init_score, dtype=np.float64)
    return np.ascontiguousarray(s.ravel(order="F") if s.ndim == 2 else s)


class Sequence:
    """Generic row-access data interface (``lightgbm.Sequence``).

    The class exists for API compatibility; constructing a ``Dataset`` from a
    ``Sequence`` is not supported yet.
    """

    batch_size = 4096

    def __getitem__(self, idx: Union[int, slice, List[int]]) -> np.ndarray:
        raise NotImplementedError("Sub-classes of lightgbm.Sequence must implement __getitem__()")

    def __len__(self) -> int:
        raise NotImplementedError("Sub-classes of lightgbm.Sequence must implement __len__()")


# --------------------------------------------------------------------------- results


class EvalResult(NamedTuple):
    """Result of an evaluation metric on a dataset (``lightgbm.EvalResult``)."""

    dataset_name: str
    metric_name: str
    metric_value: float
    maximize: bool
    metric_std_dev: Optional[float] = None

    def __len__(self) -> int:
        return 4 if self.metric_std_dev is None else 5

    def __iter__(self) -> Any:
        for i in range(len(self)):
            yield getattr(self, self._fields[i])

    def is_cv_result(self) -> bool:
        return self.metric_std_dev is not None


# --------------------------------------------------------------------------- Dataset

# upstream: Dataset.get_params (deliberately excludes min_data, nthreads and verbose)
_DATASET_PARAMS = (
    "bin_construct_sample_cnt",
    "categorical_feature",
    "data_random_seed",
    "enable_bundle",
    "feature_pre_filter",
    "forcedbins_filename",
    "group_column",
    "header",
    "ignore_column",
    "is_enable_sparse",
    "label_column",
    "linear_tree",
    "max_bin",
    "max_bin_by_feature",
    "min_data_in_bin",
    "pre_partition",
    "precise_float_parser",
    "two_round",
    "use_missing",
    "weight_column",
    "zero_as_missing",
)


class _InnerPredictor:
    """Prediction-only view of a model, used for continued training (upstream ``_InnerPredictor``).

    ``from_booster`` shares the booster, like upstream sharing the C++ handle,
    so later training of that booster is visible here.
    """

    def __init__(self, booster: "Booster", pandas_categorical: Optional[List[List]], pred_parameter: Dict[str, Any]):
        self._booster = booster
        self.pandas_categorical = pandas_categorical
        self.pred_parameter = pred_parameter
        self.num_class = booster.num_model_per_iteration()

    @classmethod
    def from_booster(cls, booster: "Booster", pred_parameter: Dict[str, Any]) -> "_InnerPredictor":
        return cls(booster=booster, pandas_categorical=booster.pandas_categorical, pred_parameter=pred_parameter)

    @classmethod
    def from_model_file(cls, model_file: Union[str, Path], pred_parameter: Dict[str, Any]) -> "_InnerPredictor":
        booster = Booster(model_file=model_file)
        return cls(booster=booster, pandas_categorical=booster.pandas_categorical, pred_parameter=pred_parameter)

    def __getstate__(self) -> Dict[str, Any]:
        this = self.__dict__.copy()
        this["_booster"] = self._booster.model_to_string(num_iteration=-1)
        return this

    def __setstate__(self, state: Dict[str, Any]) -> None:
        state["_booster"] = Booster(model_str=state["_booster"])
        self.__dict__.update(state)

    def predict(self, data: Any, start_iteration: int = 0, num_iteration: int = -1, raw_score: bool = False,
                pred_leaf: bool = False, pred_contrib: bool = False, data_has_header: bool = False,
                validate_features: bool = False) -> np.ndarray:
        if isinstance(data, Dataset):
            raise TypeError("Cannot use Dataset instance for prediction, please use raw data instead")
        return self._booster.predict(
            data,
            start_iteration=start_iteration,
            num_iteration=num_iteration if num_iteration > 0 else -1,
            raw_score=raw_score,
            pred_leaf=pred_leaf,
            pred_contrib=pred_contrib,
            validate_features=validate_features,
        )

    def current_iteration(self) -> int:
        return self._booster.current_iteration()


class Dataset:
    """Dataset in lightgbm-rust (same constructor and lazy construction as ``lightgbm.Dataset``)."""

    def __init__(
        self,
        data: Any,
        label: Any = None,
        reference: Optional["Dataset"] = None,
        weight: Any = None,
        group: Any = None,
        init_score: Any = None,
        feature_name: Union[List[str], str] = "auto",
        categorical_feature: Union[List[str], List[int], str] = "auto",
        params: Optional[Dict[str, Any]] = None,
        free_raw_data: bool = True,
        position: Any = None,
    ):
        self._rs: Optional[_rs.RsDataset] = None
        self.data = data
        self.label = label
        self.reference = reference
        self.weight = weight
        self.group = group
        self.position = position
        self.init_score = init_score
        self.feature_name = feature_name
        self.categorical_feature = categorical_feature
        self.params = copy.deepcopy(params) if params else {}
        self.free_raw_data = free_raw_data
        self.used_indices: Optional[List[int]] = None
        self._need_slice = True
        self._predictor: Any = None
        self.pandas_categorical: Optional[List[List]] = None
        self._params_back_up: Optional[Dict[str, Any]] = None
        self.version = 0
        self._has_non_default_feature_names = False

    # ---- construction

    def construct(self) -> "Dataset":
        """Lazily build the binned dataset."""
        if self._rs is not None:
            return self
        if self.used_indices is not None and self.reference is not None:
            return self._construct_subset()
        if self.data is None:
            raise ValueError("Cannot construct a Dataset whose raw data has been freed.")
        cat = self.categorical_feature
        if cat not in ("auto", None) and len(cat) > 0:
            raise _unsupported("categorical features")
        for alias in _ConfigAliases.get("categorical_feature"):
            if self.params.get(alias) not in (None, "", []):
                raise _unsupported("categorical features")

        mat, names = _to_float_matrix(self.data, self.feature_name, self.categorical_feature)
        if self.feature_name != "auto" and self.feature_name is not None:
            names = list(self.feature_name)
        self._has_non_default_feature_names = names is not None
        n = mat.shape[0]
        label = np.zeros(n, dtype=np.float32) if self.label is None else _label_to_numpy(self.label)
        weight = None
        if self.weight is not None and not _weight_is_all_ones(self.weight):
            weight = _field_1d_to_numpy(self.weight, np.float32, "weight")
        init_score = None
        if self.init_score is not None:
            init_score = _init_score_to_numpy(self.init_score)

        ref_rs = None
        if self.reference is not None:
            self.reference.construct()
            ref_rs = self.reference._rs
            # upstream: validation data inherits the reference's dataset parameters
            reference_params = self.reference.get_params()
            own_params = self.get_params()
            if own_params != reference_params:
                ignore = _ConfigAliases.get("categorical_feature")
                a = {k: v for k, v in own_params.items() if k not in ignore}
                b = {k: v for k, v in reference_params.items() if k not in ignore}
                if a != b:
                    _log_warning("Overriding the parameters from Reference Dataset.")
                self._update_params(reference_params)
        params = self.params
        predictor = self._predictor
        if predictor is not None and not isinstance(predictor, _InnerPredictor):
            raise TypeError(f"Wrong predictor type {type(predictor).__name__}")
        if predictor is not None and init_score is not None:
            # upstream _lazy_init: the init model's prediction replaces init_score
            init_score = None
        pairs = _param_dict_to_pairs(params)
        self._rs = _rs.RsDataset(
            mat,
            label,
            pairs,
            weight=weight,
            init_score=init_score,
            feature_names=names if ref_rs is None else None,
            reference=ref_rs,
        )
        _emit_engine_warnings(self._rs.config_warnings(), params)
        # upstream: _lazy_init re-reads the fields, which the engine may have modified
        self.label = self.get_field("label")
        self.weight = self.get_field("weight")
        if self.group is not None:
            self.set_group(self.group)
        if self.position is not None:
            self.set_position(self.position)
        if predictor is not None:
            self._set_init_score_by_predictor(predictor=predictor, data=self.data, used_indices=None)
        elif self.init_score is not None:
            self.init_score = self.get_field("init_score")
        if self.free_raw_data:
            self.data = None
        self.feature_name = self.get_feature_name()
        return self

    def _construct_subset(self) -> "Dataset":
        # upstream: Dataset.construct, "construct subset" branch
        assert self.reference is not None and self.used_indices is not None
        reference_params = self.reference.get_params()
        if self.get_params() != reference_params:
            ignore = _ConfigAliases.get("categorical_feature")
            a = {k: v for k, v in self.get_params().items() if k not in ignore}
            b = {k: v for k, v in reference_params.items() if k not in ignore}
            if a != b:
                _log_warning("Overriding the parameters from Reference Dataset.")
            self._update_params(reference_params)
        used_indices = _list_to_1d_numpy(data=self.used_indices, dtype=np.int32, name="used_indices")
        if self.reference.group is not None:
            group_info = np.array(self.reference.group).astype(np.int32, copy=False)
            _, self.group = np.unique(
                np.repeat(range(len(group_info)), repeats=group_info)[self.used_indices], return_counts=True
            )
        full = self.reference.construct()._rs
        assert full is not None
        self._rs = full.subset(np.ascontiguousarray(used_indices), _param_dict_to_pairs(self.params))
        _emit_engine_warnings(self._rs.config_warnings(), self.params)
        if not self.free_raw_data:
            self.get_data()
        if self.group is not None:
            self.set_group(self.group)
        if self.position is not None:
            self.set_position(self.position)
        if self.get_label() is None:
            raise ValueError("Label should not be None.")
        if isinstance(self._predictor, _InnerPredictor) and self._predictor is not self.reference._predictor:
            self.get_data()
            self._set_init_score_by_predictor(predictor=self._predictor, data=self.data, used_indices=used_indices)
        if self.free_raw_data:
            self.data = None
        self.feature_name = self.get_feature_name()
        return self

    def create_valid(
        self,
        data: Any,
        label: Any = None,
        weight: Any = None,
        group: Any = None,
        init_score: Any = None,
        params: Optional[Dict[str, Any]] = None,
        position: Any = None,
    ) -> "Dataset":
        """Create validation data aligned with the current Dataset."""
        ret = Dataset(
            data,
            label=label,
            reference=self,
            weight=weight,
            group=group,
            position=position,
            init_score=init_score,
            params=params,
            free_raw_data=self.free_raw_data,
        )
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        return ret

    def set_reference(self, reference: "Dataset") -> "Dataset":
        self.set_categorical_feature(reference.categorical_feature).set_feature_name(
            reference.feature_name
        )._set_predictor(reference._predictor)
        # we're done if self and reference share a common upstream reference
        if self.get_ref_chain().intersection(reference.get_ref_chain()):
            return self
        if self.data is not None:
            self.reference = reference
            return self._free_handle()
        raise LightGBMError(
            "Cannot set reference after freed raw data, set free_raw_data=False when construct Dataset to avoid this."
        )

    def _free_handle(self) -> "Dataset":
        self._rs = None
        self._need_slice = True
        if self.used_indices is not None:
            self.data = None
        return self

    def _set_init_score_by_predictor(
        self, predictor: Optional[_InnerPredictor], data: Any, used_indices: Optional[Union[List[int], np.ndarray]]
    ) -> "Dataset":
        # upstream: Dataset._set_init_score_by_predictor (file inputs are not supported)
        num_data = self.num_data()
        if predictor is not None:
            init_score = np.asarray(predictor.predict(data=data, raw_score=True), dtype=np.float64).ravel()
            if used_indices is not None:
                assert not self._need_slice
            if predictor.num_class > 1:
                new_init_score = np.empty(init_score.size, dtype=np.float64)
                for i in range(num_data):
                    for j in range(predictor.num_class):
                        new_init_score[j * num_data + i] = init_score[i * predictor.num_class + j]
                init_score = new_init_score
        elif self.init_score is not None:
            init_score = np.full_like(self.init_score, fill_value=0.0, dtype=np.float64)
        else:
            return self
        self.set_init_score(init_score)
        return self

    def _update_params(self, params: Optional[Dict[str, Any]]) -> "Dataset":
        if not params:
            return self
        params = copy.deepcopy(params)

        def update() -> None:
            if not self.params:
                self.params = params
            else:
                self._params_back_up = copy.deepcopy(self.params)
                self.params.update(params)

        if self._rs is None:
            update()
        else:
            try:
                _rs.dataset_update_param_checking(_param_dict_to_pairs(self.params), _param_dict_to_pairs(params))
            except LightGBMError:
                if self.data is not None:
                    update()
                    self._rs = None
                else:
                    raise
        return self

    def _reverse_update_params(self) -> "Dataset":
        if self._rs is None:
            self.params = copy.deepcopy(self._params_back_up) or {}
            self._params_back_up = None
        return self

    def _set_predictor(self, predictor: Optional[_InnerPredictor]) -> "Dataset":
        if predictor is None and self._predictor is None:
            return self
        elif isinstance(predictor, _InnerPredictor) and isinstance(self._predictor, _InnerPredictor):
            if (predictor == self._predictor) and (
                predictor.current_iteration() == self._predictor.current_iteration()
            ):
                return self
        if self._rs is None:
            self._predictor = predictor
        elif self.data is not None:
            self._predictor = predictor
            self._set_init_score_by_predictor(predictor=self._predictor, data=self.data, used_indices=None)
        elif self.used_indices is not None and self.reference is not None and self.reference.data is not None:
            self._predictor = predictor
            self._set_init_score_by_predictor(
                predictor=self._predictor, data=self.reference.data, used_indices=self.used_indices
            )
        else:
            raise LightGBMError(
                "Cannot set predictor after freed raw data, set free_raw_data=False when construct Dataset to avoid this."
            )
        return self

    # ---- fields

    def set_field(self, field_name: str, data: Any) -> "Dataset":
        if self._rs is None:
            raise Exception(f"Cannot set {field_name} before construct dataset")
        if field_name in ("label", "weight"):
            values = None if data is None else _field_1d_to_numpy(data, np.float32, field_name)
        elif field_name == "init_score":
            values = None if data is None else _init_score_to_numpy(data)
        elif field_name in ("group", "position"):
            values = None if data is None else _field_1d_to_numpy(data, np.int32, field_name)
        else:
            raise LightGBMError(f"Unknown field name: {field_name}")
        _emit_engine_warnings(self._rs.set_field(field_name, values), None)
        self.version += 1
        return self

    def get_field(self, field_name: str) -> Optional[np.ndarray]:
        if self._rs is None:
            raise Exception(f"Cannot get {field_name} before construct Dataset")
        if field_name == "label":
            return self._rs.get_label()
        if field_name == "weight":
            return self._rs.get_weight()
        if field_name == "init_score":
            arr = self._rs.get_init_score()
            if arr is not None:
                num_data = self.num_data()
                num_classes = arr.size // num_data
                if num_classes > 1:
                    arr = arr.reshape((num_data, num_classes), order="F")
            return arr
        if field_name == "group":
            return self._rs.get_group()
        if field_name == "position":
            return self._rs.get_position()
        raise LightGBMError(f"Unknown field name: {field_name}")

    def set_label(self, label: Any) -> "Dataset":
        self.label = label
        if self._rs is not None:
            if label is None:
                raise ValueError("Label should not be None")
            self.set_field("label", label)
            self.label = self.get_field("label")
        return self

    def set_weight(self, weight: Any) -> "Dataset":
        if weight is not None and _weight_is_all_ones(weight):
            weight = None
        self.weight = weight
        if self._rs is not None:
            self.set_field("weight", weight)
            self.weight = self.get_field("weight")
        return self

    def set_init_score(self, init_score: Any) -> "Dataset":
        self.init_score = init_score
        if self._rs is not None:
            self.set_field("init_score", init_score)
            self.init_score = self.get_field("init_score")
        return self

    def set_group(self, group: Any) -> "Dataset":
        self.group = group
        if self._rs is not None and group is not None:
            self.set_field("group", group)
            # original values can be modified at the engine side
            constructed_group = self.get_field("group")
            if constructed_group is not None:
                self.group = np.diff(constructed_group)
        return self

    def set_position(self, position: Any) -> "Dataset":
        self.position = position
        if self._rs is not None and position is not None:
            self.set_field("position", position)
            self.position = self.get_field("position")
        return self

    def set_feature_name(self, feature_name: Union[List[str], str]) -> "Dataset":
        if feature_name != "auto":
            self.feature_name = feature_name
            self._has_non_default_feature_names = True
        if self._rs is not None and feature_name is not None and feature_name != "auto":
            if len(feature_name) != self.num_feature():
                raise ValueError(
                    f"Length of feature_name({len(feature_name)}) and num_feature({self.num_feature()}) don't match"
                )
            _emit_engine_warnings(self._rs.set_feature_names(list(feature_name)), None)
        return self

    def set_categorical_feature(self, categorical_feature: Any) -> "Dataset":
        if categorical_feature not in ("auto", None) and len(categorical_feature) > 0:
            raise _unsupported("categorical features")
        self.categorical_feature = categorical_feature
        return self

    def get_label(self) -> Optional[np.ndarray]:
        if self.label is None and self._rs is not None:
            self.label = self.get_field("label")
        return self.label

    def get_weight(self) -> Optional[np.ndarray]:
        if self.weight is None and self._rs is not None:
            self.weight = self.get_field("weight")
        return self.weight

    def get_init_score(self) -> Optional[np.ndarray]:
        if self.init_score is None and self._rs is not None:
            self.init_score = self.get_field("init_score")
        return self.init_score

    def get_group(self) -> Any:
        if self.group is None:
            self.group = self.get_field("group")
            if self.group is not None:
                # boundaries -> group sizes
                self.group = np.diff(self.group)
        return self.group

    def get_position(self) -> Any:
        if self.position is None:
            self.position = self.get_field("position")
        return self.position

    def get_data(self) -> Any:
        if self._rs is None:
            raise Exception("Cannot get data before construct Dataset")
        if self._need_slice and self.used_indices is not None and self.reference is not None:
            self.data = self.reference.data
            if self.data is not None:
                if isinstance(self.data, np.ndarray) or _is_scipy_sparse(self.data):
                    self.data = self.data[self.used_indices, :]
                elif _is_pandas_df(self.data):
                    self.data = self.data.iloc[self.used_indices].copy()
                elif isinstance(self.data, Sequence):
                    self.data = self.data[self.used_indices]
                elif nwd.is_into_dataframe(self.data):
                    indices = np.array(self.used_indices)
                    self.data = nw.from_native(self.data)[indices].to_native()
                else:
                    _log_warning(
                        f"Cannot subset {type(self.data).__name__} type of raw data.\nReturning original raw data"
                    )
            self._need_slice = False
        if self.data is None:
            raise LightGBMError(
                "Cannot call `get_data` after freed raw data, set free_raw_data=False when construct Dataset to avoid this."
            )
        return self.data

    def get_feature_name(self) -> List[str]:
        if self._rs is None:
            raise LightGBMError("Cannot get feature_name before construct dataset")
        return self._rs.feature_names()

    def get_params(self) -> Dict[str, Any]:
        if self.params is None:
            return {}
        dataset_keys = _ConfigAliases.get(*_DATASET_PARAMS)
        return {k: v for k, v in self.params.items() if k in dataset_keys}

    def get_ref_chain(self, ref_limit: int = 100) -> Set["Dataset"]:
        head = self
        ref_chain: Set[Dataset] = set()
        while len(ref_chain) < ref_limit:
            if isinstance(head, Dataset):
                ref_chain.add(head)
                if head.reference is not None and head.reference not in ref_chain:
                    head = head.reference
                else:
                    break
            else:
                break
        return ref_chain

    def num_data(self) -> int:
        if self._rs is None:
            raise LightGBMError("Cannot get num_data before construct dataset")
        return self._rs.num_data()

    def num_feature(self) -> int:
        if self._rs is None:
            raise LightGBMError("Cannot get num_feature before construct dataset")
        return self._rs.num_feature()

    def feature_num_bin(self, feature: Union[int, str]) -> int:
        if self._rs is None:
            raise LightGBMError("Cannot get feature_num_bin before construct dataset")
        if isinstance(feature, str):
            feature = self.get_feature_name().index(feature)
        return len(self._rs.bin_upper_bounds(int(feature)))

    def subset(self, used_indices: List[int], params: Optional[Dict[str, Any]] = None) -> "Dataset":
        """Get subset of current Dataset (rows ``used_indices``, sharing its bin mappers)."""
        if params is None:
            params = self.params
        ret = Dataset(
            None,
            reference=self,
            feature_name=self.feature_name,
            categorical_feature=self.categorical_feature,
            params=params,
            free_raw_data=self.free_raw_data,
        )
        ret._predictor = self._predictor
        ret.pandas_categorical = self.pandas_categorical
        ret.used_indices = sorted(used_indices)
        return ret

    def save_binary(self, filename: Union[str, Path]) -> "Dataset":
        raise _unsupported("Dataset.save_binary()")

    def add_features_from(self, other: "Dataset") -> "Dataset":
        raise _unsupported("Dataset.add_features_from()")

    def _dump_text(self, filename: Union[str, Path]) -> "Dataset":
        """Write per-row bin indices in upstream ``Dataset::DumpTextFile`` format.

        Differences: ``num_groups`` equals the number of used features (no
        exclusive feature bundling), and ``max_bin_by_feature``/forced bins are
        always empty.
        """
        self.construct()
        assert self._rs is not None
        names = self._rs.feature_names()
        n_total = self._rs.num_feature()
        cols = [self._rs.bin_indices(j) for j in range(n_total)]
        lines = [
            f"num_features: {self._rs.num_used_features()}",
            f"num_total_features: {n_total}",
            f"num_groups: {self._rs.num_used_features()}",
            f"num_data: {self._rs.num_data()}",
            "feature_names: " + "".join(f"{n}, " for n in names),
            "max_bin_by_feature: ",
            "".join(f"{n}, " for n in names),
            "forced_bins: " + "".join(f"\nfeature {i}: " for i in range(n_total)),
        ]
        with open(filename, "w", newline="\n") as f:
            f.write("\n".join(lines))
            for i in range(self._rs.num_data()):
                f.write("\n" + "".join("NA, " if c is None else f"{int(c[i])}, " for c in cols))
        return self

    # ---- lightgbm-rust extensions (used by differential tests)

    def _bin_upper_bounds(self, feature: int) -> List[float]:
        self.construct()
        assert self._rs is not None
        return self._rs.bin_upper_bounds(feature)

    def _bin_indices(self, feature: int) -> Optional[np.ndarray]:
        self.construct()
        assert self._rs is not None
        return self._rs.bin_indices(feature)

    def _feature_infos(self) -> List[str]:
        self.construct()
        assert self._rs is not None
        return self._rs.feature_infos()


# --------------------------------------------------------------------------- Booster

_IMPORTANCE = {"split": 0, "gain": 1}


class Booster:
    """Booster in lightgbm-rust (API of ``lightgbm.Booster``)."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        train_set: Optional[Dataset] = None,
        model_file: Optional[Union[str, Path]] = None,
        model_str: Optional[str] = None,
    ):
        self._rs: Optional[_rs.RsBooster] = None
        self._network = False
        self.__need_reload_eval_info = True
        self._train_data_name = "training"
        self.__set_objective_to_none = False
        self.best_iteration = 0
        self.best_score: Dict[str, Dict[str, float]] = {}
        self.name_valid_sets: List[str] = []
        self.valid_sets: List[Dataset] = []
        self.pandas_categorical: Optional[List[List]] = None
        self.__init_predictor: Any = None
        self.__num_dataset = 0
        params = {} if params is None else copy.deepcopy(params)

        if train_set is not None:
            if not isinstance(train_set, Dataset):
                raise TypeError(f"Training data should be Dataset instance, met {type(train_set).__name__}")
            if callable(params.get("objective")):
                raise TypeError("Unknown type of parameter:objective, got:function")
            for alias in _ConfigAliases.get("machines", "num_machines"):
                if alias in params and str(params[alias]) not in ("", "1"):
                    raise _unsupported("distributed training")
            train_set._update_params(params).construct()
            params.update(train_set.get_params())
            self.train_set = train_set
            pairs = _param_dict_to_pairs(params)
            self._rs = _rs.RsBooster.for_training(train_set._rs, pairs)
            _emit_engine_warnings(self._rs.config_warnings(), params)
            self.__num_dataset = 1
            self.__init_predictor = train_set._predictor
            if self.__init_predictor is not None:
                assert self.__init_predictor._booster._rs is not None
                self._rs.merge_from(self.__init_predictor._booster._rs)
            self.pandas_categorical = train_set.pandas_categorical
            objective = _choose_param_value("objective", params, None)["objective"]
            if objective is not None and str(objective).lower() in ("none", "null", "custom", "na"):
                self.__set_objective_to_none = True
            self.params = params
        elif model_file is not None:
            with open(model_file, "r", encoding="utf-8") as f:
                self._load_model_str(f.read())
            self.params = _parse_loaded_params(self._rs.loaded_parameters())
        elif model_str is not None:
            self._load_model_str(model_str)
            self.params = _parse_loaded_params(self._rs.loaded_parameters())
        else:
            raise TypeError("Need at least one training dataset or model file or model string to create Booster instance")

    def _load_model_str(self, model_str: str) -> None:
        self._rs = _rs.RsBooster.from_model_string(model_str)
        self.__num_dataset = 0
        self.pandas_categorical = _load_pandas_categorical(model_str)

    # ---- pickling / copying

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("train_set", None)
        state.pop("valid_sets", None)
        state["_rs"] = self.model_to_string(num_iteration=-1) if self._rs is not None else None
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        model_str = state.pop("_rs")
        self.__dict__.update(state)
        self._rs = None
        if model_str is not None:
            self._rs = _rs.RsBooster.from_model_string(model_str)
        self.valid_sets = []

    def __copy__(self) -> "Booster":
        return self.__deepcopy__(None)

    def __deepcopy__(self, _: Any) -> "Booster":
        return Booster(model_str=self.model_to_string(num_iteration=-1))

    # ---- training

    def free_dataset(self) -> "Booster":
        self.__dict__.pop("train_set", None)
        self.__dict__.pop("valid_sets", None)
        self.__num_dataset = 0
        return self

    def free_network(self) -> "Booster":
        return self

    def set_network(self, *args: Any, **kwargs: Any) -> "Booster":
        raise _unsupported("distributed training")

    def set_train_data_name(self, name: str) -> "Booster":
        self._train_data_name = name
        return self

    def add_valid(self, data: Dataset, name: str) -> "Booster":
        if not isinstance(data, Dataset):
            raise TypeError(f"Validation data should be Dataset instance, met {type(data).__name__}")
        if data._predictor is not self.__init_predictor:
            raise LightGBMError("Add validation data failed, you should use same predictor for these data")
        data.construct()
        assert self._rs is not None
        self._rs.add_valid(data._rs, name)
        self.valid_sets.append(data)
        self.name_valid_sets.append(name)
        self.__num_dataset += 1
        return self

    def reset_parameter(self, params: Dict[str, Any]) -> "Booster":
        """Only ``objective`` -> none is supported (used for custom objectives)."""
        params = dict(params)
        obj = params.pop("objective", None)
        if obj is not None:
            if str(obj).lower() not in ("none", "null", "custom", "na"):
                raise _unsupported("reset_parameter(objective=...)")
            assert self._rs is not None
            self._rs.clear_objective()
        if params:
            raise _unsupported(f"reset_parameter({sorted(params)})")
        self.params.update({"objective": obj} if obj is not None else {})
        return self

    def update(self, train_set: Optional[Dataset] = None, fobj: Optional[Callable] = None) -> bool:
        """Run one boosting iteration. Returns ``True`` if no further split is possible."""
        if train_set is not None and train_set is not getattr(self, "train_set", None):
            raise _unsupported("Booster.update(train_set=<new dataset>)")
        if not hasattr(self, "train_set"):
            raise LightGBMError("Cannot update due to null training data")
        assert self._rs is not None
        if fobj is None:
            if self.__set_objective_to_none:
                raise LightGBMError("Cannot update due to null objective function.")
            return self._rs.update()
        if not self.__set_objective_to_none:
            self.reset_parameter({"objective": "none"})
            self.__set_objective_to_none = True
        grad, hess = fobj(self.__inner_predict(0), self.train_set)
        return self.__boost(grad, hess)

    def __boost(self, grad: Any, hess: Any) -> bool:
        grad = np.asarray(grad)
        hess = np.asarray(hess)
        if grad.ndim == 2:
            grad = grad.ravel(order="F")
        if hess.ndim == 2:
            hess = hess.ravel(order="F")
        grad = np.ascontiguousarray(grad, dtype=np.float32)
        hess = np.ascontiguousarray(hess, dtype=np.float32)
        if len(grad) != len(hess):
            raise ValueError(f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) don't match")
        n = self.train_set.num_data()
        k = self.num_model_per_iteration()
        if len(grad) != n * k:
            raise ValueError(
                f"Lengths of gradient ({len(grad)}) and Hessian ({len(hess)}) "
                f"don't match training data length ({n}) * "
                f"number of models per one iteration ({k})"
            )
        assert self._rs is not None
        return self._rs.update_custom(grad, hess)

    def rollback_one_iter(self) -> "Booster":
        assert self._rs is not None
        self._rs.rollback_one_iter()
        return self

    def current_iteration(self) -> int:
        assert self._rs is not None
        return self._rs.current_iteration()

    def num_model_per_iteration(self) -> int:
        assert self._rs is not None
        return self._rs.num_tree_per_iteration()

    def num_trees(self) -> int:
        assert self._rs is not None
        return self._rs.num_trees()

    def num_feature(self) -> int:
        assert self._rs is not None
        return self._rs.num_feature()

    def feature_name(self) -> List[str]:
        assert self._rs is not None
        return self._rs.feature_names()

    # ---- evaluation

    def __inner_predict(self, data_idx: int) -> np.ndarray:
        assert self._rs is not None
        transform = not self.__set_objective_to_none
        out = self._rs.inner_predict(data_idx, transform)
        k = self.num_model_per_iteration()
        if k > 1:
            out = out.reshape(k, -1).T
        return out

    def __inner_eval(self, data_name: str, data_idx: int, feval: Any) -> List[EvalResult]:
        assert self._rs is not None
        ret: List[EvalResult] = []
        if data_idx == 0:
            builtin = self._rs.eval_train()
        else:
            # every validation set carries the same metric list, in add order
            all_valid = self._rs.eval_valid()
            per_set = len(all_valid) // max(self._rs.num_valid(), 1)
            builtin = all_valid[(data_idx - 1) * per_set : data_idx * per_set]
        for _, metric, value, higher in builtin:
            ret.append(EvalResult(data_name, metric, value, higher))
        if feval is not None:
            fevals = feval if isinstance(feval, list) else [feval]
            cur_data = self.train_set if data_idx == 0 else self.valid_sets[data_idx - 1]
            for f in fevals:
                if f is None:
                    continue
                feval_ret = f(self.__inner_predict(data_idx), cur_data)
                if isinstance(feval_ret, list):
                    for name, val, hib in feval_ret:
                        ret.append(EvalResult(data_name, name, val, hib))
                else:
                    name, val, hib = feval_ret
                    ret.append(EvalResult(data_name, name, val, hib))
        return ret

    def eval(self, data: Dataset, name: str, feval: Any = None) -> List[EvalResult]:
        if not isinstance(data, Dataset):
            raise TypeError("Can only eval for Dataset instance")
        if data is getattr(self, "train_set", None):
            data_idx = 0
        else:
            data_idx = -1
            for i, v in enumerate(self.valid_sets):
                if data is v:
                    data_idx = i + 1
                    break
            if data_idx == -1:
                self.add_valid(data, name)
                data_idx = self.__num_dataset - 1
        return self.__inner_eval(name, data_idx, feval)

    def eval_train(self, feval: Any = None) -> List[EvalResult]:
        return self.__inner_eval(self._train_data_name, 0, feval)

    def eval_valid(self, feval: Any = None) -> List[EvalResult]:
        out: List[EvalResult] = []
        for i, name in enumerate(self.name_valid_sets):
            out.extend(self.__inner_eval(name, i + 1, feval))
        return out

    # ---- prediction

    def predict(
        self,
        data: Any,
        start_iteration: int = 0,
        num_iteration: Optional[int] = None,
        raw_score: bool = False,
        pred_leaf: bool = False,
        pred_contrib: bool = False,
        data_has_header: bool = False,
        validate_features: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        if isinstance(data, Dataset):
            raise TypeError("Cannot use Dataset instance for prediction, please use raw data instead")
        if pred_contrib:
            raise _unsupported("pred_contrib (SHAP values)")
        if kwargs:
            _emit_engine_warnings(_rs.validate_params(_param_dict_to_pairs(kwargs)), kwargs)
        if num_iteration is None:
            num_iteration = self.best_iteration if start_iteration <= 0 else -1
        if num_iteration is None or num_iteration <= 0:
            num_iteration = -1
        if validate_features and _is_pandas_df(data):
            # upstream: c_api.cpp LGBM_BoosterValidateFeatureNames
            names = [str(c) for c in data.columns]
            expected = self.feature_name()
            if len(names) != len(expected):
                raise LightGBMError(
                    f"Model was trained on {len(expected)} features, but got {len(names)} input features to predict."
                )
            for i, (e, got) in enumerate(zip(expected, names)):
                if e != got:
                    raise LightGBMError(f"Expected '{e}' at position {i} but found '{got}'")
        mat, _ = _to_float_matrix(data)
        kind = "leaf" if pred_leaf else ("raw" if raw_score else "normal")
        assert self._rs is not None
        preds = self._rs.predict(mat, kind, int(start_iteration), int(num_iteration))
        nrow = mat.shape[0]
        flat = preds.ravel()
        if pred_leaf:
            flat = flat.astype(np.int32)
        if flat.size != nrow:
            if nrow > 0 and flat.size % nrow == 0:
                return flat.reshape(nrow, -1)
            raise ValueError(f"Length of predict result ({flat.size}) cannot be divide nrow ({nrow})")
        return flat

    # ---- persistence

    def model_to_string(
        self, num_iteration: Optional[int] = None, start_iteration: int = 0, importance_type: str = "split"
    ) -> str:
        if num_iteration is None:
            num_iteration = self.best_iteration
        if importance_type not in _IMPORTANCE:
            raise ValueError(f"Unknown importance type: {importance_type}")
        assert self._rs is not None
        ret = self._rs.model_to_string(int(start_iteration), int(num_iteration), _IMPORTANCE[importance_type])
        return ret + _dump_pandas_categorical(self.pandas_categorical)

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> "Booster":
        text = self.model_to_string(num_iteration, start_iteration, importance_type)
        with open(filename, "w", newline="\n", encoding="utf-8") as f:
            f.write(text)
        return self

    def model_from_string(self, model_str: str) -> "Booster":
        self._load_model_str(model_str)
        self.__set_objective_to_none = False
        return self

    def dump_model(
        self,
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
        object_hook: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Dump Booster to JSON format (same contract as ``lightgbm.Booster.dump_model``)."""
        if num_iteration is None:
            num_iteration = self.best_iteration
        importance_type_int = _IMPORTANCE[importance_type]
        assert self._rs is not None
        text = self._rs.dump_model(int(start_iteration), int(num_iteration), importance_type_int)
        ret = json.loads(text, object_hook=object_hook)
        ret["pandas_categorical"] = json.loads(json.dumps(self.pandas_categorical, default=_json_default_with_numpy))
        return ret

    def trees_to_dataframe(self) -> Any:
        """Parse the fitted model into a pandas DataFrame (port of upstream ``trees_to_dataframe``)."""
        try:
            from pandas import DataFrame as pd_DataFrame  # noqa: PLC0415
        except ImportError:
            raise LightGBMError(
                "This method cannot be run without pandas installed. "
                "You must install pandas and restart your session to use this method."
            ) from None

        if self.num_trees() == 0:
            raise LightGBMError("There are no trees in this Booster and thus nothing to parse")

        def _is_split_node(tree: Dict[str, Any]) -> bool:
            return "split_index" in tree.keys()

        def create_node_record(
            tree: Dict[str, Any],
            node_depth: int = 1,
            tree_index: Optional[int] = None,
            feature_names: Optional[List[str]] = None,
            parent_node: Optional[str] = None,
        ) -> Dict[str, Any]:
            def _get_node_index(tree: Dict[str, Any], tree_index: Optional[int]) -> str:
                tree_num = f"{tree_index}-" if tree_index is not None else ""
                is_split = _is_split_node(tree)
                node_type = "S" if is_split else "L"
                # if a single node tree it won't have `leaf_index` so return 0
                node_num = tree.get("split_index" if is_split else "leaf_index", 0)
                return f"{tree_num}{node_type}{node_num}"

            def _get_split_feature(*, tree: Dict[str, Any], feature_names: Optional[List[str]]) -> Optional[str]:
                if _is_split_node(tree):
                    if feature_names is not None:
                        feature_name = feature_names[tree["split_feature"]]
                    else:
                        feature_name = tree["split_feature"]
                else:
                    feature_name = None
                return feature_name

            def _is_single_node_tree(tree: Dict[str, Any]) -> bool:
                return set(tree.keys()) == {"leaf_value", "leaf_count"}

            node: Dict[str, Union[int, str, None]] = OrderedDict()
            node["tree_index"] = tree_index
            node["node_depth"] = node_depth
            node["node_index"] = _get_node_index(tree, tree_index)
            node["left_child"] = None
            node["right_child"] = None
            node["parent_index"] = parent_node
            node["split_feature"] = _get_split_feature(tree=tree, feature_names=feature_names)
            node["split_gain"] = None
            node["threshold"] = None
            node["decision_type"] = None
            node["missing_direction"] = None
            node["missing_type"] = None
            node["value"] = None
            node["weight"] = None
            node["count"] = None

            if _is_split_node(tree):
                node["left_child"] = _get_node_index(tree["left_child"], tree_index)
                node["right_child"] = _get_node_index(tree["right_child"], tree_index)
                node["split_gain"] = tree["split_gain"]
                node["threshold"] = tree["threshold"]
                node["decision_type"] = tree["decision_type"]
                node["missing_direction"] = "left" if tree["default_left"] else "right"
                node["missing_type"] = tree["missing_type"]
                node["value"] = tree["internal_value"]
                node["weight"] = tree["internal_weight"]
                node["count"] = tree["internal_count"]
            else:
                node["value"] = tree["leaf_value"]
                if not _is_single_node_tree(tree):
                    node["weight"] = tree["leaf_weight"]
                    node["count"] = tree["leaf_count"]

            return node

        def tree_dict_to_node_list(
            tree: Dict[str, Any],
            node_depth: int = 1,
            tree_index: Optional[int] = None,
            feature_names: Optional[List[str]] = None,
            parent_node: Optional[str] = None,
        ) -> List[Dict[str, Any]]:
            node = create_node_record(
                tree=tree,
                node_depth=node_depth,
                tree_index=tree_index,
                feature_names=feature_names,
                parent_node=parent_node,
            )
            res = [node]
            if _is_split_node(tree):
                for child in ["left_child", "right_child"]:
                    res.extend(
                        tree_dict_to_node_list(
                            tree=tree[child],
                            node_depth=node_depth + 1,
                            tree_index=tree_index,
                            feature_names=feature_names,
                            parent_node=node["node_index"],
                        )
                    )
            return res

        model_dict = self.dump_model()
        feature_names = model_dict["feature_names"]
        model_list = []
        for tree in model_dict["tree_info"]:
            model_list.extend(
                tree_dict_to_node_list(
                    tree=tree["tree_structure"], tree_index=tree["tree_index"], feature_names=feature_names
                )
            )

        return pd_DataFrame(model_list, columns=model_list[0].keys())

    def refit(self, *args: Any, **kwargs: Any) -> "Booster":
        raise _unsupported("Booster.refit()")

    def shuffle_models(self, *args: Any, **kwargs: Any) -> "Booster":
        raise _unsupported("Booster.shuffle_models()")

    def get_leaf_output(self, tree_id: int, leaf_id: int) -> float:
        assert self._rs is not None
        return float(self._rs.tree_arrays()[tree_id]["leaf_value"][leaf_id])

    def feature_importance(self, importance_type: str = "split", iteration: Optional[int] = None) -> np.ndarray:
        if iteration is None:
            iteration = self.best_iteration
        if importance_type not in _IMPORTANCE:
            raise ValueError(f"Unknown importance type: {importance_type}")
        assert self._rs is not None
        result = self._rs.feature_importance(int(iteration), _IMPORTANCE[importance_type])
        if importance_type == "split":
            return result.astype(np.int32)
        return result

    def lower_bound(self) -> float:
        """Sum over trees of the smallest leaf value (upstream ``GBDT::GetLowerBoundValue``)."""
        assert self._rs is not None
        total = 0.0
        for t in self._rs.tree_arrays():
            total += min(t["leaf_value"])
        return total

    def upper_bound(self) -> float:
        """Sum over trees of the largest leaf value (upstream ``GBDT::GetUpperBoundValue``)."""
        assert self._rs is not None
        total = 0.0
        for t in self._rs.tree_arrays():
            total += max(t["leaf_value"])
        return total

    # ---- lightgbm-rust extensions (used by differential tests)

    def _tree_arrays(self) -> List[Dict[str, Any]]:
        assert self._rs is not None
        return self._rs.tree_arrays()

    def _last_gradients(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        assert self._rs is not None
        return self._rs.last_gradients()


def _json_default_with_numpy(obj: Any) -> Any:
    """Convert numpy classes to JSON serializable objects."""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def _dump_pandas_categorical(pandas_categorical: Optional[List[List]]) -> str:
    return f"\npandas_categorical:{json.dumps(pandas_categorical)}\n"


def _load_pandas_categorical(model_str: str) -> Optional[List[List]]:
    key = "pandas_categorical:"
    idx = model_str.rfind(key)
    if idx < 0:
        return None
    line = model_str[idx + len(key) :].strip().splitlines()
    return json.loads(line[0]) if line else None
