"""lightgbm-rust: a pure-Rust gradient boosting engine with a LightGBM-like Python API.

The engine does not link to or call upstream LightGBM. It tracks LightGBM
4.7.0; see ``docs/COMPATIBILITY.md`` for what is implemented and verified.
"""

from ._lightgbm_rust import UPSTREAM_COMMIT, UPSTREAM_VERSION, __version__
from .basic import Booster, Dataset, EvalResult, LightGBMError, Sequence, register_logger
from .callback import EarlyStopException, early_stopping, log_evaluation, record_evaluation, reset_parameter
from .engine import CVBooster, cv, train
from .sklearn import LGBMClassifier, LGBMModel, LGBMRanker, LGBMRegressor

__all__ = [
    "Booster",
    "CVBooster",
    "Dataset",
    "EarlyStopException",
    "EvalResult",
    "LGBMClassifier",
    "LGBMModel",
    "LGBMRanker",
    "LGBMRegressor",
    "LightGBMError",
    "Sequence",
    "UPSTREAM_COMMIT",
    "UPSTREAM_VERSION",
    "__version__",
    "cv",
    "early_stopping",
    "log_evaluation",
    "record_evaluation",
    "register_logger",
    "reset_parameter",
    "train",
]
