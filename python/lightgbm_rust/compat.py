"""Optional-dependency shims, following upstream ``python-package/lightgbm/compat.py`` (4.7.0).

Only the names used by ``cv()`` are provided so far.

Upstream code is Copyright Microsoft Corporation, MIT License (see NOTICE).
"""

try:
    from sklearn.model_selection import BaseCrossValidator, GroupKFold, StratifiedKFold

    SKLEARN_INSTALLED = True
    _LGBMBaseCrossValidator = BaseCrossValidator
    _LGBMStratifiedKFold = StratifiedKFold
    _LGBMGroupKFold = GroupKFold
except ImportError:
    SKLEARN_INSTALLED = False
    _LGBMBaseCrossValidator = None
    _LGBMStratifiedKFold = None
    _LGBMGroupKFold = None
