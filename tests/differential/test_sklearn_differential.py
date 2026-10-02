"""Differential tests of the scikit-learn estimators: lightgbm_rust.sklearn vs lightgbm.sklearn 4.7.0.

The same estimator is fitted with both packages; models, predictions,
evaluation histories and fitted attributes are compared with the tolerances
from tests/tolerances.toml.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

from .conftest import CASES

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

_FIXED = {"n_jobs": 1, "deterministic": True, "force_row_wise": True, "verbosity": -1}


def _case(name: str):
    return next(c for c in CASES if c.name == name)


def _l2_objective(y_true, y_pred):
    return y_pred - y_true, np.ones_like(y_true)


def _logloss_objective(y_true, y_pred, weight):
    p = 1.0 / (1.0 + np.exp(-y_pred))
    grad, hess = p - y_true, p * (1.0 - p)
    if weight is not None:
        grad, hess = grad * weight, hess * weight
    return grad, hess


def _softmax_objective(y_true, y_pred):
    z = y_pred - y_pred.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    onehot = np.eye(y_pred.shape[1])[y_true.astype(int)]
    return p - onehot, 2.0 * p * (1.0 - p)


def _ranking_l2_objective(y_true, y_pred, weight, group):
    assert group is not None and group.sum() == len(y_true)
    return y_pred - y_true, np.ones_like(y_true)


def _mae_metric(y_true, y_pred):
    return "custom_mae", float(np.mean(np.abs(y_true - y_pred))), False


def _two_metrics(y_true, y_pred, weight):
    w = np.ones_like(y_true) if weight is None else weight
    err = float(np.average((y_pred > 0.5) != y_true, weights=w))
    return [("custom_err", err, False), ("custom_mean_pred", float(np.mean(y_pred)), True)]


@dataclass
class SkCase:
    name: str
    estimator: str  # "LGBMRegressor" | "LGBMClassifier" | "LGBMModel"
    data: str  # name of a conftest case
    params: Dict[str, Any] = field(default_factory=dict)
    fit: Dict[str, Any] = field(default_factory=dict)
    labels: Optional[Callable[[np.ndarray], np.ndarray]] = None
    weighted: bool = False
    init_score: bool = False
    eval: bool = True
    legacy_eval: bool = False
    early_stopping: Optional[int] = None
    lr_schedule: bool = False


SK_CASES = [
    SkCase("reg_default", "LGBMRegressor", "reg_basic", {"n_estimators": 30}),
    SkCase("reg_aliases", "LGBMRegressor", "bag_basic",
           {"n_estimators": 25, "subsample": 0.7, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_alpha": 0.5,
            "reg_lambda": 2.0, "min_child_samples": 10, "min_child_weight": 0.01, "min_split_gain": 0.01,
            "random_state": 7, "subsample_for_bin": 1000, "num_leaves": 15, "max_depth": 5}),
    SkCase("reg_eval_metric_early_stopping", "LGBMRegressor", "reg_nan_zero",
           {"n_estimators": 200, "learning_rate": 0.3},
           {"eval_metric": ["l1", _mae_metric]}, early_stopping=5),
    SkCase("reg_weight_init_score", "LGBMRegressor", "reg_weighted", {"n_estimators": 20},
           weighted=True, init_score=True),
    SkCase("reg_objective_l1_importance_gain", "LGBMRegressor", "l1_basic",
           {"n_estimators": 20, "objective": "l1", "importance_type": "gain"}),
    SkCase("reg_custom_objective", "LGBMRegressor", "reg_basic",
           {"n_estimators": 20, "objective": _l2_objective}, {"eval_metric": _mae_metric}),
    SkCase("reg_lr_schedule", "LGBMRegressor", "reg_basic", {"n_estimators": 20}, lr_schedule=True),
    SkCase("bin_string_labels", "LGBMClassifier", "bin_basic", {"n_estimators": 30},
           labels=lambda y: np.where(y > 0, "yes", "no"), legacy_eval=True),
    SkCase("bin_balanced", "LGBMClassifier", "bin_unbalance", {"n_estimators": 25, "class_weight": "balanced"},
           {"eval_metric": ["auc", "binary_error"]}),
    SkCase("bin_weighted_custom_metric", "LGBMClassifier", "bin_weighted", {"n_estimators": 25},
           {"eval_metric": _two_metrics}, weighted=True, early_stopping=3),
    SkCase("bin_custom_objective", "LGBMClassifier", "bin_weighted",
           {"n_estimators": 20, "objective": _logloss_objective}, weighted=True),
    SkCase("mc_default", "LGBMClassifier", "mc_basic", {"n_estimators": 20}, {"eval_metric": "multi_error"}),
    SkCase("mc_class_weight_dict", "LGBMClassifier", "mc_weighted",
           {"n_estimators": 15, "class_weight": {10: 1.0, 11: 2.0, 12: 0.5, 13: 1.5}},
           {"eval_class_weight": [None, {10: 2.0, 11: 1.0, 12: 1.0, 13: 0.5}]}, labels=lambda y: y.astype(int) + 10,
           legacy_eval=True),
    SkCase("mc_ova", "LGBMClassifier", "ova_basic", {"n_estimators": 15, "objective": "multiclassova"}),
    SkCase("mc_custom_objective", "LGBMClassifier", "mc_basic",
           {"n_estimators": 15, "objective": _softmax_objective}),
    SkCase("model_binary", "LGBMModel", "bin_basic", {"n_estimators": 20, "objective": "binary"}),
    SkCase("cat_pandas", "LGBMRegressor", "cat_basic", {"n_estimators": 20}),
]


def _frame(X: np.ndarray, categorical: bool):
    df = pd.DataFrame(X, columns=[f"f_{j}" for j in range(X.shape[1])])
    if categorical:
        for j in (1, 3, 4):
            col = np.where(np.isnan(X[:, j]) | (X[:, j] < 0), np.nan, np.floor(X[:, j]))
            df[f"f_{j}"] = pd.Categorical(col)
    return df


def _fit(mod, sk: SkCase, X, y, Xv, yv, w, s):
    est = getattr(mod, sk.estimator)(**sk.params, **_FIXED)
    fit = dict(sk.fit)
    if sk.legacy_eval:
        # LGBMClassifier label-encodes eval_set targets but passes eval_y through unchanged
        fit.update(eval_set=[(X, y), (Xv, yv)], eval_names=["train", "valid"])
    elif sk.eval:
        fit.update(eval_X=(X, Xv), eval_y=(y, yv), eval_names=["train", "valid"])
    callbacks = []
    if sk.early_stopping:
        callbacks.append(mod.early_stopping(sk.early_stopping, verbose=False))
    if sk.lr_schedule:
        callbacks.append(mod.reset_parameter(learning_rate=lambda i: 0.3 * 0.9 ** i))
    est.fit(X, y, sample_weight=w, init_score=s, callbacks=callbacks or None, **fit)
    return est


@pytest.mark.parametrize("sk", SK_CASES, ids=[c.name for c in SK_CASES])
def test_sklearn_estimators(sk, recorder):
    case = _case(sk.data)
    rec = recorder(sk.name)
    X, Xv, y, yv = case.X, case.Xv, case.y, case.yv
    if sk.data == "cat_basic":
        X, Xv = _frame(X, True), _frame(Xv, True)
    if sk.labels is not None:
        y, yv = sk.labels(y), sk.labels(yv)
    w = case.weight if sk.weighted else None
    s = case.init_score if sk.init_score else None
    rs = _fit(lgb_rs, sk, X, y, Xv, yv, w, s)
    up = _fit(lgb_up, sk, X, y, Xv, yv, w, s)

    rec.compare("model_text", "model_text", rs.booster_.model_to_string(), up.booster_.model_to_string())
    if sk.estimator != "LGBMClassifier":
        rec.compare("predict[holdout]", "predictions", rs.predict(Xv), up.predict(Xv))
    rec.compare("predict(raw_score)[holdout]", "predictions", rs.predict(Xv, raw_score=True),
                up.predict(Xv, raw_score=True))
    rec.compare("predict(pred_leaf)[holdout]", "tree_structure", rs.predict(Xv, pred_leaf=True),
                up.predict(Xv, pred_leaf=True))
    rec.compare("predict(raw_score, start_iteration=2, num_iteration=5)", "predictions",
                rs.predict(Xv, raw_score=True, start_iteration=2, num_iteration=5),
                up.predict(Xv, raw_score=True, start_iteration=2, num_iteration=5))
    if sk.estimator == "LGBMClassifier":
        if callable(sk.params.get("objective")):
            with pytest.warns(UserWarning, match="Cannot compute class probabilities or labels"):
                p_rs = rs.predict_proba(Xv)
        else:
            p_rs = rs.predict_proba(Xv)
            rec.compare("classes_", "tree_structure", list(rs.classes_), list(up.classes_))
            rec.compare("n_classes_", "tree_structure", rs.n_classes_, up.n_classes_)
        with pytest.warns(UserWarning) if callable(sk.params.get("objective")) else _nullcontext():
            p_up = up.predict_proba(Xv)
        rec.compare("predict_proba[holdout]", "predictions", p_rs, p_up)
        if callable(sk.params.get("objective")):
            with pytest.warns(UserWarning):
                rec.compare("predict[holdout]", "predictions", rs.predict(Xv), up.predict(Xv))
        else:
            rec.compare("predict labels[holdout]", "tree_structure", list(rs.predict(Xv)), list(up.predict(Xv)))
    for attr in ("n_features_", "n_features_in_", "n_estimators_", "n_iter_", "best_iteration_", "objective_"):
        a, b = getattr(rs, attr), getattr(up, attr)
        if callable(b):
            a, b = a.__name__, b.__name__
        rec.compare(attr, "tree_structure", a, b)
    rec.compare("feature_name_", "tree_structure", rs.feature_name_, up.feature_name_)
    rec.compare("feature_importances_", "predictions", rs.feature_importances_, up.feature_importances_)
    rec.compare("evals_result_ keys", "tree_structure",
                {k: sorted(v) for k, v in rs.evals_result_.items()}, {k: sorted(v) for k, v in up.evals_result_.items()})
    for ds, metrics in up.evals_result_.items():
        for m, hist in metrics.items():
            rec.compare(f"evals_result_[{ds}][{m}]", "metrics", rs.evals_result_.get(ds, {}).get(m, []), hist)
    rec.compare("best_score_ keys", "tree_structure", {k: sorted(v) for k, v in rs.best_score_.items()},
                {k: sorted(v) for k, v in up.best_score_.items()})
    for ds, metrics in up.best_score_.items():
        for m, v in metrics.items():
            rec.compare(f"best_score_[{ds}][{m}]", "metrics", rs.best_score_[ds][m], v)
    if hasattr(up, "feature_names_in_"):
        rec.compare("feature_names_in_", "tree_structure", list(rs.feature_names_in_), list(up.feature_names_in_))
    else:
        assert not hasattr(rs, "feature_names_in_")
    rec.compare("get_params", "tree_structure", _plain_params(rs.get_params()), _plain_params(up.get_params()))
    rec.finish()


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _plain_params(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: (v.__name__ if callable(v) else v) for k, v in p.items()}


def _rank_case():
    from .test_differential import RANK_CASES

    return next(c for c in RANK_CASES if c.name == "rank_weighted")


@pytest.mark.parametrize("variant", ["default", "eval_at", "custom_objective"])
def test_sklearn_ranker(variant, recorder):
    case = _rank_case()
    rec = recorder(f"ranker_{variant}")
    params: Dict[str, Any] = {"n_estimators": 20}
    fit: Dict[str, Any] = {}
    if variant == "eval_at":
        params.update(min_child_samples=5)
        fit.update(eval_at=[1, 3, 7], eval_metric="map")
    if variant == "custom_objective":
        params["objective"] = _ranking_l2_objective
    out = {}
    for mod in (lgb_rs, lgb_up):
        est = mod.LGBMRanker(**params, **_FIXED)
        est.fit(case.X, case.y, group=case.group, sample_weight=case.weight, eval_X=(case.Xv,), eval_y=(case.yv,),
                eval_group=[case.groupv], callbacks=[mod.early_stopping(4, verbose=False)], **fit)
        out[mod.__name__] = est
    rs, up = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("model_text", "model_text", rs.booster_.model_to_string(), up.booster_.model_to_string())
    rec.compare("predict[holdout]", "predictions", rs.predict(case.Xv), up.predict(case.Xv))
    rec.compare("best_iteration_", "tree_structure", rs.best_iteration_, up.best_iteration_)
    for ds, metrics in up.evals_result_.items():
        for m, hist in metrics.items():
            rec.compare(f"evals_result_[{ds}][{m}]", "metrics", rs.evals_result_.get(ds, {}).get(m, []), hist)
    rec.finish()


def test_sklearn_refit_continued_and_pickle(recorder, tmp_path):
    """init_model=<fitted estimator>, set_params + refit, pickling, and sklearn.base.clone."""
    import pickle

    from sklearn.base import clone

    case = _case("bin_weighted")
    rec = recorder("refit_continued_pickle")
    out = {}
    for mod in (lgb_rs, lgb_up):
        base = mod.LGBMClassifier(n_estimators=10, **_FIXED).fit(case.X, case.y)
        cont = mod.LGBMClassifier(n_estimators=7, learning_rate=0.05, **_FIXED).fit(case.X, case.y, init_model=base)
        again = clone(base).set_params(num_leaves=7, n_estimators=12).fit(case.X, case.y)
        loaded = pickle.loads(pickle.dumps(cont))
        out[mod.__name__] = (cont, again, loaded)
    (c_rs, a_rs, l_rs), (c_up, a_up, l_up) = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("continued model_text", "model_text", c_rs.booster_.model_to_string(), c_up.booster_.model_to_string())
    rec.compare("cloned+set_params model_text", "model_text", a_rs.booster_.model_to_string(),
                a_up.booster_.model_to_string())
    rec.compare("unpickled predict_proba", "predictions", l_rs.predict_proba(case.Xv), l_up.predict_proba(case.Xv))
    rec.compare("unpickled n_iter_", "tree_structure", l_rs.n_iter_, l_up.n_iter_)
    rec.finish()


def test_sklearn_pred_contrib(recorder):
    case = _case("mc_basic")
    rec = recorder("pred_contrib")
    rs = lgb_rs.LGBMClassifier(n_estimators=10, **_FIXED).fit(case.X, case.y)
    up = lgb_up.LGBMClassifier(n_estimators=10, **_FIXED).fit(case.X, case.y)
    rec.compare("predict_proba(pred_contrib)", "predictions", rs.predict_proba(case.Xv, pred_contrib=True),
                up.predict_proba(case.Xv, pred_contrib=True))
    rec.finish()


def test_sklearn_errors():
    """Errors raised by both packages for the same misuse."""
    case = _case("reg_basic")
    for mod in (lgb_rs, lgb_up):
        with pytest.raises(mod.compat.LGBMNotFittedError):
            mod.LGBMRegressor().predict(case.X)
        est = mod.LGBMRegressor(n_estimators=2, **_FIXED).fit(case.X, case.y)
        with pytest.raises(ValueError, match="features"):
            est.predict(case.X[:, :3])
        with pytest.raises(ValueError, match="Specify either 'eval_set' or 'eval_X'"):
            with pytest.warns(mod.basic.LGBMDeprecationWarning):
                mod.LGBMRegressor(n_estimators=2).fit(case.X, case.y, eval_set=[(case.Xv, case.yv)],
                                                      eval_X=case.Xv, eval_y=case.yv)
        with pytest.raises(TypeError, match="Self-defined objective function should have 2, 3 or 4 arguments"):
            mod.LGBMRegressor(n_estimators=2, objective=lambda a: a, **_FIXED).fit(case.X, case.y)
        labels, labels_v = np.where(case.y > 0, "a", "b"), np.where(case.yv > 0, "a", "b")
        with pytest.raises(ValueError, match="could not convert string to float"):
            mod.LGBMClassifier(n_estimators=2, **_FIXED).fit(case.X, labels, eval_X=case.Xv, eval_y=labels_v)
        with pytest.raises(ValueError, match="eval_group cannot be None if any of eval_set"):
            mod.LGBMRanker(n_estimators=2).fit(case.X, np.zeros(len(case.y)), group=[len(case.y)],
                                               eval_X=case.Xv, eval_y=case.yv)
