"""Shared helpers for differential tests against upstream LightGBM 4.7.0.

Every comparison goes through ``Recorder.compare``, which applies the
tolerance from ``tests/tolerances.toml``, records the measured maximum
absolute/relative differences, and fails the test if the tolerance is
exceeded. Records are written to ``tests/report/differential.json`` for the
summary report.
"""

from __future__ import annotations

import json
import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

ROOT = Path(__file__).resolve().parents[2]
TOLERANCES: Dict[str, Dict[str, Any]] = tomllib.loads((ROOT / "tests" / "tolerances.toml").read_text())
REPORT = ROOT / "tests" / "report" / "differential.json"

assert lgb_up.__version__ == "4.7.0", f"reference must be lightgbm 4.7.0, found {lgb_up.__version__}"
assert not hasattr(lgb_rs.basic, "_LIB"), "lightgbm_rust must not load the upstream library"

DETERMINISTIC = {"deterministic": True, "force_row_wise": True, "num_threads": 1, "verbosity": -1}


@dataclass
class Record:
    test: str
    case: str
    quantity: str
    tolerance: str
    n: int
    max_abs: float
    max_rel: float
    exact: bool
    passed: bool
    note: str = ""


_RECORDS: List[Record] = []


def _as_float_array(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).ravel()


@dataclass
class Recorder:
    test: str
    case: str
    failures: List[str] = field(default_factory=list)

    def compare(self, quantity: str, tolerance: str, rust: Any, upstream: Any, note: str = "") -> bool:
        tol = TOLERANCES[tolerance]

        def numeric(x: Any) -> bool:
            try:
                _as_float_array(x)
                return True
            except (TypeError, ValueError):
                return False

        if isinstance(rust, str) or isinstance(upstream, str) or not (numeric(rust) and numeric(upstream)):
            exact = rust == upstream
            rec = Record(self.test, self.case, quantity, tolerance, 1, 0.0 if exact else math.inf,
                         0.0 if exact else math.inf, exact, exact, note)
            if not exact:
                detail = _first_text_diff(upstream, rust) if isinstance(rust, str) else f"upstream={upstream!r} rust={rust!r}"
                rec.note = (note + " " + detail).strip()
        else:
            a = _as_float_array(rust)
            b = _as_float_array(upstream)
            if a.shape != b.shape:
                rec = Record(self.test, self.case, quantity, tolerance, int(b.size), math.inf, math.inf, False, False,
                             f"shape mismatch: rust {a.shape} vs upstream {b.shape}")
            else:
                both_nan = np.isnan(a) & np.isnan(b)
                diff = np.where(both_nan, 0.0, np.abs(a - b))
                diff = np.where(np.isnan(diff), np.inf, diff)
                denom = np.abs(b)
                rel = np.divide(diff, denom, out=np.where(diff > 0, np.inf, 0.0), where=denom > 0)
                exact = bool(np.all((a == b) | both_nan))
                if tol["mode"] == "exact":
                    passed = exact
                else:
                    passed = bool(np.all(diff <= tol["atol"] + tol["rtol"] * denom))
                rec = Record(self.test, self.case, quantity, tolerance, int(b.size),
                             float(diff.max(initial=0.0)), float(rel.max(initial=0.0)), exact, passed, note)
        _RECORDS.append(rec)
        if not rec.passed:
            self.failures.append(
                f"{quantity} [{tolerance}]: max_abs={rec.max_abs:.3e} max_rel={rec.max_rel:.3e} {rec.note}".strip()
            )
        return rec.passed

    def finish(self) -> None:
        if self.failures:
            pytest.fail(f"{self.case}: " + "; ".join(self.failures), pytrace=False)


def _first_text_diff(a: str, b: str) -> str:
    al, bl = a.splitlines(), b.splitlines()
    for i, (x, y) in enumerate(zip(al, bl)):
        if x != y:
            return f"first difference at line {i + 1}: upstream={x[:120]!r} rust={y[:120]!r}"
    if len(al) != len(bl):
        return f"line count differs: upstream={len(al)} rust={len(bl)}"
    return "texts differ"


@pytest.fixture
def recorder(request: pytest.FixtureRequest) -> Callable[[str], Recorder]:
    def make(case: str) -> Recorder:
        return Recorder(request.node.originalname or request.node.name, case)

    return make


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _RECORDS:
        return
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"upstream_version": lgb_up.__version__, "records": [asdict(r) for r in _RECORDS]},
        indent=1,
        allow_nan=True,
    ))


# --------------------------------------------------------------------------- cases


@dataclass
class Case:
    name: str
    objective: str
    params: Dict[str, Any]
    X: np.ndarray
    y: np.ndarray
    Xv: np.ndarray
    yv: np.ndarray
    weight: Optional[np.ndarray] = None
    init_score: Optional[np.ndarray] = None
    num_boost_round: int = 30

    @property
    def full_params(self) -> Dict[str, Any]:
        return {"objective": self.objective, **DETERMINISTIC, **self.params}


def _features(rng: np.random.Generator, n: int, p: int, kind: str) -> np.ndarray:
    X = rng.normal(size=(n, p))
    if kind == "nan_zero":
        X[rng.random(X.shape) < 0.1] = np.nan
        X[rng.random(X.shape) < 0.3] = 0.0
    elif kind == "discrete":
        X = rng.integers(0, 6, size=(n, p)).astype(np.float64)
        X[:, -1] = 3.0  # constant -> trivial feature, dropped from training
        X[:, 0] = np.round(rng.normal(size=n), 1)  # many ties
    elif kind == "heavy_tail":
        X = rng.standard_cauchy(size=(n, p))
        X[:, 1] = np.exp(rng.normal(size=n) * 3)
    elif kind == "sparse":
        # mostly-zero, nearly exclusive columns: exercises upstream's feature bundling
        slot = rng.integers(0, p, size=n)
        X = np.where(slot[:, None] == np.arange(p)[None, :], X, 0.0)
        X[:, 0] = rng.normal(size=n)
        X[rng.random(X.shape) < 0.01] = 1.5
    elif kind == "categorical":
        # col 1: 30 categories (+NaN, negatives -> NaN, fractional values truncated);
        # col 3: 3 categories; col 4: 100 categories with a long tail of rare ones
        X[:, 1] = rng.integers(0, 30, size=n)
        X[rng.random(n) < 0.05, 1] = np.nan
        X[rng.random(n) < 0.01, 1] = -2.0
        X[rng.random(n) < 0.05, 1] += 0.4
        X[:, 3] = rng.integers(0, 3, size=n)
        X[:, 4] = np.minimum(rng.geometric(0.05, size=n) - 1, 99)
    return X


REGRESSION_OBJECTIVES = ("regression", "regression_l1", "huber", "fair", "poisson", "quantile", "mape", "gamma",
                         "tweedie")


MULTICLASS_OBJECTIVES = ("multiclass", "multiclassova")


def _target(rng: np.random.Generator, X: np.ndarray, objective: str, num_labels: int = 1,
            kind: str = "normal") -> np.ndarray:
    Z = np.nan_to_num(X)
    if kind == "categorical":
        # unordered category effects (fixed pseudo-random lookup tables)
        effect = np.random.default_rng(99).normal(size=(3, 100))
        Z = Z.copy()
        Z[:, 1] = effect[0, np.clip(Z[:, 1].astype(int), 0, 99)]
        Z[:, 2] = Z[:, 2] + effect[1, Z[:, 3].astype(int)] + 0.5 * effect[2, Z[:, 4].astype(int)]
    f = np.tanh(Z[:, 0]) * 2 + 0.5 * Z[:, 1] - 0.3 * Z[:, 2] * (Z[:, 0] > 0)
    n = len(f)
    if objective in MULTICLASS_OBJECTIVES:
        # class c has logit (c - mid) * f + 0.5 * Z[:, c mod p]; sample from the softmax
        mid = (num_labels - 1) / 2
        logits = np.stack([(c - mid) * f + 0.5 * Z[:, c % Z.shape[1]] for c in range(num_labels)], axis=1)
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        u = rng.random(n)[:, None]
        return np.minimum((prob.cumsum(axis=1) < u).sum(axis=1), num_labels - 1).astype(np.float64)
    if objective in ("regression", "regression_l1", "huber", "fair", "quantile"):
        return f + rng.normal(scale=0.5, size=n)
    if objective == "mape":
        # spans |label| < 1 (clamped to 1 in the MAPE weights) and larger values
        return (f + rng.normal(scale=0.5, size=n)) * 3
    if objective == "poisson":
        return rng.poisson(np.exp(f / 2)).astype(np.float64)
    if objective == "gamma":
        return rng.gamma(2.0, np.exp(f / 2) / 2.0)
    if objective == "tweedie":
        # compound Poisson-gamma: exact zeros plus a continuous positive part
        counts = rng.poisson(np.exp(f / 2) * 0.8)
        return np.array([rng.gamma(2.0, 0.5, size=k).sum() for k in counts])
    if objective in ("cross_entropy", "cross_entropy_lambda", "xentropy"):
        # soft labels in [0, 1], about 10% clipped to exactly 0 or 1
        return np.clip(1 / (1 + np.exp(-f)) + rng.normal(scale=0.2, size=n), 0.0, 1.0)
    return (rng.random(n) < 1 / (1 + np.exp(-f))).astype(np.float64)


def make_case(name: str, objective: str, params: Dict[str, Any], *, n: int = 3000, p: int = 6,
              kind: str = "normal", weighted: bool = False, init_score: bool = False, seed: int = 0,
              rounds: int = 30, imbalance: float = 0.0, num_labels: Optional[int] = None) -> Case:
    """`num_labels` (multiclass): distinct labels generated; defaults to `num_class`."""
    rng = np.random.default_rng(seed)
    k = params.get("num_class", 1)
    num_labels = k if num_labels is None else num_labels
    X = _features(rng, n, p, kind)
    Xv = _features(rng, n // 3, p, kind)
    y = _target(rng, X, objective, num_labels, kind)
    yv = _target(rng, Xv, objective, num_labels, kind)
    if imbalance:
        keep = (y == 1) | (rng.random(n) > imbalance)
        X, y = X[keep], y[keep]
    w = rng.uniform(0.2, 3.0, size=len(y)) if weighted else None
    shape = (len(y), k) if k > 1 else len(y)
    s = rng.normal(scale=0.3, size=shape) if init_score else None
    return Case(name, objective, params, X, y, Xv, yv, w, s, rounds)


_MC = [1, -1, 1, 0, -1, 0]
_IC = [[0, 1], [2, 3], [4, 5]]
_IC_OVERLAP = [[0, 1, 2], [2, 3], [4]]
_RF_BAG = {"boosting": "rf", "bagging_freq": 1, "bagging_fraction": 0.7}
_FORCED_A = str(ROOT / "tests" / "differential" / "data" / "forced_bins_a.json")
_FORCED_CAT = str(ROOT / "tests" / "differential" / "data" / "forced_bins_categorical.json")
_CEGB_LAZY = [0.1, 0.5, 0.02, 1.0, 0.3, 0.05]
_CEGB_COUPLED = [50.0, 200.0, 10.0, 400.0, 30.0, 5.0]

CASES = [
    make_case("reg_basic", "regression", {}),
    make_case("reg_nan_zero", "regression", {}, kind="nan_zero"),
    make_case("reg_zero_as_missing", "regression", {"zero_as_missing": True}, kind="nan_zero"),
    make_case("reg_no_missing", "regression", {"use_missing": False}, kind="nan_zero"),
    make_case("reg_discrete", "regression", {"min_data_in_bin": 1}, kind="discrete"),
    make_case("reg_heavy_tail", "regression", {"metric": ["l1", "rmse"]}, kind="heavy_tail"),
    make_case("reg_weighted", "regression", {}, weighted=True),
    make_case("reg_regularized", "regression",
              {"lambda_l1": 1.0, "lambda_l2": 5.0, "min_gain_to_split": 0.1, "max_delta_step": 0.5,
               "path_smooth": 2.0, "min_sum_hessian_in_leaf": 1.0}),
    make_case("reg_depth", "regression", {"max_depth": 3, "num_leaves": 50, "min_data_in_leaf": 5}),
    make_case("reg_depth_without_num_leaves", "regression", {"max_depth": 2}),
    make_case("reg_sqrt", "regression", {"reg_sqrt": True}),
    make_case("reg_init_score", "regression", {}, init_score=True),
    make_case("reg_no_boost_from_average", "regression", {"boost_from_average": False}),
    make_case("reg_sampled_bins", "regression",
              {"bin_construct_sample_cnt": 500, "max_bin": 31, "min_data_in_bin": 5, "data_random_seed": 7}),
    make_case("reg_max_bin_1023", "regression", {"max_bin": 1023}, n=6000),
    make_case("reg_seed", "regression", {"seed": 123, "learning_rate": 0.3, "num_leaves": 7}),
    make_case("reg_100_rounds", "regression", {"num_leaves": 63}, n=8000, rounds=100),
    make_case("bin_basic", "binary", {"metric": ["binary_logloss", "auc", "binary_error"]}),
    make_case("bin_nan_zero", "binary", {}, kind="nan_zero"),
    make_case("bin_weighted", "binary", {}, weighted=True),
    make_case("bin_unbalance", "binary", {"is_unbalance": True}, imbalance=0.8),
    make_case("bin_scale_pos_weight", "binary", {"scale_pos_weight": 3.0}),
    make_case("bin_sigmoid", "binary", {"sigmoid": 0.7}),
    make_case("bin_init_score", "binary", {}, init_score=True),
    make_case("bin_100_rounds", "binary", {"num_leaves": 63, "min_data_in_leaf": 10}, n=8000, rounds=100),
    make_case("reg_loss_metrics", "regression", {"metric": ["quantile", "huber", "fair", "mape"], "alpha": 0.7}),
    make_case("l1_basic", "regression_l1", {}),
    make_case("l1_weighted", "regression_l1", {}, weighted=True),
    make_case("l1_sqrt", "regression_l1", {"reg_sqrt": True}),
    make_case("l1_no_boost_from_average", "regression_l1", {"boost_from_average": False}),
    make_case("huber_basic", "huber", {}),
    make_case("huber_weighted", "huber", {"alpha": 0.5}, weighted=True),
    make_case("fair_basic", "fair", {"fair_c": 0.5}),
    make_case("fair_weighted", "fair", {}, weighted=True),
    make_case("poisson_basic", "poisson", {}),
    make_case("poisson_weighted", "poisson", {"poisson_max_delta_step": 0.3}, weighted=True),
    make_case("quantile_basic", "quantile", {"alpha": 0.3}),
    make_case("quantile_weighted", "quantile", {}, weighted=True),
    make_case("quantile_init_score", "quantile", {"alpha": 0.6}, init_score=True),
    make_case("mape_basic", "mape", {}),
    make_case("mape_weighted", "mape", {}, weighted=True),
    make_case("gamma_basic", "gamma", {"metric": ["gamma", "gamma_deviance"]}),
    make_case("gamma_weighted", "gamma", {}, weighted=True),
    make_case("tweedie_basic", "tweedie", {"tweedie_variance_power": 1.2}),
    make_case("tweedie_weighted", "tweedie", {"metric": ["tweedie", "poisson", "l2"]}, weighted=True),
    # upstream xentropy_objective.hpp / xentropy_metric.hpp
    make_case("xent_basic", "cross_entropy", {"metric": ["cross_entropy", "kullback_leibler", "cross_entropy_lambda"]}),
    make_case("xent_weighted", "cross_entropy", {"metric": ["cross_entropy", "kullback_leibler"]}, weighted=True),
    make_case("xent_init_score", "cross_entropy", {}, init_score=True),
    make_case("xent_no_boost_from_average", "cross_entropy", {"boost_from_average": False}, kind="nan_zero"),
    make_case("xent_alias", "xentropy", {"metric": "kldiv"}),
    make_case("xentlambda_basic", "cross_entropy_lambda",
              {"metric": ["cross_entropy_lambda", "cross_entropy", "kullback_leibler"]}),
    make_case("xentlambda_weighted", "cross_entropy_lambda", {"metric": ["cross_entropy_lambda", "cross_entropy"]},
              weighted=True),
    make_case("xentlambda_init_score", "cross_entropy_lambda", {}, init_score=True, weighted=True),
    make_case("mc_basic", "multiclass", {"num_class": 3, "metric": ["multi_logloss", "multi_error"]}),
    make_case("mc_weighted", "multiclass", {"num_class": 4}, weighted=True),
    make_case("mc_top_k", "multiclass",
              {"num_class": 5, "metric": ["multi_error", "multi_logloss"], "multi_error_top_k": 2}),
    make_case("mc_absent_class", "multiclass", {"num_class": 4}, num_labels=3),
    make_case("mc_init_score", "multiclass", {"num_class": 3}, init_score=True),
    make_case("mc_nan_zero", "multiclass", {"num_class": 3, "num_leaves": 15}, kind="nan_zero"),
    make_case("mc_60_rounds", "multiclass", {"num_class": 3, "num_leaves": 63}, n=8000, rounds=60),
    make_case("ova_basic", "multiclassova", {"num_class": 3, "metric": ["multi_logloss", "multi_error"]}),
    make_case("ova_weighted_sigmoid", "multiclassova", {"num_class": 4, "sigmoid": 0.7}, weighted=True),
    make_case("ova_unbalance", "multiclassova", {"num_class": 3, "is_unbalance": True}),
    make_case("ova_absent_class", "multiclassova", {"num_class": 4}, num_labels=3),
    # sampling (upstream bagging.hpp / goss.hpp / col_sampler.hpp / extra trees)
    make_case("bag_basic", "regression", {"bagging_fraction": 0.7, "bagging_freq": 1}),
    make_case("bag_freq3_subset", "regression", {"bagging_fraction": 0.5, "bagging_freq": 3, "bagging_seed": 7}),
    make_case("bag_small_fraction", "regression", {"bagging_fraction": 0.3, "bagging_freq": 1, "min_data_in_leaf": 5}),
    make_case("bag_seed_derived", "regression",
              {"seed": 5, "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8}),
    make_case("bag_balanced_binary", "binary",
              {"pos_bagging_fraction": 0.8, "neg_bagging_fraction": 0.4, "bagging_freq": 1}),
    make_case("bag_l1_weighted", "regression_l1", {"bagging_fraction": 0.6, "bagging_freq": 2}, weighted=True),
    make_case("bag_quantile", "quantile", {"bagging_fraction": 0.75, "bagging_freq": 1, "alpha": 0.4}),
    make_case("bag_multiclass", "multiclass", {"num_class": 3, "bagging_fraction": 0.7, "bagging_freq": 1}),
    make_case("ff_bytree", "regression", {"feature_fraction": 0.6}),
    make_case("ff_bynode", "regression", {"feature_fraction_bynode": 0.5}),
    make_case("ff_both", "binary",
              {"feature_fraction": 0.7, "feature_fraction_bynode": 0.6, "feature_fraction_seed": 11}, p=10),
    make_case("ff_multiclass", "multiclass", {"num_class": 3, "feature_fraction": 0.5}),
    make_case("extra_trees", "regression", {"extra_trees": True}),
    make_case("extra_trees_nan", "binary", {"extra_trees": True, "extra_seed": 3}, kind="nan_zero"),
    make_case("extra_trees_sparse", "regression", {"extra_trees": True}, kind="sparse", p=12),
    make_case("reg_sparse", "regression", {}, kind="sparse", p=12),
    make_case("goss_basic", "regression", {"data_sample_strategy": "goss", "learning_rate": 0.2}),
    make_case("goss_no_subset", "binary",
              {"data_sample_strategy": "goss", "top_rate": 0.4, "other_rate": 0.2, "learning_rate": 0.25}),
    make_case("goss_boosting_alias", "multiclass", {"boosting": "goss", "num_class": 3, "learning_rate": 0.3}),
    make_case("sampling_combo", "regression",
              {"bagging_fraction": 0.8, "bagging_freq": 2, "feature_fraction": 0.8, "feature_fraction_bynode": 0.7,
               "extra_trees": True, "num_leaves": 15}, n=6000, p=8, weighted=True),
    # categorical features (upstream bin.cpp categorical FindBin, feature_histogram.hpp categorical split search)
    make_case("cat_basic", "regression", {"categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("cat_onehot", "regression", {"categorical_feature": "1,3", "max_cat_to_onehot": 32},
              kind="categorical"),
    make_case("cat_params", "regression",
              {"categorical_feature": "1,4", "cat_smooth": 1.0, "cat_l2": 1.0, "max_cat_threshold": 8,
               "min_data_per_group": 10, "min_data_in_leaf": 5}, kind="categorical"),
    make_case("cat_binary", "binary", {"categorical_feature": "1,3,4", "metric": ["binary_logloss", "auc"]},
              kind="categorical"),
    make_case("cat_multiclass", "multiclass", {"num_class": 3, "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("cat_weighted", "regression", {"categorical_feature": "1,3,4"}, kind="categorical", weighted=True),
    make_case("cat_extra_trees", "regression", {"categorical_feature": "1,3,4", "extra_trees": True},
              kind="categorical"),
    make_case("cat_sampling", "regression",
              {"categorical_feature": "1,3,4", "bagging_fraction": 0.7, "bagging_freq": 1,
               "feature_fraction_bynode": 0.7}, kind="categorical"),
    make_case("cat_max_bin", "regression", {"categorical_feature": "1,4", "max_bin": 15, "min_data_in_bin": 1},
              kind="categorical"),
    make_case("cat_no_missing", "regression", {"categorical_feature": "1,3,4", "use_missing": False},
              kind="categorical"),
    make_case("cat_100_rounds", "binary", {"categorical_feature": "1,3,4", "num_leaves": 63, "min_data_in_leaf": 10},
              kind="categorical", n=8000, rounds=100),
    # monotone constraints (upstream monotone_constraints.hpp); several oppose the target's trend
    make_case("mono_basic", "regression", {"monotone_constraints": _MC}),
    make_case("mono_intermediate", "regression", {"monotone_constraints": _MC, "mc_method": "intermediate"}),
    make_case("mono_advanced", "regression", {"monotone_constraints": _MC, "mc_method": "advanced"}),
    make_case("mono_penalty", "regression", {"monotone_constraints": _MC, "mc_method": "advanced",
                                             "monotone_penalty": 1.5}),
    make_case("mono_penalty_basic", "binary", {"monotone_constraints": _MC, "monotone_penalty": 0.5}, weighted=True),
    make_case("mono_regularized", "regression",
              {"monotone_constraints": _MC, "mc_method": "intermediate", "lambda_l1": 1.0, "lambda_l2": 5.0,
               "max_delta_step": 0.5, "path_smooth": 2.0, "min_gain_to_split": 0.1}),
    make_case("mono_adv_regularized", "regression",
              {"monotone_constraints": _MC, "mc_method": "advanced", "lambda_l1": 1.0, "lambda_l2": 5.0,
               "max_delta_step": 0.5, "path_smooth": 2.0}, weighted=True),
    make_case("mono_binary", "binary", {"monotone_constraints": _MC, "mc_method": "advanced"}, weighted=True),
    make_case("mono_multiclass", "multiclass", {"num_class": 3, "monotone_constraints": _MC,
                                                "mc_method": "intermediate"}),
    make_case("mono_nan_zero", "regression", {"monotone_constraints": _MC, "mc_method": "advanced"},
              kind="nan_zero"),
    make_case("mono_discrete", "regression", {"monotone_constraints": _MC, "mc_method": "advanced",
                                              "min_data_in_bin": 1}, kind="discrete"),
    make_case("mono_categorical", "regression",
              {"monotone_constraints": [1, 0, -1, 0, 0, -1], "mc_method": "advanced",
               "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("mono_extra_trees", "regression", {"monotone_constraints": _MC, "mc_method": "advanced",
                                                 "extra_trees": True}),
    make_case("mono_bynode_fallback", "regression", {"monotone_constraints": _MC, "mc_method": "advanced",
                                                     "feature_fraction_bynode": 0.7}),
    make_case("mono_bagging", "regression", {"monotone_constraints": _MC, "mc_method": "intermediate",
                                             "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.8}),
    make_case("mono_depth", "regression", {"monotone_constraints": _MC, "mc_method": "intermediate",
                                           "max_depth": 4, "num_leaves": 15, "monotone_penalty": 2.0}),
    make_case("mono_100_rounds", "regression", {"monotone_constraints": _MC, "mc_method": "advanced",
                                                "num_leaves": 63}, n=8000, rounds=100),
    # feature_contri (upstream FeatureHistogram::FindBestThreshold, gain *= meta_->penalty)
    make_case("contri_basic", "regression", {"feature_contri": [1.0, 0.5, 0.2, 1.5, 1.0, 0.8]}),
    make_case("contri_zero", "binary", {"feature_contri": [0.0, 1.0, 1.0, 0.3, 1.0, 1.0]}, weighted=True),
    make_case("contri_categorical", "regression", {"feature_contri": [0.7, 0.4, 1.0, 2.0, 0.5, 1.0],
                                                   "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("contri_monotone", "regression", {"feature_contri": [0.6, 1.0, 0.9, 1.0, 0.3, 1.2],
                                                "monotone_constraints": _MC, "mc_method": "advanced",
                                                "monotone_penalty": 1.0}),
    make_case("contri_multiclass", "multiclass", {"num_class": 3, "feature_contri": [0.9, 0.3, 1.0, 1.0, 0.6, 1.1],
                                                  "extra_trees": True}),
    # upstream DatasetLoader::GetForcedBins, BinMapper::FindBin with forced bounds, max_bin_by_feature
    make_case("bins_forced", "regression", {"forcedbins_filename": _FORCED_A, "max_bin": 15}),
    make_case("bins_forced_wide", "binary", {"forcedbins_filename": _FORCED_A}, weighted=True),
    make_case("bins_forced_nan_zero", "regression", {"forcedbins_filename": _FORCED_A, "max_bin": 12},
              kind="nan_zero"),
    make_case("bins_forced_zero_as_missing", "regression", {"forcedbins_filename": _FORCED_A, "max_bin": 9,
                                                            "zero_as_missing": True}, kind="nan_zero"),
    make_case("bins_forced_categorical", "regression", {"forcedbins_filename": _FORCED_CAT, "max_bin": 6,
                                                        "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("bins_by_feature", "regression", {"max_bin_by_feature": [4, 255, 16, 63, 2, 7]}, kind="nan_zero"),
    make_case("bins_by_feature_categorical", "binary", {"max_bin_by_feature": [8, 10, 32, 3, 40, 255],
                                                        "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("bins_by_feature_forced", "regression", {"max_bin_by_feature": [5, 3, 4, 300, 2, 6],
                                                       "forcedbins_filename": _FORCED_A, "min_data_in_bin": 1},
              kind="discrete"),
    # CEGB (upstream cost_effective_gradient_boosting.hpp); lazy bits are numbered by bag position in subset
    # mode (bagging_fraction 0.4, GOSS 0.2 + 0.1), where the bag size changes every tree
    make_case("cegb_split", "regression", {"cegb_penalty_split": 0.05}),
    make_case("cegb_coupled", "regression", {"cegb_penalty_feature_coupled": _CEGB_COUPLED}),
    make_case("cegb_lazy", "regression", {"cegb_penalty_feature_lazy": _CEGB_LAZY}),
    make_case("cegb_all", "binary", {"cegb_penalty_split": 0.001, "cegb_penalty_feature_coupled": [2, 8, 1, 20, 3, 0.5],
                                     "cegb_penalty_feature_lazy": [0.002, 0.01, 0.0, 0.02, 0.005, 0.001],
                                     "cegb_tradeoff": 0.5}, weighted=True),
    make_case("cegb_tradeoff_only", "regression", {"cegb_tradeoff": 0.3}),
    make_case("cegb_lazy_bagging", "regression", {"cegb_penalty_feature_lazy": _CEGB_LAZY, "bagging_fraction": 0.8,
                                                  "bagging_freq": 1}),
    make_case("cegb_lazy_subset", "regression", {"cegb_penalty_feature_lazy": _CEGB_LAZY, "bagging_fraction": 0.4,
                                                 "bagging_freq": 1}),
    make_case("cegb_goss", "regression", {"cegb_penalty_feature_lazy": _CEGB_LAZY,
                                          "cegb_penalty_feature_coupled": _CEGB_COUPLED,
                                          "data_sample_strategy": "goss", "top_rate": 0.2, "other_rate": 0.1}),
    make_case("cegb_monotone", "regression", {"cegb_penalty_feature_coupled": _CEGB_COUPLED,
                                              "monotone_constraints": _MC, "mc_method": "advanced",
                                              "monotone_penalty": 1.0}),
    make_case("cegb_categorical", "binary", {"cegb_penalty_feature_lazy": [0.01, 0.05, 0.002, 0.1, 0.03, 0.005],
                                             "cegb_penalty_feature_coupled": [5, 20, 1, 40, 3, 0.5],
                                             "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("cegb_multiclass", "multiclass", {"num_class": 3, "cegb_penalty_split": 0.002,
                                                "cegb_penalty_feature_lazy": [0.01, 0.05, 0.002, 0.1, 0.03, 0.005],
                                                "extra_trees": True}),
    make_case("cegb_rf", "regression", {**_RF_BAG, "cegb_penalty_feature_lazy": _CEGB_LAZY,
                                        "cegb_penalty_feature_coupled": _CEGB_COUPLED}),
    make_case("cegb_dart", "regression", {"boosting": "dart", "cegb_penalty_feature_lazy": _CEGB_LAZY,
                                          "bagging_fraction": 0.3, "bagging_freq": 2}),
    # interaction constraints (upstream ColSampler::GetByNode with Tree::branch_features)
    make_case("ic_disjoint", "regression", {"interaction_constraints": _IC}),
    make_case("ic_overlap", "regression", {"interaction_constraints": _IC_OVERLAP}),
    make_case("ic_partial", "regression", {"interaction_constraints": [[0, 1]]}),
    make_case("ic_bynode", "regression", {"interaction_constraints": _IC_OVERLAP, "feature_fraction_bynode": 0.6}),
    make_case("ic_bytree", "binary", {"interaction_constraints": _IC_OVERLAP, "feature_fraction": 0.7},
              weighted=True),
    make_case("ic_both", "regression", {"interaction_constraints": _IC_OVERLAP, "feature_fraction": 0.7,
                                        "feature_fraction_bynode": 0.6, "bagging_fraction": 0.8, "bagging_freq": 1}),
    make_case("ic_monotone", "regression", {"interaction_constraints": _IC_OVERLAP, "monotone_constraints": _MC,
                                            "mc_method": "advanced"}),
    make_case("ic_categorical", "binary", {"interaction_constraints": [[0, 1, 3], [2, 4, 5], [1]],
                                           "categorical_feature": "1,3,4"}, kind="categorical"),
    make_case("ic_multiclass", "multiclass", {"num_class": 3, "interaction_constraints": _IC_OVERLAP}),
    make_case("ic_100_rounds", "regression", {"interaction_constraints": _IC_OVERLAP, "num_leaves": 63,
                                              "feature_fraction": 0.8, "feature_fraction_bynode": 0.7},
              n=8000, rounds=100),
    # DART (upstream dart.hpp): weighted and uniform drops, skip_drop, max_drop limits, xgboost mode
    make_case("dart_basic", "regression", {"boosting": "dart"}),
    make_case("dart_no_skip", "regression", {"boosting": "dart", "skip_drop": 0.0, "drop_rate": 0.3}),
    make_case("dart_xgboost", "regression", {"boosting": "dart", "xgboost_dart_mode": True, "skip_drop": 0.0}),
    make_case("dart_uniform", "regression", {"boosting": "dart", "uniform_drop": True, "skip_drop": 0.2,
                                             "max_drop": 3}),
    make_case("dart_max_drop_zero", "regression", {"boosting": "dart", "max_drop": 0, "skip_drop": 0.0,
                                                   "drop_rate": 0.5}),
    make_case("dart_no_drop_limit", "regression", {"boosting": "dart", "max_drop": -1, "skip_drop": 0.0,
                                                   "drop_rate": 0.5, "uniform_drop": True}),
    make_case("dart_binary", "binary", {"boosting": "dart", "skip_drop": 0.1, "drop_seed": 11}, weighted=True),
    make_case("dart_multiclass", "multiclass", {"boosting": "dart", "num_class": 3, "skip_drop": 0.1}),
    make_case("dart_bagging", "regression", {"boosting": "dart", "bagging_fraction": 0.7, "bagging_freq": 1,
                                             "skip_drop": 0.0, "seed": 7}),
    make_case("dart_l1", "regression_l1", {"boosting": "dart", "skip_drop": 0.0}),
    make_case("dart_100_rounds", "regression", {"boosting": "dart", "num_leaves": 63, "skip_drop": 0.2},
              n=8000, rounds=100),
    make_case("rf_basic", "regression", {**_RF_BAG}),
    make_case("rf_feature_fraction_only", "regression", {"boosting": "rf", "feature_fraction": 0.6}),
    make_case("rf_goss", "regression", {"boosting": "rf", "data_sample_strategy": "goss", "feature_fraction": 0.8}),
    make_case("rf_small_bag", "regression", {**_RF_BAG, "bagging_fraction": 0.3}),
    make_case("rf_no_boost_from_average", "regression", {**_RF_BAG, "boost_from_average": False}),
    make_case("rf_binary", "binary", {**_RF_BAG}, weighted=True),
    make_case("rf_multiclass", "multiclass", {**_RF_BAG, "num_class": 3}),
    make_case("rf_l1", "regression_l1", {**_RF_BAG}),
    make_case("rf_quantile", "quantile", {**_RF_BAG, "alpha": 0.3}),
    make_case("rf_mape", "mape", {**_RF_BAG}),
    make_case("rf_poisson", "poisson", {**_RF_BAG}),
    make_case("rf_categorical", "regression", {**_RF_BAG, "feature_fraction": 0.8}, kind="categorical"),
]


def case_ids() -> List[str]:
    return [c.name for c in CASES]


# --------------------------------------------------------------------------- engines


def train_both(case: Case, params: Optional[Dict[str, Any]] = None, rounds: Optional[int] = None,
               valid: bool = True, callbacks_factory: Optional[Callable[[Any], List[Any]]] = None):
    params = dict(case.full_params if params is None else params)
    rounds = case.num_boost_round if rounds is None else rounds
    out = []
    for mod in (lgb_rs, lgb_up):
        train = mod.Dataset(case.X, label=case.y, weight=case.weight, init_score=case.init_score,
                            free_raw_data=False)
        valid_sets = [train.create_valid(case.Xv, label=case.yv)] if valid else None
        cbs = callbacks_factory(mod) if callbacks_factory else None
        out.append(mod.train(params, train, num_boost_round=rounds, valid_sets=valid_sets, callbacks=cbs))
    return out[0], out[1]


_MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def avoid_inf(x: float) -> float:
    """upstream: Common::AvoidInf."""
    if math.isnan(x):
        return 0.0
    if x >= 1e300:
        return 1e300
    if x <= -1e300:
        return -1e300
    return x


def upstream_tree_arrays(booster: Any) -> List[Dict[str, Any]]:
    """Flatten upstream ``dump_model()`` trees into the layout of ``RsBooster.tree_arrays``.

    ``dump_model`` prints doubles with 17 significant digits, so values
    round-trip exactly.
    """
    trees = []
    for info in booster.dump_model()["tree_info"]:
        nl = info["num_leaves"]
        ni = nl - 1
        t: Dict[str, Any] = {"num_leaves": nl, "shrinkage": info["shrinkage"]}
        for k in ("split_feature", "split_gain", "threshold", "decision_type", "left_child", "right_child",
                  "internal_value", "internal_weight", "internal_count"):
            t[k] = [None] * ni
        for k in ("leaf_value", "leaf_weight", "leaf_count"):
            t[k] = [None] * nl

        def idx(node: Dict[str, Any]) -> int:
            return node["split_index"] if "split_index" in node else ~node.get("leaf_index", 0)

        def visit(node: Dict[str, Any]) -> None:
            if "split_index" in node:
                s = node["split_index"]
                t["split_feature"][s] = node["split_feature"]
                t["split_gain"][s] = node["split_gain"]
                t["threshold"][s] = node["threshold"]
                t["decision_type"][s] = ((1 if node["decision_type"] == "==" else 0) | (2 if node["default_left"] else 0)
                                         | (_MISSING[node["missing_type"]] << 2))
                t["left_child"][s] = idx(node["left_child"])
                t["right_child"][s] = idx(node["right_child"])
                t["internal_value"][s] = node["internal_value"]
                t["internal_weight"][s] = node["internal_weight"]
                t["internal_count"][s] = node["internal_count"]
                visit(node["left_child"])
                visit(node["right_child"])
            else:
                leaf = node.get("leaf_index", 0)
                t["leaf_value"][leaf] = node["leaf_value"]
                t["leaf_weight"][leaf] = node.get("leaf_weight", 0.0)
                t["leaf_count"][leaf] = node.get("leaf_count", 0)

        visit(info["tree_structure"])
        trees.append(t)
    return trees


def rust_tree_arrays(booster: Any) -> List[Dict[str, Any]]:
    trees = booster._tree_arrays()
    for t in trees:
        if not t["leaf_weight"]:
            # a reloaded one-leaf tree stores no weight; upstream_tree_arrays defaults the missing JSON key to 0.0
            t["leaf_weight"] = [0.0] * t["num_leaves"]
        # categorical nodes: dump_model writes the category list as "a||b"
        t["threshold"] = [avoid_inf(v) if c is None else "||".join(map(str, c))
                          for v, c in zip(t["threshold"], t.pop("cat_threshold"))]
        t["split_gain"] = [avoid_inf(v) for v in t["split_gain"]]
    return trees


def parse_dump_text(path: Path, num_data: int) -> List[Optional[np.ndarray]]:
    """Per-column bin indices from ``Dataset._dump_text`` (``None`` for unused features)."""
    lines = path.read_text().splitlines()
    rows = [ln.split(", ")[:-1] for ln in lines[-num_data:]]
    ncol = len(rows[0])
    cols: List[Optional[np.ndarray]] = []
    for j in range(ncol):
        vals = [r[j] for r in rows]
        cols.append(None if vals[0] == "NA" else np.array([int(v) for v in vals], dtype=np.int64))
    return cols
