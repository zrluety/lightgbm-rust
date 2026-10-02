"""Differential tests for quantized training (``use_quantized_grad``): gradient discretization with
stochastic rounding, integer histograms and split search, and leaf renewing, at 1 and 4 threads,
row-wise and col-wise, against LightGBM 4.7.0.

upstream: src/treelearner/gradient_discretizer.cpp, feature_histogram.hpp (the *Int functions),
leaf_splits.hpp, serial_tree_learner.cpp, src/io/dataset.cpp (ConstructHistogramsInt)
"""

from __future__ import annotations

from typing import Any, Dict

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

BASE = {"verbosity": -1, "deterministic": True, "seed": 7, "num_leaves": 15, "min_data_in_leaf": 5,
        "use_quantized_grad": True}
LAYOUTS = {"row_wise": {"force_row_wise": True}, "col_wise": {"force_col_wise": True}}
ROUNDS = 12


def _data(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    X[:, 6] = rng.integers(0, 12, size=n)
    X[rng.random(n) < 0.1, 2] = np.nan
    y = np.tanh(X[:, 0]) + 0.5 * X[:, 1] + 0.3 * (X[:, 6] % 3) + rng.normal(scale=0.3, size=n)
    return X, y


def _sparse(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = np.where(rng.random((n, 120)) < 0.03, rng.normal(size=(n, 120)), 0.0)
    X[:, :3] = rng.normal(size=(n, 3))
    y = X[:, 0] + 0.5 * X[:, 1] + X[:, 3:].sum(axis=1) * 0.4 + rng.normal(scale=0.3, size=n)
    return X, y


DATASETS = {"dense": _data(4000, 0), "sparse": _sparse(4000, 1)}
OBJECTIVES: Dict[str, Dict[str, Any]] = {
    # constant hessian: one hessian bin per row
    "regression": {"objective": "regression"},
    "binary": {"objective": "binary"},
    "huber": {"objective": "huber"},
    "poisson": {"objective": "poisson"},
}
VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "no_stochastic": {"stochastic_rounding": False},
    "bins_16": {"num_grad_quant_bins": 16},
    "bins_2": {"num_grad_quant_bins": 2},
    "renew": {"quant_train_renew_leaf": True},
    # bagging without a subset (upstream's float-scale root sums)
    "bagging": {"bagging_fraction": 0.7, "bagging_freq": 1},
    "bagging_subset": {"bagging_fraction": 0.3, "bagging_freq": 1},
    "goss": {"data_sample_strategy": "goss"},
    "regularized": {"lambda_l1": 0.5, "lambda_l2": 2.0, "max_delta_step": 0.8, "path_smooth": 1.0,
                    "min_sum_hessian_in_leaf": 2.0},
    "extra_trees": {"extra_trees": True},
    "categorical": {"categorical_feature": [6], "max_cat_to_onehot": 4, "cat_smooth": 5.0},
    "onehot_categorical": {"categorical_feature": [6], "max_cat_to_onehot": 16},
    "monotone": {"monotone_constraints": [1, 0, 0, 0, 0, 0, 0, 0]},
}


def _target(objective: str, y: np.ndarray) -> np.ndarray:
    if objective == "binary":
        return (y > np.median(y)).astype(float)
    if objective == "poisson":
        return np.floor(np.exp(np.clip(y, -3, 3) * 0.5))
    return y


def _split(X, y):
    k = len(y) * 3 // 4
    return X[:k], y[:k], X[k:], y[k:]


def _trees(b) -> str:
    return b.model_to_string().split("end of trees")[0]


def _train(mod, X, y, Xv, yv, params, weight=None):
    p = dict(params)
    cat = p.pop("categorical_feature", "auto")
    ds = mod.Dataset(X, label=y, weight=weight, params=p, categorical_feature=cat, free_raw_data=False)
    hist: Dict[str, Any] = {}
    b = mod.train(p, ds, num_boost_round=ROUNDS, valid_sets=[ds.create_valid(Xv, label=yv)],
                  callbacks=[mod.record_evaluation(hist)])
    return b, hist


def _compare(rec, threads, rs, hrs, up, hup, Xv):
    if threads == 1:
        rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
        for m, v in hup["valid_0"].items():
            rec.compare(f"valid_0.{m}[per iteration]", "metrics", hrs["valid_0"].get(m), v)
        rec.compare("raw_score[holdout]", "predictions", rs.predict(Xv, raw_score=True), up.predict(Xv, raw_score=True))
    else:
        # integer histograms do not depend on the thread count; upstream's root sums and renewed leaf
        # outputs are OpenMP reductions (summed here in row order), so values carry the multithread tolerance
        rec.compare("trees (4 threads)", "tree_structure", _structure(rs), _structure(up))
        for m, v in hup["valid_0"].items():
            rec.compare(f"valid_0.{m}[per iteration] (4 threads)", "multithread", hrs["valid_0"].get(m), v)
        rec.compare("raw_score[holdout] (4 threads)", "multithread", rs.predict(Xv, raw_score=True),
                    up.predict(Xv, raw_score=True))


def _structure(b):
    """Split features, thresholds and leaf counts of every tree."""
    out = []
    for t in b.dump_model()["tree_info"]:
        stack = [t["tree_structure"]]
        while stack:
            n = stack.pop()
            if "split_feature" in n:
                out.append((n["split_feature"], n.get("threshold"), n["internal_count"]))
                stack += [n["left_child"], n["right_child"]]
            else:
                out.append(("leaf", n.get("leaf_count")))
    return out


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_quantized_training(objective, variant, layout, threads, recorder):
    """Trees, validation metrics and predictions with quantized gradients on dense data."""
    rec = recorder(f"{objective}[{variant},{layout},{threads} threads]")
    X, y = DATASETS["dense"]
    X, y, Xv, yv = _split(X, _target(objective, y))
    params = {**BASE, **OBJECTIVES[objective], **LAYOUTS[layout], **VARIANTS[variant], "num_threads": threads}
    try:
        up, hup = _train(lgb_up, X, y, Xv, yv, params)
    except lgb_up.basic.LightGBMError as e:
        # poisson with bagging: upstream's bagged root sums read the packed integers by bag position, so a
        # split can send every row left and upstream's own CHECK fails; the port must fail the same way
        with pytest.raises(lgb_rs.basic.LightGBMError) as rs_err:
            _train(lgb_rs, X, y, Xv, yv, params)
        rec.compare("error", "model_text", str(rs_err.value).split(": ", 1)[-1], str(e).split(" at /")[0])
        rec.finish()
        return
    rs, hrs = _train(lgb_rs, X, y, Xv, yv, params)
    _compare(rec, threads, rs, hrs, up, hup, Xv)
    rec.finish()


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("variant", ["default", "renew", "bagging_subset"])
def test_quantized_training_sparse_groups(variant, layout, threads, recorder):
    """Quantized histograms of bundled, sparse and multi-value feature groups."""
    rec = recorder(f"{variant},{layout},{threads} threads")
    X, y, Xv, yv = _split(*DATASETS["sparse"])
    params = {**BASE, "objective": "regression", **LAYOUTS[layout], **VARIANTS[variant], "num_threads": threads}
    (rs, hrs), (up, hup) = (_train(mod, X, y, Xv, yv, params) for mod in (lgb_rs, lgb_up))
    _compare(rec, threads, rs, hrs, up, hup, Xv)
    rec.finish()


@pytest.mark.parametrize("renew", [False, True])
def test_quantized_training_weighted_and_ranking(renew, recorder):
    """Weighted regression (non-constant hessians) and lambdarank, where upstream recommends renewing."""
    rec = recorder(f"renew={renew}")
    X, y, Xv, yv = _split(*DATASETS["dense"])
    w = np.random.default_rng(5).uniform(0.5, 2.0, size=len(y))
    params = {**BASE, "objective": "regression", "num_threads": 1, "quant_train_renew_leaf": renew}
    (rs, hrs), (up, hup) = (_train(mod, X, y, Xv, yv, params, weight=w) for mod in (lgb_rs, lgb_up))
    rec.compare("model_text[weighted]", "model_text", rs.model_to_string(), up.model_to_string())
    rel = np.clip(np.floor(y - y.min()), 0, 4)
    group = [50] * (len(y) // 50)
    p = {**BASE, "objective": "lambdarank", "num_threads": 1, "quant_train_renew_leaf": renew}
    models = [mod.train(p, mod.Dataset(X, label=rel, group=group), num_boost_round=ROUNDS) for mod in (lgb_rs, lgb_up)]
    rec.compare("model_text[lambdarank]", "model_text", models[0].model_to_string(), models[1].model_to_string())
    rec.finish()


def test_quantized_gates():
    """Combinations whose upstream results read memory it did not write are rejected."""
    X, y, _, _ = _split(*DATASETS["sparse"])
    ds = lambda: lgb_rs.Dataset(X, label=y, free_raw_data=False)  # noqa: E731
    p = {**BASE, "objective": "regression", "num_threads": 1}
    with pytest.raises(lgb_rs.basic.LightGBMError, match="feature_fraction"):
        lgb_rs.train({**p, "force_row_wise": True, "feature_fraction": 0.5}, ds(), num_boost_round=1)
    with pytest.raises(lgb_rs.basic.LightGBMError, match="monotone_constraints_method"):
        lgb_rs.train({**p, "monotone_constraints": [1] + [0] * 119, "monotone_constraints_method": "advanced"},
                     ds(), num_boost_round=1)
    b = lgb_rs.train({**p, "use_quantized_grad": False}, ds(), num_boost_round=1, keep_training_booster=True)
    with pytest.raises(lgb_rs.basic.LightGBMError, match="turning on use_quantized_grad"):
        b.reset_parameter({"use_quantized_grad": True})
