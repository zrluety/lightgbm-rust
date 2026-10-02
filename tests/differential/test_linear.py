"""Differential tests for linear trees (``linear_tree``): raw feature storage for every input path, the
per-leaf least-squares fits, linear prediction, model text / JSON fields, refit and the parameter checks,
at 1 and 4 threads, against LightGBM 4.7.0.

upstream: src/treelearner/linear_tree_learner.cpp/.h, src/io/tree.cpp (linear fields, AddPredictionToScore),
src/io/dataset.cpp and dataset_loader.cpp (raw data), src/io/config.cpp (CheckParamConflict)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import lightgbm as lgb_up
import numpy as np
import pytest
import scipy.sparse as sp

import lightgbm_rust as lgb_rs

BASE = {"verbosity": -1, "deterministic": True, "seed": 3, "num_leaves": 15, "min_data_in_leaf": 10,
        "linear_tree": True}
ROUNDS = 10


def _data(n: int, f: int, seed: int, nan: bool = True):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    X[:, 4] = rng.integers(0, 8, size=n)
    X[rng.random((n, f)) < 0.15] = 0.0
    if nan:
        X[rng.random(n) < 0.1, 1] = np.nan
    y = (2 * X[:, 0] + np.sin(np.nan_to_num(X[:, 1])) + 0.3 * X[:, 2] * X[:, 3] + 0.2 * X[:, 5:].sum(axis=1)
         + rng.normal(scale=0.2, size=n))
    return X, y


def _wide(n: int, f: int, seed: int):
    """Every feature carries signal, so deep trees put many distinct features on a leaf's path."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, f))
    X[rng.random(n) < 0.02, 3] = np.nan
    return X, np.sin(np.nan_to_num(X)).sum(axis=1) + rng.normal(scale=0.1, size=n)


DATASETS = {"dense": _data(3000, 8, 0), "no_nan": _data(3000, 8, 1, nan=False), "wide": _wide(6000, 24, 2)}
OBJECTIVES: Dict[str, Dict[str, Any]] = {
    "regression": {"objective": "regression"},
    "binary": {"objective": "binary"},
    "multiclass": {"objective": "multiclass", "num_class": 3},
    "poisson": {"objective": "poisson"},
    "huber": {"objective": "huber"},
    "quantile": {"objective": "quantile", "alpha": 0.7},
}
VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "lambda": {"linear_lambda": 2.5},
    "bagging": {"bagging_fraction": 0.7, "bagging_freq": 1},
    "bagging_subset": {"bagging_fraction": 0.3, "bagging_freq": 1},
    "goss": {"data_sample_strategy": "goss"},
    "dart": {"boosting": "dart", "drop_rate": 0.3},
    "rf": {"boosting": "rf", "bagging_fraction": 0.6, "bagging_freq": 1},
    "categorical": {"categorical_feature": [4], "max_cat_to_onehot": 4},
    "regularized": {"lambda_l2": 1.0, "path_smooth": 0.5, "max_delta_step": 0.7},
    "col_wise": {"force_col_wise": True},
    "learning_rate": {"learning_rate": 0.3, "boost_from_average": False},
    # the tree learner is forced to serial with a warning
    "data_parallel": {"tree_learner": "data"},
}


def _target(objective: str, y: np.ndarray) -> np.ndarray:
    if objective == "binary":
        return (y > np.median(y)).astype(float)
    if objective == "multiclass":
        return np.digitize(y, np.quantile(y, [1 / 3, 2 / 3])).astype(float)
    if objective == "poisson":
        return np.floor(np.exp(np.clip(y, -3, 3) * 0.5))
    return y


def _split(X, y):
    k = len(y) * 2 // 3
    return X[:k], y[:k], X[k:], y[k:]


def _train(mod, X, y, Xv, yv, params, rounds=ROUNDS, callbacks=(), **kw):
    p = dict(params)
    cat = p.pop("categorical_feature", "auto")
    ds = mod.Dataset(X, label=y, params=p, categorical_feature=cat, free_raw_data=False)
    hist: Dict[str, Any] = {}
    b = mod.train(p, ds, num_boost_round=rounds, valid_sets=[ds.create_valid(Xv, label=yv)],
                  callbacks=[mod.record_evaluation(hist), *callbacks], **kw)
    return b, hist


def _compare(rec, threads, rs, hrs, up, hup, Xv):
    # the leaf fits are exact at any thread count (per-thread partial sums combined in thread order);
    # only metrics that upstream reduces over OpenMP threads carry the multithread tolerance
    suffix = "" if threads == 1 else f" ({threads} threads)"
    rec.compare("model_text" + suffix, "model_text", rs.model_to_string(), up.model_to_string())
    for m, v in hup["valid_0"].items():
        rec.compare(f"valid_0.{m}[per iteration]{suffix}", "metrics" if threads == 1 else "multithread",
                    hrs["valid_0"].get(m), v)
    rec.compare("raw_score[holdout]" + suffix, "predictions", rs.predict(Xv, raw_score=True),
                up.predict(Xv, raw_score=True))


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("objective", list(OBJECTIVES))
def test_linear_training(objective, variant, threads, recorder):
    """Trees, leaf models, validation metrics and predictions of linear trees on data with NaN."""
    rec = recorder(f"{objective}[{variant},{threads} threads]")
    X, y = DATASETS["dense"]
    X, y, Xv, yv = _split(X, _target(objective, y))
    params = {**BASE, **OBJECTIVES[objective], **VARIANTS[variant], "num_threads": threads}
    (rs, hrs), (up, hup) = (_train(mod, X, y, Xv, yv, params) for mod in (lgb_rs, lgb_up))
    _compare(rec, threads, rs, hrs, up, hup, Xv)
    rec.finish()


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("data", ["no_nan", "wide"])
def test_linear_systems(data, threads, recorder):
    """Without NaN (upstream's HAS_NAN=false paths) and with leaf systems of up to 15 unknowns, where
    Eigen's inverse-times-vector product switches from the coefficient-based to the GEMV kernel and the
    triangular solves span several panels."""
    rec = recorder(f"{data},{threads} threads")
    X, y, Xv, yv = _split(*DATASETS[data])
    shape = {"num_leaves": 511, "min_data_in_leaf": 2, "learning_rate": 0.5} if data == "wide" else \
        {"num_leaves": 63, "min_data_in_leaf": 5}
    params = {**BASE, **shape, "num_threads": threads}
    (rs, hrs), (up, hup) = (_train(mod, X, y, Xv, yv, params) for mod in (lgb_rs, lgb_up))
    _compare(rec, threads, rs, hrs, up, hup, Xv)
    rec.finish()


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("fmt", ["csr", "csc", "csr_valid", "arrow"])
def test_linear_sparse_inputs(fmt, threads, recorder):
    """Raw values from CSR (stored entries, zeros elsewhere), CSC and Arrow (every value) training data."""
    rec = recorder(f"{fmt},{threads} threads")
    X, y, Xv, yv = _split(*DATASETS["dense"])
    if fmt == "arrow":
        import pyarrow as pa

        Xt = pa.table({f"c{i}": X[:, i] for i in range(X.shape[1])})
        Xvv = Xv
    else:
        conv = sp.csc_matrix if fmt == "csc" else sp.csr_matrix
        Xt = conv(np.nan_to_num(X, nan=0.0)) if fmt != "csr_valid" else X
        Xvv = sp.csr_matrix(np.nan_to_num(Xv, nan=0.0)) if fmt == "csr_valid" else Xv
    params = {**BASE, "num_threads": threads}
    (rs, hrs), (up, hup) = (_train(mod, Xt, y, Xvv, yv, params) for mod in (lgb_rs, lgb_up))
    _compare(rec, threads, rs, hrs, up, hup, Xv)
    rec.finish()


def _write_text(path: Path, X: np.ndarray, y: np.ndarray, libsvm: bool) -> str:
    with open(path, "w") as f:
        for lab, row in zip(y, X):
            if libsvm:
                f.write(" ".join([repr(float(lab))] + [f"{j}:{float(v)!r}" for j, v in enumerate(row) if v != 0]) + "\n")
            else:
                f.write(",".join([repr(float(lab))] + [repr(float(v)) for v in row]) + "\n")
    return str(path)


@pytest.mark.parametrize("threads", [1, 3, 4])
@pytest.mark.parametrize("fmt", ["csv", "libsvm"])
def test_linear_text_files(fmt, threads, recorder, tmp_path):
    """Raw values of text files: upstream keeps each loader thread's previous row for the zeros a line
    leaves out, so the stored values depend on the thread count."""
    rec = recorder(f"{fmt},{threads} threads")
    X, y = DATASETS["dense"]
    X, y, Xv, yv = _split(X[:1500], y[:1500])
    train = _write_text(tmp_path / f"train.{fmt}", X, y, fmt == "libsvm")
    valid = _write_text(tmp_path / f"valid.{fmt}", Xv, yv, fmt == "libsvm")
    params = {**BASE, "num_threads": threads}
    models = []
    for mod in (lgb_rs, lgb_up):
        ds = mod.Dataset(train, params=params)
        hist: Dict[str, Any] = {}
        b = mod.train(params, ds, num_boost_round=ROUNDS, valid_sets=[ds.create_valid(valid)],
                      callbacks=[mod.record_evaluation(hist)])
        models.append((b, hist))
    (rs, hrs), (up, hup) = models
    _compare(rec, threads, rs, hrs, up, hup, Xv)
    rec.finish()


def test_linear_binary_dataset(recorder, tmp_path):
    """Raw values survive save_binary (each library reads its own binary format)."""
    rec = recorder("save_binary")
    X, y, Xv, yv = _split(*DATASETS["dense"])
    params = {**BASE, "num_threads": 1}
    out = []
    for mod in (lgb_rs, lgb_up):
        path = tmp_path / f"{mod.__name__}.bin"
        mod.Dataset(X, label=y, params=params).save_binary(str(path))
        out.append(mod.train(params, mod.Dataset(str(path), params=params), num_boost_round=ROUNDS))
    rec.compare("model_text[binary dataset]", "model_text", out[0].model_to_string(), out[1].model_to_string())
    rec.compare("raw_score[holdout]", "predictions", out[0].predict(Xv, raw_score=True),
                out[1].predict(Xv, raw_score=True))
    rec.finish()


def test_linear_model_io(recorder):
    """Model text round trip, dump_model fields, leaf prediction, continued training and refit."""
    rec = recorder("model io")
    X, y, Xv, yv = _split(*DATASETS["dense"])
    params = {**BASE, "num_threads": 1}
    (rs, _), (up, _) = (_train(mod, X, y, Xv, yv, params) for mod in (lgb_rs, lgb_up))
    text = up.model_to_string()
    loaded = [mod.Booster(model_str=text) for mod in (lgb_rs, lgb_up)]
    rec.compare("model_text[loaded upstream model]", "model_text", loaded[0].model_to_string(),
                loaded[1].model_to_string())
    rec.compare("predict[loaded]", "predictions", loaded[0].predict(Xv), loaded[1].predict(Xv))
    rec.compare("dump_model", "model_text", json.dumps(rs.dump_model()), json.dumps(up.dump_model()))
    rec.compare("dump_model[loaded]", "model_text", json.dumps(loaded[0].dump_model()),
                json.dumps(loaded[1].dump_model()))
    rec.compare("pred_leaf", "predictions", rs.predict(Xv, pred_leaf=True), up.predict(Xv, pred_leaf=True))
    cont = [_train(mod, X, y, Xv, yv, params, rounds=5, init_model=b)[0] for mod, b in ((lgb_rs, rs), (lgb_up, up))]
    rec.compare("model_text[init_model]", "model_text", cont[0].model_to_string(), cont[1].model_to_string())
    for decay in (0.9, 0.3):
        ref = [b.refit(Xv, yv, decay_rate=decay) for b in (rs, up)]
        rec.compare(f"model_text[refit decay={decay}]", "model_text", ref[0].model_to_string(),
                    ref[1].model_to_string())
    ref = [mod.Booster(model_str=text).refit(Xv, yv) for mod in (lgb_rs, lgb_up)]
    rec.compare("model_text[refit loaded]", "model_text", ref[0].model_to_string(), ref[1].model_to_string())
    rec.finish()


def test_linear_reset_parameter(recorder):
    """linear_lambda changed between iterations."""
    rec = recorder("reset linear_lambda")
    X, y, Xv, yv = _split(*DATASETS["dense"])
    params = {**BASE, "num_threads": 1}
    cb = lambda mod: mod.reset_parameter(linear_lambda=[0.0, 1.0, 5.0, 0.5, 2.0] * 2)  # noqa: E731
    (rs, hrs), (up, hup) = (_train(mod, X, y, Xv, yv, params, callbacks=[cb(mod)]) for mod in (lgb_rs, lgb_up))
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    rec.finish()


@pytest.mark.parametrize("case", ["regression_l1", "zero_as_missing", "contrib", "dataset_param"])
def test_linear_errors_match_upstream(case):
    """Upstream's fatal checks for linear trees."""
    X, y, Xv, _ = _split(*DATASETS["dense"])
    params = {**BASE, "num_threads": 1}
    for mod in (lgb_rs, lgb_up):
        if case == "regression_l1":
            with pytest.raises(mod.basic.LightGBMError, match="Cannot use regression_l1 objective"):
                mod.train({**params, "objective": "regression_l1"}, mod.Dataset(X, label=y), num_boost_round=1)
        elif case == "zero_as_missing":
            with pytest.raises(mod.basic.LightGBMError, match="zero_as_missing must be false"):
                mod.train({**params, "zero_as_missing": True}, mod.Dataset(X, label=y), num_boost_round=1)
        elif case == "contrib":
            b = mod.train(params, mod.Dataset(X, label=y), num_boost_round=2)
            with pytest.raises(mod.basic.LightGBMError, match="SHAP feature contributions is not implemented"):
                b.predict(Xv, pred_contrib=True)
        else:
            ds = mod.Dataset(X, label=y, params={k: v for k, v in params.items() if k != "linear_tree"}).construct()
            with pytest.raises(mod.basic.LightGBMError, match="Cannot change linear_tree"):
                mod.train(params, ds, num_boost_round=1)


def test_linear_gates(tmp_path):
    """Combinations whose upstream results depend on memory it does not write are rejected."""
    X, y, Xv, yv = _split(*DATASETS["dense"])
    p = {**BASE, "num_threads": 1}
    with pytest.raises(lgb_rs.basic.LightGBMError, match="use_quantized_grad"):
        lgb_rs.train({**p, "use_quantized_grad": True}, lgb_rs.Dataset(X, label=y), num_boost_round=1)
    ds = lgb_rs.Dataset(X, label=y, params=p)
    with pytest.raises(lgb_rs.basic.LightGBMError, match="CSC validation data"):
        lgb_rs.train(p, ds, num_boost_round=1,
                     valid_sets=[ds.create_valid(sp.csc_matrix(np.nan_to_num(Xv)), label=yv)])
    path = _write_text(tmp_path / "t.csv", X, y, False)
    with pytest.raises(lgb_rs.basic.LightGBMError, match="two_round"):
        lgb_rs.Dataset(path, params={**p, "two_round": True}).construct()
    b = lgb_rs.train(p, lgb_rs.Dataset(X[:500], label=y[:500], params=p), num_boost_round=1,
                     keep_training_booster=True)
    with pytest.raises(lgb_rs.basic.LightGBMError, match="more rows than the first training set"):
        b.update(train_set=lgb_rs.Dataset(X, label=y, params=p, reference=b.train_set))
