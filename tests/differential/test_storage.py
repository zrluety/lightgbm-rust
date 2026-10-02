"""Differential tests for feature-group storage: exclusive feature bundles, multi-value groups, sparse
and 4-bit bins, sparse input without densifying, streamed text files and binary files, at 1 and 4
threads, row-wise and col-wise, against LightGBM 4.7.0.

upstream: src/io/dataset.cpp (Construct, PushDataToMultiValBin, ConstructHistograms), feature_group.h,
src/io/dense_bin.hpp, sparse_bin.hpp, multi_val_dense_bin.hpp, multi_val_sparse_bin.hpp,
src/io/train_share_states.cpp, src/io/dataset_loader.cpp (ExtractFeaturesFromMemory / FromFile)
"""

from __future__ import annotations

import json
from typing import Any, Dict

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

BASE = {"objective": "regression", "verbosity": -1, "deterministic": True, "seed": 3, "num_leaves": 15,
        "min_data_in_leaf": 5}
LAYOUTS = {"row_wise": {"force_row_wise": True}, "col_wise": {"force_col_wise": True}}
ROUNDS = 15


def _data(n: int, p: int, density: float, seed: int, onehot: int = 0):
    """`p` columns, nonzero with probability `density`, except 3 dense columns and an optional
    block of `onehot` mutually exclusive columns (bundled into one group)."""
    rng = np.random.default_rng(seed)
    X = np.where(rng.random((n, p)) < density, rng.normal(size=(n, p)), 0.0)
    X[:, :3] = rng.normal(size=(n, 3))
    if onehot:
        slot = rng.integers(0, onehot, size=n)
        X[:, 3:3 + onehot] = (slot[:, None] == np.arange(onehot)[None, :]) * rng.normal(size=(n, 1))
    y = np.tanh(X[:, 0]) + 0.5 * X[:, 1] + X[:, 3:].sum(axis=1) * 0.3 + rng.normal(scale=0.3, size=n)
    return X, y


# one-third holdout rows for validation sets (single-feature groups, sparse bins for sparse features)
DATASETS = {
    # 98% zeros: one multi-value sparse group (row-wise sparse multi-value bin)
    "wide_sparse": _data(6000, 200, 0.02, 0),
    # a 40-column exclusive block (bundled) next to sparse columns
    "onehot": _data(5000, 60, 0.05, 1, onehot=40),
    # 70% zeros: bundles of conflicting columns, row-wise sparse rate below the dense threshold of some layouts
    "medium": _data(5000, 40, 0.3, 2),
}
VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {},
    # by-tree features under 60% of the dense rate: upstream's sub-column multi-value bin
    "feature_fraction": {"feature_fraction": 0.3},
    # bagging subset mode (upstream's sub-row multi-value bin)
    "bagging_subset": {"bagging_fraction": 0.3, "bagging_freq": 1},
    # at most 16 bins per group: 4-bit dense bins
    "max_bin_15": {"max_bin": 15},
    "no_bundle": {"enable_bundle": False},
    "no_sparse": {"is_enable_sparse": False},
}


def _split(X, y):
    k = len(y) * 3 // 4
    return X[:k], y[:k], X[k:], y[k:]


def _trees(b) -> str:
    return b.model_to_string().split("end of trees")[0]


def _train(mod, X, y, Xv, yv, params, valid=True):
    ds = mod.Dataset(X, label=y, params=params, free_raw_data=False)
    hist: Dict[str, Any] = {}
    kw = {}
    if valid:
        kw = {"valid_sets": [ds.create_valid(Xv, label=yv)], "callbacks": [mod.record_evaluation(hist)]}
    return mod.train(params, ds, num_boost_round=ROUNDS, **kw), hist


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_grouped_storage(dataset, variant, layout, threads, recorder):
    """Trees, validation metrics and predictions on bundled / multi-value / sparse feature groups."""
    rec = recorder(f"{dataset}[{variant},{layout},{threads} threads]")
    X, y, Xv, yv = _split(*DATASETS[dataset])
    params = {**BASE, **LAYOUTS[layout], **VARIANTS[variant], "num_threads": threads}
    (rs, hrs), (up, hup) = _train(lgb_rs, X, y, Xv, yv, params), _train(lgb_up, X, y, Xv, yv, params)
    if threads == 1:
        rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
        for m, v in hup["valid_0"].items():
            rec.compare(f"valid_0.{m}[per iteration]", "metrics", hrs["valid_0"].get(m), v)
        rec.compare("raw_score[holdout]", "predictions", rs.predict(Xv, raw_score=True), up.predict(Xv, raw_score=True))
    else:
        # upstream's row blocks, merge order and sub-column moves are mirrored, so the trees are expected
        # to be identical; values are reported under the multithread tolerance
        rec.compare("trees (4 threads)", "model_text", _trees(rs), _trees(up))
        for m, v in hup["valid_0"].items():
            rec.compare(f"valid_0.{m}[per iteration] (4 threads)", "multithread", hrs["valid_0"].get(m), v)
        rec.compare("raw_score[holdout] (4 threads)", "multithread", rs.predict(Xv, raw_score=True),
                    up.predict(Xv, raw_score=True))
    rec.finish()


def _to_input(fmt, X, y, tmp_path, name):
    import scipy.sparse as sp

    if fmt == "csr":
        return sp.csr_matrix(X)
    if fmt == "csc":
        return sp.csc_matrix(X)
    if fmt in ("csv", "csv_two_round"):
        path = tmp_path / f"{name}.csv"
        np.savetxt(path, np.column_stack([y, X]), delimiter=",", fmt="%.17g")
        return str(path)
    path = tmp_path / f"{name}.svm"
    with open(path, "w") as f:
        for i in range(len(y)):
            f.write(repr(float(y[i])) + "".join(f" {j}:{float(X[i, j])!r}" for j in np.flatnonzero(X[i])) + "\n")
    return str(path)


@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("fmt", ["csr", "csc", "csv", "csv_two_round", "libsvm"])
@pytest.mark.parametrize("dataset", list(DATASETS))
def test_grouped_storage_inputs(dataset, fmt, layout, recorder, tmp_path):
    """scipy.sparse (binned without densifying) and text files (streamed into the groups), for training
    and validation data, against upstream on the same input."""
    rec = recorder(f"{dataset}[{fmt},{layout}]")
    X, y, Xv, yv = _split(*DATASETS[dataset])
    params = {**BASE, **LAYOUTS[layout], "num_threads": 1}
    if fmt == "csv_two_round":
        params["two_round"] = True
    text = fmt in ("csv", "csv_two_round", "libsvm")
    train_in, valid_in = _to_input(fmt, X, y, tmp_path, "train"), _to_input(fmt, Xv, yv, tmp_path, "valid")

    def run(mod):
        h: Dict[str, Any] = {}
        ds = mod.Dataset(train_in, label=None if text else y, params=params, free_raw_data=False)
        valid = ds.create_valid(valid_in, label=None if text else yv)
        b = mod.train(params, ds, num_boost_round=ROUNDS, valid_sets=[valid], callbacks=[mod.record_evaluation(h)])
        return b, h

    (rs, hrs), (up, hup) = run(lgb_rs), run(lgb_up)
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    for m, v in hup["valid_0"].items():
        rec.compare(f"valid_0.{m}[per iteration]", "metrics", hrs["valid_0"].get(m), v)
    rec.compare("raw_score[holdout]", "predictions", rs.predict(Xv, raw_score=True), up.predict(Xv, raw_score=True))
    rec.finish()


@pytest.mark.parametrize("dataset", list(DATASETS))
def test_grouped_storage_dump_and_binary(dataset, recorder, tmp_path):
    """``_dump_text`` (num_groups and every feature's bins) against upstream, and a binary file round trip
    that keeps the groups (the model trained from it equals upstream's from the array)."""
    rec = recorder(dataset)
    X, y, _, _ = _split(*DATASETS[dataset])
    params = {**BASE, "num_threads": 1, "force_row_wise": True}
    for mod, name in ((lgb_rs, "rs"), (lgb_up, "up")):
        mod.Dataset(X, label=y, params=params).construct()._dump_text(tmp_path / f"{name}.txt")
    rec.compare("_dump_text", "bins", (tmp_path / "rs.txt").read_text(), (tmp_path / "up.txt").read_text())
    lgb_rs.Dataset(X, label=y, params=params).save_binary(tmp_path / "rs.bin")
    up = lgb_up.train(params, lgb_up.Dataset(X, label=y, params=params), num_boost_round=ROUNDS)
    for layout, lp in LAYOUTS.items():
        p = {**params, "force_row_wise": False, **lp}
        rs = lgb_rs.train(p, lgb_rs.Dataset(str(tmp_path / "rs.bin"), params=p), num_boost_round=ROUNDS)
        want = up if layout == "row_wise" else lgb_up.train(p, lgb_up.Dataset(X, label=y, params=p),
                                                            num_boost_round=ROUNDS)
        rec.compare(f"model_text[binary file, {layout}]", "model_text", rs.model_to_string(), want.model_to_string())
    rec.finish()


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_forced_splits_on_bundles(layout, threads, recorder, tmp_path):
    """Forced splits on bundled and multi-value features, including below leaves that are not split
    (which read the histograms upstream's group-wise builds left in the pool)."""
    rec = recorder(f"{layout}[{threads} threads]")
    X, y, Xv, yv = _split(*DATASETS["onehot"])
    # features 5 and 12 are in the exclusive block, 45 and 50 in the sparse part
    spec = {"feature": 0, "threshold": 0.0,
            "left": {"feature": 5, "threshold": -0.3, "right": {"feature": 50, "threshold": -0.1}},
            "right": {"feature": 45, "threshold": -0.5, "right": {"feature": 12, "threshold": -0.2}}}
    path = tmp_path / "forced.json"
    path.write_text(json.dumps(spec))
    params = {**BASE, **LAYOUTS[layout], "num_threads": threads, "forcedsplits_filename": str(path),
              "max_depth": 4, "min_data_in_leaf": 30}
    (rs, _), (up, _) = (_train(mod, X, y, Xv, yv, params, valid=False) for mod in (lgb_rs, lgb_up))
    if threads == 1:
        rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    else:
        rec.compare("trees (4 threads)", "model_text", _trees(rs), _trees(up))
        rec.compare("raw_score[holdout] (4 threads)", "multithread", rs.predict(Xv, raw_score=True),
                    up.predict(Xv, raw_score=True))
    rec.finish()
