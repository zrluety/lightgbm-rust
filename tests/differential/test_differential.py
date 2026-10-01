"""Differential tests: lightgbm-rust vs LightGBM 4.7.0 on the regression/binary subset.

Each test compares intermediate and final quantities; tolerances come from
tests/tolerances.toml (see its rationale entries).
"""

from __future__ import annotations

import math

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

from .conftest import (
    CASES,
    DETERMINISTIC,
    case_ids,
    parse_dump_text,
    rust_tree_arrays,
    train_both,
    upstream_tree_arrays,
)

_BIN_PARAMS = ("max_bin", "min_data_in_bin", "bin_construct_sample_cnt", "data_random_seed", "seed", "use_missing",
               "zero_as_missing", "feature_pre_filter", "min_data_in_leaf")


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_bins(case, recorder, tmp_path):
    rec = recorder(case.name)
    params = {k: v for k, v in case.full_params.items() if k in _BIN_PARAMS}
    params["verbosity"] = -1
    up = lgb_up.Dataset(case.X, label=case.y, params=params, free_raw_data=False).construct()
    up._dump_text(tmp_path / "up.txt")
    up_cols = parse_dump_text(tmp_path / "up.txt", len(case.y))
    rs = lgb_rs.Dataset(case.X, label=case.y, params=params, free_raw_data=False).construct()
    used_rs = [j for j in range(case.X.shape[1]) if rs._bin_indices(j) is not None]
    used_up = [j for j, c in enumerate(up_cols) if c is not None]
    rec.compare("used_features", "bins", used_rs, used_up)
    for j in used_up:
        if j in used_rs:
            rec.compare(f"bin_index[f{j}]", "bins", rs._bin_indices(j), up_cols[j])
    rec.finish()


def _seqsum(x):
    """Left-to-right double sum (upstream's single-thread loop); np.sum is pairwise."""
    return float(np.cumsum(x, dtype=np.float64)[-1])


def _reference_gradients(case):
    """float64 NumPy evaluation of upstream objective formulas, rounded to float32."""
    p = case.full_params
    y = case.y.astype(np.float32).astype(np.float64)
    w = None if case.weight is None else case.weight.astype(np.float32).astype(np.float64)
    ww = np.ones_like(y) if w is None else w
    if case.objective == "regression":
        if p.get("reg_sqrt"):
            # upstream stores the transformed label as label_t (float32)
            y = (np.sign(y) * np.sqrt(np.abs(y))).astype(np.float32).astype(np.float64)
        if case.init_score is not None:
            score = case.init_score.astype(np.float64)
        elif p.get("boost_from_average", True):
            score = np.full_like(y, _seqsum(y * ww) / _seqsum(ww))
        else:
            score = np.zeros_like(y)
        # upstream: static_cast<score_t>(static_cast<score_t>(score - label) * weight)
        g = (score - y).astype(np.float32)
        if w is not None:
            g = g * w.astype(np.float32)
        h = np.ones_like(y, dtype=np.float32) if w is None else w.astype(np.float32)
        return g, h
    sigmoid = p.get("sigmoid", 1.0)
    pos = y > 0
    cnt_pos, cnt_neg = int(pos.sum()), int((~pos).sum())
    lw = [1.0, 1.0]
    if p.get("is_unbalance") and cnt_pos > 0 and cnt_neg > 0:
        if cnt_pos > cnt_neg:
            lw = [cnt_pos / cnt_neg, 1.0]
        else:
            lw = [1.0, cnt_neg / cnt_pos]
    lw[1] *= p.get("scale_pos_weight", 1.0)
    if case.init_score is not None:
        score = case.init_score.astype(np.float64)
    else:
        pavg = _seqsum(pos * ww) / _seqsum(ww)
        pavg = min(max(pavg, 1e-15), 1 - 1e-15)
        score = np.full_like(y, math.log(pavg / (1 - pavg)) / sigmoid)
    label = np.where(pos, 1.0, -1.0)
    label_weight = np.where(pos, lw[1], lw[0])
    response = -label * sigmoid / (1.0 + np.exp(label * sigmoid * score))
    abs_r = np.abs(response)
    g = response * label_weight * ww
    h = abs_r * (sigmoid - abs_r) * label_weight * ww
    return g.astype(np.float32), h.astype(np.float32)


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_first_iteration_gradients(case, recorder):
    rec = recorder(case.name)
    ds = lgb_rs.Dataset(case.X, label=case.y, weight=case.weight, init_score=case.init_score)
    bst = lgb_rs.Booster(case.full_params, ds)
    bst.update()
    g, h = bst._last_gradients()
    g_ref, h_ref = _reference_gradients(case)
    rec.compare("gradient", "gradients", g, g_ref, note="reference: NumPy formula")
    rec.compare("hessian", "gradients", h, h_ref, note="reference: NumPy formula")
    rec.finish()


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_training_parity(case, recorder):
    rec = recorder(case.name)
    rs, up = train_both(case, valid=False)

    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())

    t_rs, t_up = rust_tree_arrays(rs), upstream_tree_arrays(up)
    rec.compare("num_trees", "tree_structure", len(t_rs), len(t_up))
    for i, (a, b) in enumerate(zip(t_rs, t_up)):
        if a["num_leaves"] != b["num_leaves"]:
            rec.compare(f"tree{i}.num_leaves", "tree_structure", a["num_leaves"], b["num_leaves"])
            break
        for k in ("split_feature", "threshold", "decision_type", "left_child", "right_child", "internal_count",
                  "leaf_count"):
            rec.compare(f"tree{i}.{k}", "tree_structure", a[k], b[k])
        rec.compare(f"tree{i}.split_gain", "split_gain", a["split_gain"], b["split_gain"])
        for k in ("leaf_value", "leaf_weight", "internal_value", "internal_weight"):
            rec.compare(f"tree{i}.{k}", "leaf_value", a[k], b[k])
        rec.compare(f"tree{i}.shrinkage", "leaf_value", a["shrinkage"], b["shrinkage"])

    for name, X in (("train", case.X), ("holdout", case.Xv)):
        rec.compare(f"raw_score[{name}]", "predictions", rs.predict(X, raw_score=True), up.predict(X, raw_score=True))
        rec.compare(f"prediction[{name}]", "predictions", rs.predict(X), up.predict(X))
    rec.compare("leaf_index[holdout]", "tree_structure", rs.predict(case.Xv, pred_leaf=True),
                up.predict(case.Xv, pred_leaf=True))
    half = max(case.num_boost_round // 2, 1)
    rec.compare("raw_score[iteration window]", "predictions",
                rs.predict(case.Xv, raw_score=True, start_iteration=2, num_iteration=half),
                up.predict(case.Xv, raw_score=True, start_iteration=2, num_iteration=half))
    for imp in ("split", "gain"):
        rec.compare(f"feature_importance[{imp}]", "split_gain", rs.feature_importance(imp),
                    up.feature_importance(imp))
    rec.finish()


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_model_cross_loading(case, recorder):
    """Upstream reads our model files and we read upstream's, with identical predictions."""
    rec = recorder(case.name)
    rs, up = train_both(case, valid=False)
    up_reads_rs = lgb_up.Booster(model_str=rs.model_to_string())
    rs_reads_up = lgb_rs.Booster(model_str=up.model_to_string())
    rec.compare("upstream(rust model)", "predictions", up_reads_rs.predict(case.Xv), rs.predict(case.Xv))
    rec.compare("rust(upstream model)", "predictions", rs_reads_up.predict(case.Xv), up.predict(case.Xv))
    rec.compare("re-saved upstream model", "model_text", rs_reads_up.model_to_string(), up.model_to_string())
    rec.finish()


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_metrics_and_early_stopping(case, recorder):
    rec = recorder(case.name)
    hist = {}

    def cbs(mod):
        h = {}
        hist[mod.__name__] = h
        return [mod.record_evaluation(h), mod.early_stopping(5, verbose=False)]

    rs, up = train_both(case, rounds=200, callbacks_factory=cbs)
    h_rs, h_up = hist["lightgbm_rust"], hist["lightgbm"]
    rec.compare("metric_names", "tree_structure", sorted(h_rs["valid_0"]), sorted(h_up["valid_0"]))
    for m in h_up["valid_0"]:
        if m in h_rs["valid_0"]:
            rec.compare(f"valid_0.{m}[per iteration]", "metrics", h_rs["valid_0"][m], h_up["valid_0"][m])
    rec.compare("best_iteration", "tree_structure", rs.best_iteration, up.best_iteration)
    for m, v in up.best_score["valid_0"].items():
        rec.compare(f"best_score.{m}", "metrics", rs.best_score["valid_0"].get(m, float("nan")), v)
    rec.finish()


MT_CASES = [c for c in CASES if c.name in ("reg_basic", "reg_nan_zero", "bin_basic", "reg_100_rounds", "bin_100_rounds")]


@pytest.mark.parametrize("case", MT_CASES, ids=[c.name for c in MT_CASES])
def test_multithread(case, recorder):
    """Reported separately: num_threads=4 on both engines."""
    rec = recorder(case.name)
    params = {**case.full_params, "num_threads": 4}
    rs, up = train_both(case, params=params, valid=False)
    rec.compare("raw_score[holdout] (4 threads)", "multithread", rs.predict(case.Xv, raw_score=True),
                up.predict(case.Xv, raw_score=True))
    rec.compare("leaf_value (4 threads, row-wise)", "multithread", leaf_values(rs), leaf_values(up))

    # col-wise histograms are feature-parallel, so the thread count cannot
    # change any sum (row-wise uses upstream's thread-dependent row blocks).
    def col_wise(threads):
        p = {**case.full_params, "force_row_wise": False, "force_col_wise": True, "num_threads": threads}
        return lgb_rs.train(p, lgb_rs.Dataset(case.X, label=case.y, params=p), num_boost_round=case.num_boost_round)

    def trees(b):
        return b.model_to_string().split("end of trees")[0]

    rec.compare("rust col-wise 4 threads vs 1 thread (trees)", "model_text", trees(col_wise(4)), trees(col_wise(1)))
    rec.finish()


def leaf_values(b):
    return np.concatenate([np.asarray(t["leaf_value"], dtype=float) for t in _trees_of(b)])


def _trees_of(b):
    out = []
    for block in b.model_to_string().split("Tree=")[1:]:
        vals = {}
        for line in block.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                vals[k] = v
        out.append({"leaf_value": [float(x) for x in vals.get("leaf_value", "").split()]})
    return out


def test_reference_is_upstream_not_shim():
    assert lgb_up.__file__ != lgb_rs.__file__
    assert hasattr(lgb_up.basic, "_LIB")
    assert DETERMINISTIC["num_threads"] == 1
