"""Differential tests: lightgbm-rust vs LightGBM 4.7.0 on the regression-family/binary/multiclass subset.

Each test compares intermediate and final quantities; tolerances come from
tests/tolerances.toml (see its rationale entries).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

from .conftest import (
    CASES,
    DETERMINISTIC,
    REGRESSION_OBJECTIVES,
    case_ids,
    make_case,
    parse_dump_text,
    rust_tree_arrays,
    train_both,
    upstream_tree_arrays,
)

_BIN_PARAMS = ("max_bin", "min_data_in_bin", "bin_construct_sample_cnt", "data_random_seed", "seed", "use_missing",
               "zero_as_missing", "feature_pre_filter", "min_data_in_leaf", "categorical_feature",
               "max_bin_by_feature", "forcedbins_filename")


def _dump_bin_params(path):
    """The ``max_bin_by_feature`` and ``forced_bins`` lines of ``_dump_text``."""
    lines = path.read_text().splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("forced_bins: "))
    n_total = int(lines[1].split(": ")[1])
    return "\n".join([ln for ln in lines if ln.startswith("max_bin_by_feature: ")] + lines[i:i + 1 + n_total])


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
    if "max_bin_by_feature" in params or "forcedbins_filename" in params:
        rs._dump_text(tmp_path / "rs.txt")
        rec.compare("dump_bin_params", "bins", _dump_bin_params(tmp_path / "rs.txt"),
                    _dump_bin_params(tmp_path / "up.txt"))
    rec.finish()


def _seqsum(x):
    """Left-to-right double sum (upstream's single-thread loop); np.sum is pairwise."""
    return float(np.cumsum(x, dtype=np.float64)[-1])


def _cexp(x):
    """C library exp (what upstream calls), not NumPy's SIMD implementation."""
    return np.array([math.exp(v) for v in np.asarray(x, dtype=np.float64)])


def _percentile(vals, alpha):
    """upstream PercentileFun; `vals` is float32 (label_t) or float64, arithmetic as in the C++ macro."""
    t = vals.dtype.type
    n = len(vals)
    if n <= 1:
        return vals[0]
    float_pos = (n - 1) * (1.0 - alpha)
    pos = int(float_pos) + 1
    if pos < 1:
        return vals.max()
    if pos >= n:
        return vals.min()
    bias = float_pos - (pos - 1)
    desc = np.sort(vals)[::-1]
    v1, v2 = desc[pos - 1], desc[pos]
    return t(float(v1) - float(t(v1 - v2)) * bias)


def _weighted_percentile(vals, weights, alpha):
    """upstream WeightedPercentileFun (stable sort, double cumulative weights)."""
    t = vals.dtype.type
    n = len(vals)
    if n <= 1:
        return vals[0]
    order = np.argsort(vals, kind="stable")
    cdf = np.cumsum(np.asarray(weights, dtype=np.float64)[order])
    threshold = cdf[-1] * alpha
    pos = min(int(np.searchsorted(cdf, threshold, side="right")), n - 1)
    if pos == 0 or pos == n - 1:
        return vals[order[pos]]
    v1, v2 = vals[order[pos - 1]], vals[order[pos]]
    if cdf[pos] - cdf[pos - 1] >= 1.0:
        return t((threshold - cdf[pos - 1]) / (cdf[pos] - cdf[pos - 1]) * float(t(v2 - v1)) + float(v1))
    return v1


_SQRT_ALLOWED = ("regression", "regression_l1", "fair", "quantile", "mape")


def _reference_regression_gradients(case, y, w):
    """Losses derived from upstream RegressionL2loss (regression_objective.hpp)."""
    p = case.full_params
    obj = case.objective
    f32 = np.float32
    if p.get("reg_sqrt") and obj in _SQRT_ALLOWED:
        y = (np.sign(y) * np.sqrt(np.abs(y))).astype(f32).astype(np.float64)
    y32 = y.astype(f32)
    w32 = None if w is None else w.astype(f32)
    ww = np.ones_like(y) if w is None else w
    alpha = p.get("alpha", 0.9)
    a32 = f32(alpha)
    lw32 = f32(1.0) / np.maximum(f32(1.0), np.abs(y32))
    if w32 is not None:
        lw32 = lw32 * w32

    def mean():
        return _seqsum(y * ww) / _seqsum(ww)

    if case.init_score is not None:
        score = case.init_score.astype(np.float64)
    elif not p.get("boost_from_average", True):
        score = np.zeros_like(y)
    else:
        if obj in ("regression", "huber", "fair"):
            init = mean()
        elif obj in ("poisson", "gamma", "tweedie"):
            m = mean()
            init = math.log(m) if m > 0 else -math.inf
        elif obj in ("regression_l1", "quantile"):
            a = 0.5 if obj == "regression_l1" else float(a32)
            init = float(_percentile(y32, a) if w32 is None else _weighted_percentile(y32, w32, a))
        else:  # mape
            init = float(_weighted_percentile(y32, lw32, 0.5))
        score = np.full_like(y, init)

    def wt(x):
        return x if w is None else x * w

    hess_w = np.ones_like(y, dtype=f32) if w is None else w32
    diff = score - y
    if obj == "regression":
        # upstream: static_cast<score_t>(static_cast<score_t>(score - label) * weight)
        g = diff.astype(f32)
        return (g if w is None else g * w32), hess_w
    if obj == "regression_l1":
        return wt(np.sign(diff)).astype(f32), hess_w
    if obj == "huber":
        inside = np.abs(diff) <= alpha
        g = np.where(inside, wt(diff), wt(np.sign(diff)) * alpha)
        return g.astype(f32), hess_w
    if obj == "fair":
        c = p.get("fair_c", 1.0)
        ax = np.abs(diff)
        return wt(c * diff / (ax + c)).astype(f32), wt(c * c / ((ax + c) * (ax + c))).astype(f32)
    if obj == "poisson":
        emds = math.exp(p.get("poisson_max_delta_step", 0.7))
        e = _cexp(score)
        return wt(e - y).astype(f32), wt(e * emds).astype(f32)
    if obj == "quantile":
        delta = diff.astype(f32)
        pos, neg = f32(1.0) - a32, -a32
        if w32 is None:
            g = np.where(delta >= 0, pos, neg).astype(f32)
        else:
            g = np.where(delta >= 0, pos * w32, neg * w32).astype(f32)
        return g, hess_w
    if obj == "mape":
        return (np.sign(diff) * lw32.astype(np.float64)).astype(f32), hess_w
    if obj == "gamma":
        e = _cexp(-score)
        return wt(1.0 - y * e).astype(f32), wt(y * e).astype(f32)
    if obj == "tweedie":
        rho = p.get("tweedie_variance_power", 1.5)
        e1, e2 = _cexp((1 - rho) * score), _cexp((2 - rho) * score)
        return wt(-y * e1 + e2).astype(f32), wt(-y * (1 - rho) * e1 + (2 - rho) * e2).astype(f32)
    raise AssertionError(f"no reference for objective {obj}")


def _reference_gradients(case):
    """float64 NumPy evaluation of upstream objective formulas, rounded to float32."""
    p = case.full_params
    y = case.y.astype(np.float32).astype(np.float64)
    w = None if case.weight is None else case.weight.astype(np.float32).astype(np.float64)
    ww = np.ones_like(y) if w is None else w
    if case.objective != "regression" and case.objective in REGRESSION_OBJECTIVES:
        return _reference_regression_gradients(case, y, w)
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
    if case.objective in ("cross_entropy", "xentropy", "cross_entropy_lambda"):
        return _reference_xent_gradients(case, y, w)
    if case.objective == "multiclass":
        return _reference_softmax_gradients(case, y, w)
    if case.objective == "multiclassova":
        n, k = len(y), p["num_class"]
        init = None if case.init_score is None else case.init_score.astype(np.float64)
        parts = [_reference_binary_gradients(p, y.astype(np.int64) == c, ww, None if init is None else init[:, c])
                 for c in range(k)]
        return np.concatenate([g for g, _ in parts]), np.concatenate([h for _, h in parts])
    init = None if case.init_score is None else case.init_score.astype(np.float64)
    return _reference_binary_gradients(p, y > 0, ww, init)


def _reference_xent_gradients(case, y, w):
    """upstream CrossEntropy / CrossEntropyLambda (xentropy_objective.hpp)."""
    lam = case.objective == "cross_entropy_lambda"
    ww = np.ones_like(y) if w is None else w
    if case.init_score is not None:
        score = case.init_score.astype(np.float64)
    elif case.full_params.get("boost_from_average", True):
        avg = _seqsum(y * ww) / _seqsum(ww)
        if lam:
            init = math.log(math.expm1(avg))
        else:
            avg = min(max(avg, 1e-15), 1 - 1e-15)
            init = math.log(avg / (1 - avg))
        score = np.full_like(y, init if abs(init) > 1e-15 else 0.0)
    else:
        score = np.zeros_like(y)
    if lam and w is None:
        z = 1 / (1 + _cexp(-score))
        return (z - y).astype(np.float32), (z * (1 - z)).astype(np.float32)
    if lam:
        epf = _cexp(score)
        hhat = np.array([math.log1p(v) for v in epf])
        z = 1 - _cexp(-w * hhat)
        g = (1 - y / z) * w / (1 + 1 / epf)
        c = 1 / (1 - z)
        a = w * epf / ((1 + epf) * (1 + epf))
        b = (c / ((c - 1) * (c - 1))) * (1 + w * epf - c)
        return g.astype(np.float32), (a * (1 + y * b)).astype(np.float32)
    big = score > -37.0
    e = np.where(big, _cexp(-score), _cexp(score))
    one_minus_y = (np.float32(1) - y.astype(np.float32)).astype(np.float64)  # float arithmetic upstream
    g = np.where(big, (one_minus_y - y * e) / (1 + e), e - y)
    h = np.where(big, e / ((1 + e) * (1 + e)), e)
    return (g * ww).astype(np.float32), (h * ww).astype(np.float32)


def _reference_binary_gradients(p, pos, ww, init):
    """upstream BinaryLogloss; `pos` is the is_pos predicate applied to the labels."""
    sigmoid = p.get("sigmoid", 1.0)
    cnt_pos, cnt_neg = int(pos.sum()), int((~pos).sum())
    if cnt_pos == 0 or cnt_neg == 0:
        # "Contains only one class": no gradients are written
        return np.zeros(len(pos), dtype=np.float32), np.zeros(len(pos), dtype=np.float32)
    lw = [1.0, 1.0]
    if p.get("is_unbalance") and cnt_pos > 0 and cnt_neg > 0:
        if cnt_pos > cnt_neg:
            lw = [cnt_pos / cnt_neg, 1.0]
        else:
            lw = [1.0, cnt_neg / cnt_pos]
    lw[1] *= p.get("scale_pos_weight", 1.0)
    if init is not None:
        score = init
    else:
        pavg = _seqsum(pos * ww) / _seqsum(ww)
        pavg = min(max(pavg, 1e-15), 1 - 1e-15)
        score = np.full(len(pos), math.log(pavg / (1 - pavg)) / sigmoid)
    label = np.where(pos, 1.0, -1.0)
    label_weight = np.where(pos, lw[1], lw[0])
    response = -label * sigmoid / (1.0 + _cexp(label * sigmoid * score))
    abs_r = np.abs(response)
    g = response * label_weight * ww
    h = abs_r * (sigmoid - abs_r) * label_weight * ww
    return g.astype(np.float32), h.astype(np.float32)


def _reference_softmax_gradients(case, y, w):
    """upstream MulticlassSoftmax (class-major output, Common::Softmax per row)."""
    p = case.full_params
    k = p["num_class"]
    n = len(y)
    label = y.astype(np.int64)
    if case.init_score is not None:
        score = case.init_score.astype(np.float64)
    else:
        prior = np.zeros(k)
        for i in range(n):
            prior[label[i]] += 1.0 if w is None else w[i]
        prior /= n if w is None else _seqsum(w)
        init = [math.log(max(1e-15, q)) for q in prior]
        if not p.get("boost_from_average", True):
            init = [0.0] * k
        score = np.tile(np.array(init), (n, 1))
    factor = k / float(np.float32(k) - np.float32(1.0))
    g = np.empty((k, n), dtype=np.float32)
    h = np.empty((k, n), dtype=np.float32)
    for i in range(n):
        row = score[i]
        wmax = row[0]
        for v in row[1:]:
            wmax = wmax if v < wmax else v
        e = [math.exp(v - wmax) for v in row]
        wsum = 0.0
        for v in e:
            wsum += v
        wi = 1.0 if w is None else w[i]
        for c in range(k):
            q = e[c] / wsum
            gi = q - 1.0 if label[i] == c else q
            hi = factor * q * (1.0 - q)
            g[c, i] = gi if w is None else gi * wi
            h[c, i] = hi if w is None else hi * wi
    return g.ravel(), h.ravel()


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


MT_CASES = [c for c in CASES if c.name in ("reg_basic", "reg_nan_zero", "bin_basic", "reg_100_rounds", "bin_100_rounds",
                                           "l1_weighted", "quantile_basic", "mape_weighted", "poisson_weighted",
                                           "tweedie_basic", "mc_weighted", "ova_basic", "bag_basic", "goss_basic",
                                           "goss_no_subset", "sampling_combo", "cat_basic", "cat_100_rounds")]


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
    def col_wise(mod, threads):
        p = {**case.full_params, "force_row_wise": False, "force_col_wise": True, "num_threads": threads}
        return mod.train(p, mod.Dataset(case.X, label=case.y, params=p), num_boost_round=case.num_boost_round)

    def trees(b):
        return b.model_to_string().split("end of trees")[0]

    if case.full_params.get("data_sample_strategy") == "goss" or case.full_params.get("boosting") == "goss":
        # GOSS picks its top rows per thread chunk (upstream goss.hpp), so the
        # thread count changes the sample by design; compare against upstream instead.
        rec.compare("col-wise 4 threads (trees, GOSS)", "model_text", trees(col_wise(lgb_rs, 4)),
                    trees(col_wise(lgb_up, 4)))
    else:
        rec.compare("rust col-wise 4 threads vs 1 thread (trees)", "model_text", trees(col_wise(lgb_rs, 4)),
                    trees(col_wise(lgb_rs, 1)))
    rec.finish()


def _split_internal(obj, path=""):
    """Flatten a dump_model dict into (exact items, internal_value/internal_weight values)."""
    exact, internal = [], []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("internal_value", "internal_weight"):
                internal.append(v)
            else:
                e, i = _split_internal(v, f"{path}.{k}")
                exact += e
                internal += i
    elif isinstance(obj, list):
        for j, v in enumerate(obj):
            e, i = _split_internal(v, f"{path}[{j}]")
            exact += e
            internal += i
    else:
        exact.append(f"{path}={obj!r}")
    return exact, internal


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_dump_model(case, recorder):
    """JSON dump_model and trees_to_dataframe, for trained models and for the same loaded upstream model."""
    rec = recorder(case.name)
    rs, up = train_both(case, valid=False)
    e_rs, i_rs = _split_internal(rs.dump_model())
    e_up, i_up = _split_internal(up.dump_model())
    rec.compare("dump_model (trained, all but internal_*)", "model_text", "\n".join(e_rs), "\n".join(e_up))
    rec.compare("dump_model internal_value/weight (trained)", "leaf_value", i_rs, i_up)
    text = up.model_to_string()
    b_rs, b_up = lgb_rs.Booster(model_str=text), lgb_up.Booster(model_str=text)
    rec.compare("dump_model (loaded upstream model)", "model_text", repr(b_rs.dump_model()), repr(b_up.dump_model()))
    rec.compare("dump_model(num_iteration=3, start_iteration=2, gain) (loaded)", "model_text",
                repr(b_rs.dump_model(3, 2, "gain")), repr(b_up.dump_model(3, 2, "gain")))
    df_rs = lgb_rs.Booster(model_str=text).trees_to_dataframe()
    df_up = lgb_up.Booster(model_str=text).trees_to_dataframe()
    rec.compare("trees_to_dataframe (loaded upstream model)", "model_text", df_rs.to_csv(), df_up.to_csv())
    rec.compare("lower/upper bound", "predictions", [rs.lower_bound(), rs.upper_bound()],
                [up.lower_bound(), up.upper_bound()])
    rec.finish()


CV_CASES = [c for c in CASES if c.name in ("reg_basic", "bin_weighted", "mc_basic", "l1_weighted", "bag_basic",
                                           "bin_init_score", "cat_basic", "cat_binary", "xent_weighted",
                                           "xentlambda_weighted")]


@pytest.mark.parametrize("case", CV_CASES, ids=[c.name for c in CV_CASES])
def test_cv(case, recorder):
    """cv(): fold assignment (stratified for classification), Dataset.subset, per-fold boosters, aggregation."""
    rec = recorder(case.name)
    classification = case.objective in ("binary", "multiclass", "multiclassova")

    def run(mod):
        ds = mod.Dataset(case.X, label=case.y, weight=case.weight, init_score=case.init_score)
        r = mod.cv(dict(case.full_params, early_stopping_round=5), ds, num_boost_round=60, nfold=4,
                   stratified=classification, eval_train_metric=True, return_cvbooster=True, seed=7)
        cvb = r.pop("cvbooster")
        return r, cvb

    (r_rs, b_rs), (r_up, b_up) = run(lgb_rs), run(lgb_up)
    rec.compare("result keys", "tree_structure", list(r_rs), list(r_up))
    for k in r_up:
        rec.compare(f"cv[{k}]", "metrics", r_rs.get(k, []), r_up[k])
    rec.compare("best_iteration", "tree_structure", b_rs.best_iteration, b_up.best_iteration)
    rec.compare("fold models", "model_text", b_rs.model_to_string(), b_up.model_to_string())
    rec.compare("fold predictions", "predictions", np.asarray(b_rs.predict(case.Xv)), np.asarray(b_up.predict(case.Xv)))
    rec.finish()


CONT_CASES = [c for c in CASES if c.name in ("reg_basic", "bin_weighted", "mc_basic", "l1_weighted", "bag_basic",
                                             "goss_basic", "bin_init_score", "ova_basic", "cat_basic",
                                             "xentlambda_basic")]


@pytest.mark.parametrize("init_kind", ["booster", "model_file", "reused_dataset"])
@pytest.mark.parametrize("case", CONT_CASES, ids=[c.name for c in CONT_CASES])
def test_continued_training(case, init_kind, recorder, tmp_path):
    """init_model: init trees merged first, init-model raw scores as init_score, validation, early stopping."""
    rec = recorder(f"{case.name}[{init_kind}]")
    hist = {}

    def run(mod):
        params = case.full_params
        ds = mod.Dataset(case.X, label=case.y, weight=case.weight, init_score=case.init_score, free_raw_data=False)
        first = mod.train(params, ds, num_boost_round=case.num_boost_round // 2)
        init = first
        if init_kind == "model_file":
            init = tmp_path / f"{mod.__name__}.txt"
            first.save_model(init)
        ds2 = ds if init_kind == "reused_dataset" else mod.Dataset(case.X, label=case.y, weight=case.weight)
        h = hist.setdefault(mod.__name__, {})
        return mod.train(params, ds2, num_boost_round=case.num_boost_round, init_model=init,
                         valid_sets=[ds2.create_valid(case.Xv, label=case.yv)],
                         callbacks=[mod.record_evaluation(h), mod.early_stopping(5, verbose=False)])

    rs, up = run(lgb_rs), run(lgb_up)
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    rec.compare("raw_score[holdout]", "predictions", rs.predict(case.Xv, raw_score=True),
                up.predict(case.Xv, raw_score=True))
    rec.compare("valid_0[per iteration]", "metrics", hist["lightgbm_rust"], hist["lightgbm"])
    rec.compare("current/best iteration", "tree_structure", [rs.current_iteration(), rs.best_iteration],
                [up.current_iteration(), up.best_iteration])
    rec.finish()


@pytest.mark.parametrize("case", [c for c in CONT_CASES if c.name in ("reg_basic", "mc_basic")], ids=lambda c: c.name)
def test_cv_init_model(case, recorder):
    rec = recorder(case.name)

    def run(mod):
        init = mod.train(case.full_params, mod.Dataset(case.X, label=case.y), num_boost_round=5)
        r = mod.cv(case.full_params, mod.Dataset(case.X, label=case.y), num_boost_round=10, nfold=3, stratified=False,
                   init_model=init, return_cvbooster=True, seed=3)
        return r.pop("cvbooster"), r

    (b_rs, r_rs), (b_up, r_up) = run(lgb_rs), run(lgb_up)
    rec.compare("cv results", "metrics", r_rs, r_up)
    rec.compare("fold models", "model_text", b_rs.model_to_string(), b_up.model_to_string())
    rec.finish()


ARROW_CASES = [c for c in CASES if c.name in ("reg_basic", "reg_nan_zero", "bin_weighted", "mc_basic", "bin_init_score",
                                              "reg_sparse", "cat_basic")]


def _arrow_table(pa, X):
    """Chunked columns of mixed types: int64 with nulls, float32, float64 with nulls for NaN, and bool."""
    cols, names = [], []
    for j in range(X.shape[1]):
        col = X[:, j]
        half = len(col) // 3
        if j % 4 == 0:
            ints = np.where(np.isnan(col), 0, np.round(col * 10)).astype(np.int64)
            chunks = [pa.array(ints[:half], mask=np.isnan(col[:half])), pa.array(ints[half:], mask=np.isnan(col[half:]))]
        elif j % 4 == 1:
            chunks = [pa.array(col.astype(np.float32))]
        elif j % 4 == 2:
            chunks = [pa.array(col[:half], mask=np.isnan(col[:half])), pa.array([], type=pa.float64()),
                      pa.array(col[half:], mask=np.isnan(col[half:]))]
        else:
            chunks = [pa.array(col > 0)]
        cols.append(pa.chunked_array(chunks))
        names.append(f"f{j}")
    return pa.Table.from_arrays(cols, names=names)


@pytest.mark.parametrize("frame", ["pyarrow", "polars"])
@pytest.mark.parametrize("case", ARROW_CASES, ids=[c.name for c in ARROW_CASES])
def test_arrow_inputs(case, frame, recorder):
    """pyarrow Table / polars DataFrame features and Arrow label/weight/init_score, against upstream on the same input."""
    pa = pytest.importorskip("pyarrow")
    rec = recorder(f"{case.name}[{frame}]")
    table, valid = _arrow_table(pa, case.X), _arrow_table(pa, case.Xv)
    label = pa.chunked_array([pa.array(case.y[:100]), pa.array(case.y[100:])])
    weight = None if case.weight is None else pa.chunked_array([pa.array(case.weight)])
    init_score = case.init_score
    if init_score is not None:
        init_score = (pa.chunked_array([pa.array(init_score)]) if init_score.ndim == 1 else
                      pa.Table.from_arrays([pa.array(init_score[:, k]) for k in range(init_score.shape[1])],
                                           names=[f"k{k}" for k in range(init_score.shape[1])]))
    if frame == "polars":
        pl = pytest.importorskip("polars")
        table, valid, label = pl.from_arrow(table), pl.from_arrow(valid), pl.from_arrow(label)
        weight = None if weight is None else pl.from_arrow(weight)
        if init_score is not None:
            init_score = pl.from_arrow(init_score)

    def run(mod):
        ds = mod.Dataset(table, label=label, weight=weight, init_score=init_score)
        return mod.train(case.full_params, ds, num_boost_round=case.num_boost_round)

    rs, up = run(lgb_rs), run(lgb_up)
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    rec.compare("raw_score[holdout table]", "predictions", rs.predict(valid, raw_score=True),
                up.predict(valid, raw_score=True))
    rec.compare("pred_leaf[holdout table]", "tree_structure", rs.predict(valid, pred_leaf=True),
                up.predict(valid, pred_leaf=True))
    rec.finish()


# --------------------------------------------------------------------------- scipy.sparse


def _most_freq_not_zero(case):
    """Column 1 is mostly 1.0, so its most frequent bin differs from the zero bin
    (upstream's CSC loader then pushes every row, and CSR pushes zeros)."""
    def mostly_one(X, seed):
        X = X.copy()
        X[np.random.default_rng(seed).random(len(X)) < 0.85, 1] = 1.0
        return X
    return replace(case, name=f"{case.name}_most_freq_one", X=mostly_one(case.X, 1), Xv=mostly_one(case.Xv, 2))


SPARSE_CASES = [c for c in CASES if c.name in ("reg_basic", "reg_nan_zero", "reg_sparse", "bin_weighted", "mc_basic",
                                               "bag_basic", "bin_init_score", "cat_basic")]
SPARSE_CASES.append(_most_freq_not_zero(next(c for c in CASES if c.name == "reg_sparse")))
SPARSE_FORMATS = ["csr", "csc", "csr_f32_int64_indptr", "csr_explicit_zeros", "coo"]


def _to_sparse(X, fmt):
    sp = pytest.importorskip("scipy.sparse")
    if fmt == "csr":
        return sp.csr_matrix(X)
    if fmt == "csc":
        return sp.csc_matrix(X)
    if fmt == "csr_f32_int64_indptr":
        m = sp.csr_matrix(X.astype(np.float32))
        m.indptr = m.indptr.astype(np.int64)
        return m
    if fmt == "csr_explicit_zeros":
        rows, cols = np.indices(X.shape)
        m = sp.csr_matrix((X.ravel(), (rows.ravel(), cols.ravel())), shape=X.shape)
        assert m.nnz == X.size
        return m
    return sp.coo_matrix(X)


@pytest.mark.parametrize("fmt", SPARSE_FORMATS)
@pytest.mark.parametrize("case", SPARSE_CASES, ids=[c.name for c in SPARSE_CASES])
def test_sparse_inputs(case, fmt, recorder):
    """scipy.sparse training, validation and prediction input, against upstream on the same input."""
    rec = recorder(f"{case.name}[{fmt}]")
    X, Xv = _to_sparse(case.X, fmt), _to_sparse(case.Xv, fmt)
    hist = {}

    def run(mod):
        h = {}
        hist[mod.__name__] = h
        ds = mod.Dataset(X, label=case.y, weight=case.weight, init_score=case.init_score, free_raw_data=False)
        valid = ds.create_valid(Xv, label=case.yv)
        return mod.train(case.full_params, ds, num_boost_round=case.num_boost_round, valid_sets=[valid],
                         callbacks=[mod.record_evaluation(h)])

    rs, up = run(lgb_rs), run(lgb_up)
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    for m, v in hist["lightgbm"]["valid_0"].items():
        rec.compare(f"valid_0.{m}[per iteration]", "metrics", hist["lightgbm_rust"]["valid_0"].get(m), v)
    rec.compare("raw_score[holdout sparse]", "predictions", rs.predict(Xv, raw_score=True),
                up.predict(Xv, raw_score=True))
    rec.compare("predict[holdout sparse]", "predictions", rs.predict(Xv), up.predict(Xv))
    rec.compare("pred_leaf[holdout sparse]", "tree_structure", rs.predict(Xv, pred_leaf=True),
                up.predict(Xv, pred_leaf=True))
    dense = lgb_rs.train(case.full_params, lgb_rs.Dataset(X.toarray(), label=case.y, weight=case.weight,
                                                          init_score=case.init_score),
                         num_boost_round=case.num_boost_round)
    rec.compare("model_text[same data, dense]", "model_text", dense.model_to_string(), up.model_to_string())
    rec.finish()


@pytest.mark.parametrize("case", CASES, ids=case_ids())
def test_pred_contrib(case, recorder):
    """SHAP values (dense output) of identically trained models, and of upstream's model loaded into lightgbm-rust."""
    rec = recorder(case.name)
    rs, up = train_both(case, valid=False)
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    rec.compare("pred_contrib[holdout]", "predictions", rs.predict(case.Xv, pred_contrib=True),
                up.predict(case.Xv, pred_contrib=True))
    window = dict(start_iteration=3, num_iteration=10)
    rec.compare("pred_contrib[holdout, iterations 3..13]", "predictions",
                rs.predict(case.Xv, pred_contrib=True, **window), up.predict(case.Xv, pred_contrib=True, **window))
    loaded = lgb_rs.Booster(model_str=up.model_to_string())
    rec.compare("pred_contrib[upstream model]", "predictions", loaded.predict(case.Xv, pred_contrib=True),
                up.predict(case.Xv, pred_contrib=True))
    rec.finish()


CONTRIB_SPARSE_CASES = [c for c in SPARSE_CASES if c.name in ("reg_nan_zero", "reg_sparse", "mc_basic",
                                                              "reg_sparse_most_freq_one")]


@pytest.mark.parametrize("fmt", ["csr", "csc", "csr_f32_int64_indptr"])
@pytest.mark.parametrize("case", CONTRIB_SPARSE_CASES, ids=[c.name for c in CONTRIB_SPARSE_CASES])
def test_pred_contrib_sparse(case, fmt, recorder):
    """Sparse SHAP output: matrix type, dtypes, shape, stored cells and values, against upstream."""
    import scipy.sparse as sp

    rec = recorder(f"{case.name}[{fmt}]")
    rs, up = train_both(case, valid=False)
    Xv = _to_sparse(case.Xv, fmt)
    got, want = rs.predict(Xv, pred_contrib=True), up.predict(Xv, pred_contrib=True)
    got, want = (got, want) if isinstance(want, list) else ([got], [want])
    rec.compare("num_matrices", "tree_structure", len(got), len(want))
    for k, (g, w) in enumerate(zip(got, want)):
        rec.compare(f"[{k}] format", "tree_structure", float(sp.isspmatrix_csr(g)), float(sp.isspmatrix_csr(w)))
        rec.compare(f"[{k}] dtypes", "tree_structure", float(g.dtype == w.dtype and g.indptr.dtype == w.indptr.dtype),
                    1.0)
        rec.compare(f"[{k}] shape", "tree_structure", list(g.shape), list(w.shape))
        cells = lambda m: sorted(zip(*m.nonzero())) if m.nnz == 0 else sorted(zip(*m.tocoo().coords))  # noqa: E731
        rec.compare(f"[{k}] nnz", "tree_structure", g.nnz, w.nnz)
        rec.compare(f"[{k}] stored cells equal", "tree_structure", float(cells(g) == cells(w)), 1.0)
        rec.compare(f"[{k}] values", "predictions", g.toarray(), w.toarray())
    rec.finish()


def test_sparse_input_errors():
    """upstream argument checks in Dataset.__init_from_csr and the predictor."""
    sp = pytest.importorskip("scipy.sparse")
    X = sp.csr_matrix(np.eye(20, 3, dtype=np.int64))
    for mod in (lgb_rs, lgb_up):
        with pytest.raises(TypeError, match=r"Expected np.float32 or np.float64, met type\(int64\)"):
            mod.Dataset(X, label=np.zeros(20)).construct()
        with pytest.raises(TypeError, match="Cannot initialize Dataset from coo_matrix"):
            mod.Dataset(sp.coo_matrix(X), label=np.zeros(20)).construct()
        bad = sp.csr_matrix(np.eye(20, 3))
        bad.indices = bad.indices[:-1]
        with pytest.raises(ValueError, match="Length mismatch: 2 vs 3"):
            mod.Dataset(bad, label=np.zeros(20)).construct()


# --------------------------------------------------------------------------- ranking


@dataclass
class RankCase:
    name: str
    objective: str
    params: Dict[str, Any]
    X: np.ndarray
    y: np.ndarray
    group: np.ndarray
    Xv: np.ndarray
    yv: np.ndarray
    groupv: np.ndarray
    weight: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    positionv: Optional[np.ndarray] = None
    rounds: int = 30

    @property
    def full_params(self) -> Dict[str, Any]:
        return {"objective": self.objective, **DETERMINISTIC, **self.params}


def make_rank_case(name: str, objective: str, params: Dict[str, Any], *, n_queries: int = 150,
                   weighted: bool = False, positions: bool = False, seed: int = 0, rounds: int = 30,
                   max_label: int = 4) -> RankCase:
    """Graded relevance 0..max_label from the features; query sizes 1..39; the first query has no relevant rows.

    With `positions`, each row gets a display position within its query and relevance is hidden
    (set to 0) more often at deep positions, i.e. position-biased labels.
    """
    rng = np.random.default_rng(seed)

    def part(nq):
        sizes = rng.integers(1, 40, size=nq)
        n = int(sizes.sum())
        X = rng.normal(size=(n, 6))
        f = np.tanh(X[:, 0]) * 2 + 0.5 * X[:, 1] - 0.3 * X[:, 2] * (X[:, 0] > 0)
        rel = np.clip(np.round(f + rng.normal(scale=0.7, size=n) + 1), 0, max_label)
        rel[: sizes[0]] = 0
        pos = None
        if positions:
            pos = np.concatenate([rng.permutation(s) for s in sizes]).astype(np.int32)
            rel = np.where(rng.random(n) < 1 / (1 + 0.3 * pos), rel, 0)
        return X, rel, sizes, pos

    X, y, g, pos = part(n_queries)
    Xv, yv, gv, posv = part(n_queries // 3)
    w = rng.uniform(0.2, 3.0, size=len(y)) if weighted else None
    return RankCase(name, objective, params, X, y, g, Xv, yv, gv, w, pos, posv, rounds)


RANK_CASES = [
    make_rank_case("rank_basic", "lambdarank", {}),
    make_rank_case("rank_weighted", "lambdarank", {"metric": ["ndcg", "map"]}, weighted=True, seed=1),
    make_rank_case("rank_eval_at_gain", "lambdarank",
                   {"eval_at": [10, 1, 3], "label_gain": [0, 1, 2, 5, 9.5], "lambdarank_truncation_level": 5,
                    "lambdarank_norm": False, "metric": ["map", "ndcg"]}, seed=2),
    make_rank_case("rank_sigmoid", "lambdarank", {"sigmoid": 2.0, "learning_rate": 0.2}, seed=3),
    make_rank_case("rank_position", "lambdarank", {"lambdarank_position_bias_regularization": 0.1},
                   positions=True, seed=4),
    make_rank_case("rank_position_unregularized", "lambdarank", {}, positions=True, weighted=True, seed=5),
    make_rank_case("rank_bagging", "lambdarank", {"bagging_fraction": 0.7, "bagging_freq": 1, "num_leaves": 15},
                   seed=6),
    make_rank_case("xendcg_basic", "rank_xendcg", {}, seed=7),
    make_rank_case("xendcg_seeded_weighted", "rank_xendcg", {"objective_seed": 11, "metric": ["map"]},
                   weighted=True, seed=8),
    make_rank_case("xendcg_alias_seed", "xendcg", {"seed": 3, "eval_at": [2, 4]}, seed=9),
    # bagging_by_query: average bag rate <= 0.5 trains on a subset; above it upstream's out-of-bag
    # update reads the stale tail of its index buffer
    make_rank_case("rank_bagging_by_query", "lambdarank",
                   {"bagging_by_query": True, "bagging_fraction": 0.3, "bagging_freq": 1}, seed=10),
    make_rank_case("rank_bagging_by_query_stale", "lambdarank",
                   {"bagging_by_query": True, "bagging_fraction": 0.7, "bagging_freq": 1, "bagging_seed": 5},
                   seed=11),
    make_rank_case("rank_bagging_by_query_position", "lambdarank",
                   {"bagging_by_query": True, "bagging_fraction": 0.8, "bagging_freq": 2,
                    "lambdarank_position_bias_regularization": 0.1}, positions=True, seed=12),
    make_rank_case("xendcg_bagging_by_query", "rank_xendcg",
                   {"bagging_by_query": True, "bagging_fraction": 0.6, "bagging_freq": 1}, seed=13),
]


def _rank_datasets(mod, case: RankCase):
    train = mod.Dataset(case.X, label=case.y, group=case.group, weight=case.weight, position=case.position,
                        free_raw_data=False)
    valid = train.create_valid(case.Xv, label=case.yv, group=case.groupv, position=case.positionv)
    return train, valid


@pytest.mark.parametrize("case", RANK_CASES, ids=[c.name for c in RANK_CASES])
def test_ranking(case, recorder):
    """lambdarank / rank_xendcg training, ndcg/map metric histories, early stopping, group/position fields."""
    rec = recorder(case.name)
    out = {}
    for mod in (lgb_rs, lgb_up):
        train, valid = _rank_datasets(mod, case)
        h = {}
        bst = mod.train(case.full_params, train, num_boost_round=case.rounds, valid_sets=[train, valid],
                        valid_names=["train", "valid"],
                        callbacks=[mod.record_evaluation(h), mod.early_stopping(5, verbose=False)])
        out[mod.__name__] = (bst, h, train)
    (rs, h_rs, t_rs), (up, h_up, t_up) = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    for name, X in (("train", case.X), ("holdout", case.Xv)):
        rec.compare(f"raw_score[{name}]", "predictions", rs.predict(X, raw_score=True), up.predict(X, raw_score=True))
    rec.compare("leaf_index[holdout]", "tree_structure", rs.predict(case.Xv, pred_leaf=True),
                up.predict(case.Xv, pred_leaf=True))
    rec.compare("metric_names", "tree_structure", [sorted(h_rs.get(k, {})) for k in ("train", "valid")],
                [sorted(h_up[k]) for k in ("train", "valid")])
    for ds in ("train", "valid"):
        for m in h_up[ds]:
            rec.compare(f"{ds}.{m}[per iteration]", "metrics", h_rs.get(ds, {}).get(m, []), h_up[ds][m])
    rec.compare("best_iteration", "tree_structure", rs.best_iteration, up.best_iteration)
    rec.compare("get_group", "tree_structure", t_rs.get_group(), t_up.get_group())
    rec.compare("get_field(group)", "tree_structure", t_rs.get_field("group"), t_up.get_field("group"))
    if case.position is not None:
        rec.compare("get_position", "tree_structure", t_rs.get_position(), t_up.get_position())
    rec.compare("re-saved upstream model", "model_text",
                lgb_rs.Booster(model_str=up.model_to_string()).model_to_string(), up.model_to_string())
    rec.compare("upstream(rust model)", "predictions",
                lgb_up.Booster(model_str=rs.model_to_string()).predict(case.Xv), rs.predict(case.Xv))
    rec.finish()


def test_bagging_by_query_edge_cases():
    """bagging_by_query changes the model, falls back for goss / unsampled configs, and matches upstream
    without query data (empty bags) and for non-ranking objectives."""
    case = next(c for c in RANK_CASES if c.name == "rank_bagging_by_query_stale")

    def fit(mod, params, rounds=10, group=True, fobj=None):
        ds = mod.Dataset(case.X, label=case.y, group=case.group if group else None)
        p = {**DETERMINISTIC, **params}
        if fobj is not None:
            p["objective"] = fobj
        return mod.train(p, ds, num_boost_round=rounds)

    by_query = {"objective": "lambdarank", "bagging_fraction": 0.7, "bagging_freq": 1}
    with_q = fit(lgb_rs, {**by_query, "bagging_by_query": True}).predict(case.Xv, raw_score=True)
    without = fit(lgb_rs, by_query).predict(case.Xv, raw_score=True)
    assert np.max(np.abs(with_q - without)) > 1e-3

    variants = [
        ({"objective": "lambdarank", "bagging_by_query": True, "data_sample_strategy": "goss"}, True),
        ({**by_query, "bagging_by_query": True, "bagging_fraction": 1.0}, True),
        ({"objective": "regression", "bagging_by_query": True, "bagging_fraction": 0.7, "bagging_freq": 1}, True),
        ({"objective": "regression", "bagging_by_query": True, "bagging_fraction": 0.4, "bagging_freq": 1}, False),
        ({"objective": "regression", "bagging_by_query": True, "bagging_fraction": 0.8, "bagging_freq": 1}, False),
    ]
    for params, group in variants:
        rs, up = fit(lgb_rs, params, group=group), fit(lgb_up, params, group=group)
        assert rs.model_to_string() == up.model_to_string(), params
        np.testing.assert_array_equal(rs.predict(case.Xv, raw_score=True), up.predict(case.Xv, raw_score=True))
    assert rs.num_trees() == 1  # no queries: every bag is empty

    def fobj(preds, ds):
        return preds - ds.get_label(), np.ones_like(preds)

    with pytest.raises(lgb_rs.basic.LightGBMError, match="bagging_by_query with custom gradients"):
        fit(lgb_rs, {"bagging_by_query": True, "bagging_fraction": 0.7, "bagging_freq": 1}, fobj=fobj)


# --------------------------------------------------------------------------- save_binary


SAVE_BINARY_CASES = [c for c in CASES if c.name in ("reg_basic", "reg_nan_zero", "bin_weighted", "bin_init_score",
                                                    "mc_basic", "reg_sampled_bins", "reg_sparse", "cat_basic",
                                                    "bag_basic")]


@pytest.mark.parametrize("case", SAVE_BINARY_CASES, ids=[c.name for c in SAVE_BINARY_CASES])
def test_save_binary(case, recorder, tmp_path):
    """Each package's own binary format: a Dataset saved and loaded back (training and validation sets)
    trains the same model as the in-memory Dataset, and the loaded fields match upstream's."""
    rec = recorder(case.name)
    params = case.full_params
    out = {}
    for mod in (lgb_rs, lgb_up):
        d = tmp_path / mod.__name__
        d.mkdir()
        train = mod.Dataset(case.X, label=case.y, weight=case.weight, init_score=case.init_score, params=params)
        valid = train.create_valid(case.Xv, label=case.yv)
        train.save_binary(d / "train.bin")
        valid.save_binary(d / "valid.bin")
        mem = mod.train(params, train, num_boost_round=10, valid_sets=[valid], callbacks=[])
        loaded = mod.Dataset(d / "train.bin", params=params)
        loaded_valid = loaded.create_valid(str(d / "valid.bin"))
        h = {}
        bst = mod.train(params, loaded, num_boost_round=10, valid_sets=[loaded_valid],
                        callbacks=[mod.record_evaluation(h)])
        fields = {f: loaded.get_field(f) for f in ("label", "weight", "init_score", "group")}
        out[mod.__name__] = (mem, bst, h, fields, loaded)
    (mem_rs, rs, h_rs, f_rs, ds_rs), (mem_up, up, h_up, f_up, ds_up) = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("model_text(loaded)", "model_text", rs.model_to_string(), up.model_to_string())
    if case.init_score is None:  # init_score is not saved
        rec.compare("model_text(loaded vs in-memory)", "model_text", rs.model_to_string(), mem_rs.model_to_string())
    rec.compare("raw_score[holdout]", "predictions", rs.predict(case.Xv, raw_score=True),
                up.predict(case.Xv, raw_score=True))
    for m in h_up["valid_0"]:
        rec.compare(f"valid.{m}[per iteration]", "metrics", h_rs["valid_0"][m], h_up["valid_0"][m])
    for f in ("label", "weight", "group"):
        rec.compare(f"loaded {f}", "tree_structure", f_rs[f], f_up[f])
    rec.compare("loaded init_score is None", "tree_structure", f_rs["init_score"] is None,
                f_up["init_score"] is None)
    rec.compare("num_data, num_feature", "tree_structure", [ds_rs.num_data(), ds_rs.num_feature()],
                [ds_up.num_data(), ds_up.num_feature()])
    rec.compare("feature_name", "tree_structure", ds_rs.get_feature_name(), ds_up.get_feature_name())
    rec.compare("feature_num_bin", "tree_structure", [ds_rs.feature_num_bin(i) for i in range(ds_rs.num_feature())],
                [ds_up.feature_num_bin(i) for i in range(ds_up.num_feature())])
    rec.finish()


def test_save_binary_ranking(tmp_path):
    """Query boundaries survive the round trip, positions do not (as upstream), also for a subset."""
    case = next(c for c in RANK_CASES if c.name == "rank_position")
    out = {}
    for mod in (lgb_rs, lgb_up):
        d = tmp_path / mod.__name__
        d.mkdir()
        ds = mod.Dataset(case.X, label=case.y, group=case.group, position=case.position, free_raw_data=False)
        ds.save_binary(d / "rank.bin")
        ds.subset(list(range(0, 200))).save_binary(d / "subset.bin")
        loaded = mod.Dataset(str(d / "rank.bin")).construct()
        sub = mod.Dataset(str(d / "subset.bin")).construct()
        bst = mod.train({**case.full_params, "verbosity": -1}, loaded, num_boost_round=5)
        out[mod.__name__] = (loaded.get_group(), loaded.get_position(), sub.get_group(), sub.num_data(),
                             bst.model_to_string())
    rs, up = out["lightgbm_rust"], out["lightgbm"]
    np.testing.assert_array_equal(rs[0], up[0])
    assert rs[1] is None and up[1] is None
    np.testing.assert_array_equal(rs[2], up[2])
    assert rs[3] == up[3] == 200
    assert rs[4] == up[4]


def test_save_binary_behaviors(tmp_path):
    """Same observable behavior as upstream for existing files, the `.bin` suffix lookup, parameter checks,
    field overrides, and prediction from a binary file; each package rejects the other's files."""
    X, y = CASES[0].X, CASES[0].y
    params = {"max_bin": 63, "min_data_in_bin": 5, "verbosity": -1}
    files = {}
    for mod in (lgb_rs, lgb_up):
        d = tmp_path / mod.__name__
        d.mkdir()
        path = d / "data.bin"
        mod.Dataset(X, label=y, params=params).save_binary(path)
        files[mod.__name__] = path
        before = path.read_bytes()
        mod.Dataset(X[:50], label=y[:50], params=params).save_binary(path)
        assert path.read_bytes() == before, "an existing file is left untouched"
        # `<name>.bin` is tried before `<name>`
        assert mod.Dataset(str(d / "data"), params=params).construct().num_data() == len(y)
        for key, value, stored in (("max_bin", 255, 63), ("min_data_in_bin", 3, 5), ("use_missing", False, 1),
                                   ("zero_as_missing", True, 0), ("bin_construct_sample_cnt", 100, 200000)):
            shown = int(value) if isinstance(value, bool) else value
            with pytest.raises(mod.basic.LightGBMError,
                               match=rf"Dataset was constructed with parameter {key}={stored}\. "
                                     rf"It cannot be changed to {shown} when loading from binary file\."):
                mod.Dataset(path, params={**params, key: value}).construct()
        new_label = (y > np.median(y)).astype(float)
        ds = mod.Dataset(path, label=new_label, weight=np.full(len(y), 2.0), params=params).construct()
        np.testing.assert_array_equal(ds.get_label(), new_label.astype(np.float32))
        np.testing.assert_array_equal(ds.get_weight(), np.full(len(y), 2.0, dtype=np.float32))
        bst = mod.train({**params, "objective": "binary"}, ds, num_boost_round=2)
        with pytest.raises(mod.basic.LightGBMError, match="Unknown format of training data"):
            bst.predict(str(path))
        with pytest.raises(mod.basic.LightGBMError, match="Cannot open data file"):
            mod.Dataset(d / "missing.bin").construct()
    with pytest.raises(lgb_rs.basic.LightGBMError, match="upstream LightGBM binary dataset file"):
        lgb_rs.Dataset(files["lightgbm"]).construct()
    with pytest.raises(lgb_up.basic.LightGBMError):
        lgb_up.Dataset(files["lightgbm_rust"]).construct()


RANK_CV_CASES = [c for c in RANK_CASES if c.name in ("rank_basic", "rank_weighted", "xendcg_basic")]


@pytest.mark.parametrize("case", RANK_CV_CASES, ids=[c.name for c in RANK_CV_CASES])
def test_ranking_cv_and_subset(case, recorder):
    """cv() group folds (GroupKFold) and Dataset.subset group recomputation."""
    rec = recorder(case.name)

    def run(mod):
        ds = mod.Dataset(case.X, label=case.y, group=case.group, weight=case.weight, free_raw_data=False)
        r = mod.cv(case.full_params, ds, num_boost_round=15, nfold=3, eval_train_metric=True, return_cvbooster=True)
        cvb = r.pop("cvbooster")
        # whole queries from the middle of the data
        bounds = np.concatenate([[0], np.cumsum(case.group)])
        idx = list(range(int(bounds[10]), int(bounds[40])))
        sub = ds.subset(idx).construct()
        sub_model = mod.train(case.full_params, sub, num_boost_round=5)
        return r, cvb, sub, sub_model

    (r_rs, b_rs, s_rs, m_rs), (r_up, b_up, s_up, m_up) = run(lgb_rs), run(lgb_up)
    rec.compare("result keys", "tree_structure", list(r_rs), list(r_up))
    for k in r_up:
        rec.compare(f"cv[{k}]", "metrics", r_rs.get(k, []), r_up[k])
    rec.compare("fold models", "model_text", b_rs.model_to_string(), b_up.model_to_string())
    rec.compare("subset get_group", "tree_structure", s_rs.get_group(), s_up.get_group())
    rec.compare("subset model", "model_text", m_rs.model_to_string(), m_up.model_to_string())
    rec.finish()


RANK_MT_CASES = [c for c in RANK_CASES if c.name in ("rank_basic", "rank_weighted", "rank_eval_at_gain", "xendcg_basic")]


@pytest.mark.parametrize("case", RANK_MT_CASES, ids=[c.name for c in RANK_MT_CASES])
def test_ranking_multithread(case, recorder):
    """num_threads=4 on both engines; NDCG reproduces OpenMP's static per-thread partial sums.

    MAP is excluded: upstream sums it with schedule(guided), whose partition is not reproducible.
    """
    rec = recorder(case.name)
    params = {**case.full_params, "num_threads": 4, "metric": ["ndcg"]}
    hist = {}
    models = []
    for mod in (lgb_rs, lgb_up):
        train, valid = _rank_datasets(mod, case)
        h = hist.setdefault(mod.__name__, {})
        models.append(mod.train(params, train, num_boost_round=case.rounds, valid_sets=[valid],
                                callbacks=[mod.record_evaluation(h)]))
    rec.compare("model_text", "model_text", models[0].model_to_string(), models[1].model_to_string())
    rec.compare("valid_0[per iteration]", "metrics", hist["lightgbm_rust"], hist["lightgbm"])
    rec.finish()


# --------------------------------------------------------------------------- prediction early stopping


PRED_ES_CASES = [c for c in CASES if c.name in ("bin_basic", "bin_init_score", "mc_basic", "ova_basic", "reg_basic",
                                                "l1_basic", "cat_binary", "mc_60_rounds")]
PRED_ES_PARAMS = [
    {"pred_early_stop": True},
    {"pred_early_stop": True, "pred_early_stop_freq": 1, "pred_early_stop_margin": 0.5},
    {"pred_early_stop": True, "pred_early_stop_freq": 3, "pred_early_stop_margin": 2.0},
    {"pred_early_stop": True, "pred_early_stop_freq": 7, "pred_early_stop_margin": 0.0},
    {"pred_early_stop": False, "pred_early_stop_freq": 1, "pred_early_stop_margin": 0.1},
]


@pytest.mark.parametrize("case", PRED_ES_CASES, ids=[c.name for c in PRED_ES_CASES])
def test_pred_early_stop(case, recorder):
    """upstream prediction_early_stop.cpp: binary / multiclass margins; no effect for regression objectives."""
    rec = recorder(case.name)
    rs, up = train_both(case, valid=False)
    stops = not np.array_equal(up.predict(case.Xv, raw_score=True, **PRED_ES_PARAMS[1]), up.predict(case.Xv, raw_score=True))
    assert stops == (case.objective in ("binary", "multiclass", "multiclassova")), "case does not exercise early stopping"
    for i, p in enumerate(PRED_ES_PARAMS):
        for kw in ({}, {"raw_score": True}, {"start_iteration": 2, "num_iteration": 11, "raw_score": True},
                   {"pred_leaf": True}):
            rec.compare(f"params[{i}] {kw}", "predictions", rs.predict(case.Xv, **kw, **p), up.predict(case.Xv, **kw, **p))
    sp = pytest.importorskip("scipy.sparse")
    p = PRED_ES_PARAMS[2]
    rec.compare("csr input", "predictions", rs.predict(sp.csr_matrix(case.Xv), **p), up.predict(sp.csr_matrix(case.Xv), **p))
    # a model loaded from text keeps its objective, so early stopping still applies
    text = up.model_to_string()
    rec.compare("loaded upstream model", "predictions", lgb_rs.Booster(model_str=text).predict(case.Xv, **p),
                lgb_up.Booster(model_str=text).predict(case.Xv, **p))
    rec.finish()


@pytest.mark.parametrize("case", [c for c in RANK_CASES if c.name in ("rank_basic", "xendcg_basic")],
                         ids=lambda c: c.name)
def test_pred_early_stop_ranking(case, recorder):
    rec = recorder(case.name)
    models = []
    for mod in (lgb_rs, lgb_up):
        train, _ = _rank_datasets(mod, case)
        models.append(mod.train(case.full_params, train, num_boost_round=case.rounds))
    assert not np.array_equal(models[1].predict(case.Xv, **PRED_ES_PARAMS[1]), models[1].predict(case.Xv))
    for i, p in enumerate(PRED_ES_PARAMS):
        rec.compare(f"params[{i}]", "predictions", models[0].predict(case.Xv, **p), models[1].predict(case.Xv, **p))
    rec.finish()


def test_pred_early_stop_errors():
    case = next(c for c in CASES if c.name == "bin_basic")
    rs, up = train_both(case, valid=False, rounds=3)
    reg = next(c for c in CASES if c.name == "reg_basic")
    rs_reg, up_reg = train_both(reg, valid=False, rounds=3)
    for b, mod in ((rs, lgb_rs), (up, lgb_up)):
        with pytest.raises(mod.basic.LightGBMError, match=r"Check failed: \(early_stop_freq\) > \(0\)"):
            b.predict(case.Xv, pred_early_stop=True, pred_early_stop_freq=0)
        with pytest.raises(mod.basic.LightGBMError, match=r"Check failed: \(early_stop_margin\) >= \(0\)"):
            b.predict(case.Xv, pred_early_stop=True, pred_early_stop_margin=-1.0)
    # objectives needing accurate predictions skip the checks entirely
    np.testing.assert_array_equal(rs_reg.predict(reg.Xv, pred_early_stop=True, pred_early_stop_freq=0),
                                  up_reg.predict(reg.Xv, pred_early_stop=True, pred_early_stop_freq=0))


# --------------------------------------------------------------------------- pandas categorical


def _pandas_frames(case):
    pd = pytest.importorskip("pandas")

    def frame(X, categories=None):
        df = pd.DataFrame({f"f{j}": X[:, j] for j in range(X.shape[1])})
        # f1: unordered strings with missing values; f3: ordered (numerical unless named); f4: unordered ints
        f1 = np.where(np.isnan(X[:, 1]) | (X[:, 1] < 0), None, np.char.add("c", np.nan_to_num(X[:, 1]).astype(int).astype(str)))
        df["f1"] = pd.Categorical(f1, categories=categories)
        df["f3"] = pd.Categorical(X[:, 3].astype(int), ordered=True)
        df["f4"] = pd.Categorical(X[:, 4].astype(int))
        return df

    train = frame(case.X)
    # validation frame: categories in a different order, one category unseen in training
    cats = list(reversed(train["f1"].cat.categories)) + ["unseen"]
    valid = frame(case.Xv, categories=cats)
    valid.loc[valid.index[:20], "f1"] = "unseen"
    return train, valid


PANDAS_CAT_CASES = [c for c in CASES if c.name in ("cat_basic", "cat_binary", "cat_multiclass")]


@pytest.mark.parametrize("categorical_feature", ["auto", ["f1", "f3", "f4"], [1, 4]])
@pytest.mark.parametrize("case", PANDAS_CAT_CASES, ids=[c.name for c in PANDAS_CAT_CASES])
def test_pandas_categorical(case, categorical_feature, recorder):
    """pandas category columns (upstream _data_from_pandas), pandas_categorical footer, validation re-coding."""
    cf_id = categorical_feature if isinstance(categorical_feature, str) else ",".join(map(str, categorical_feature))
    rec = recorder(f"{case.name}[{cf_id}]")
    train, valid = _pandas_frames(case)
    params = {k: v for k, v in case.full_params.items() if k != "categorical_feature"}
    hist = {}

    def run(mod):
        ds = mod.Dataset(train, label=case.y, categorical_feature=categorical_feature, free_raw_data=False)
        h = hist.setdefault(mod.__name__, {})
        return mod.train(params, ds, num_boost_round=case.num_boost_round,
                         valid_sets=[ds.create_valid(valid, label=case.yv)], callbacks=[mod.record_evaluation(h)])

    rs, up = run(lgb_rs), run(lgb_up)
    rec.compare("model_text", "model_text", rs.model_to_string(), up.model_to_string())
    rec.compare("pandas_categorical", "model_text", repr(rs.pandas_categorical), repr(up.pandas_categorical))
    rec.compare("valid_0[per iteration]", "metrics", hist["lightgbm_rust"], hist["lightgbm"])
    rec.compare("prediction[valid frame]", "predictions", rs.predict(valid), up.predict(valid))
    rec.compare("pred_contrib[valid frame]", "predictions", rs.predict(valid, pred_contrib=True),
                up.predict(valid, pred_contrib=True))
    reloaded = lgb_rs.Booster(model_str=up.model_to_string())
    rec.compare("rust(upstream model).pandas_categorical", "model_text", repr(reloaded.pandas_categorical),
                repr(up.pandas_categorical))
    rec.compare("rust(upstream model) prediction[valid frame]", "predictions", reloaded.predict(valid),
                up.predict(valid))
    rec.compare("dump_model", "model_text", repr(rs.dump_model()), repr(up.dump_model()))
    rec.compare("trees_to_dataframe", "model_text", rs.trees_to_dataframe().to_csv(),
                up.trees_to_dataframe().to_csv())
    rec.finish()


def test_categorical_errors_and_warnings():
    """Messages from upstream config.cpp / dataset_loader.cpp / basic.py for categorical inputs."""
    pd = pytest.importorskip("pandas")
    X = np.column_stack([np.arange(200) % 7, np.arange(200) * 0.5])
    y = np.arange(200, dtype=float)
    for mod in (lgb_rs, lgb_up):
        with pytest.raises(mod.basic.LightGBMError, match="categorical_feature is not a number"):
            mod.Dataset(X, y, params={"categorical_feature": "a", "verbose": -1}).construct()
        with pytest.raises(mod.basic.LightGBMError, match="Could not find categorical_feature x in data file"):
            mod.Dataset(X, y, params={"categorical_feature": "name:x", "verbose": -1}).construct()
        with pytest.raises(TypeError, match=r"Wrong type\(str\) or unknown name\(zz\) in categorical_feature"):
            mod.Dataset(X, y, feature_name=["a", "b"], categorical_feature=["zz"]).construct()
        df = pd.DataFrame({"a": pd.Categorical(["x", "y"] * 100), "b": X[:, 1]})
        bst = mod.train({"verbose": -1, "min_data_in_leaf": 5}, mod.Dataset(df, y), 2)
        with pytest.raises(ValueError, match="train and valid dataset categorical_feature do not match."):
            bst.predict(df.assign(c=pd.Categorical(["u"] * 200)))
    # out-of-range indices are ignored; params alias overridden by the Dataset argument (warning text from basic.py)
    out = {}
    for mod in (lgb_rs, lgb_up):
        with pytest.warns(UserWarning) as record:
            mod.Dataset(X, y, categorical_feature=[0], params={"cat_feature": "1", "verbose": -1}).construct()
        out[mod.__name__] = [str(w.message) for w in record]
        ds2 = mod.Dataset(X, y, params={"categorical_feature": "0,9", "verbose": -1}).construct()
        assert ds2.num_feature() == 2
    assert out["lightgbm_rust"] == out["lightgbm"]
    assert "cat_feature in param dict is overridden." in out["lightgbm"]


RESET_LR_CASES = [c for c in CASES if c.name in ("reg_basic", "bin_weighted", "mc_basic", "bag_basic")]


@pytest.mark.parametrize("case", RESET_LR_CASES, ids=[c.name for c in RESET_LR_CASES])
def test_reset_learning_rate(case, recorder):
    """reset_parameter(learning_rate=...) via the callback, Booster.reset_parameter with an alias, and cv."""
    rec = recorder(case.name)
    schedule = [0.3 * 0.93 ** i for i in range(case.num_boost_round)]
    rs, up = train_both(case, callbacks_factory=lambda mod: [mod.reset_parameter(learning_rate=schedule)])
    rec.compare("model_text[callback]", "model_text", rs.model_to_string(), up.model_to_string())
    out = {}
    for mod in (lgb_rs, lgb_up):
        bst = mod.Booster(case.full_params, mod.Dataset(case.X, label=case.y, weight=case.weight))
        for i in range(8):
            if i % 3:
                bst.reset_parameter({"eta": 0.123456789012345 + i / 7})
            bst.update()
        out[mod.__name__] = bst.model_to_string()
    rec.compare("model_text[Booster.reset_parameter(eta)]", "model_text", out["lightgbm_rust"], out["lightgbm"])
    res = {}
    for mod in (lgb_rs, lgb_up):
        res[mod.__name__] = mod.cv(case.full_params, mod.Dataset(case.X, label=case.y), num_boost_round=10, nfold=3,
                                   stratified=False, seed=3,
                                   callbacks=[mod.reset_parameter(learning_rate=lambda i: 0.05 + 0.01 * i)])
    for k in res["lightgbm"]:
        rec.compare(f"cv[{k}]", "metrics", res["lightgbm_rust"][k], res["lightgbm"][k])
    rec.finish()


def test_reset_parameter_errors():
    case = next(c for c in CASES if c.name == "reg_basic")
    for mod in (lgb_rs, lgb_up):
        bst = mod.Booster(case.full_params, mod.Dataset(case.X, label=case.y))
        with pytest.raises(mod.basic.LightGBMError, match=r"Check failed: \(learning_rate\) > \(0.0\)"):
            bst.reset_parameter({"learning_rate": -0.5})
        with pytest.raises(mod.basic.LightGBMError, match="Unknown token x in data file"):
            bst.reset_parameter({"learning_rate": "x"})
        with pytest.raises(mod.basic.LightGBMError, match='Parameter learning_rate should be of type double, got "1.5x"'):
            bst.reset_parameter({"learning_rate": "1.5x"})
        with pytest.raises(mod.basic.LightGBMError, match="Unknown token abc in data file"):
            mod.train({**case.full_params, "lambda_l2": "abc"}, mod.Dataset(case.X, label=case.y), num_boost_round=1)


METRIC_CASES = [
    make_case("r2_basic", "regression", {"metric": ["r2", "l2"]}),
    make_case("r2_weighted", "regression", {"metric": "r2"}, weighted=True),
    make_case("r2_poisson", "poisson", {"metric": ["r2", "poisson"]}),
    make_case("r2_discrete", "regression", {"metric": "r2", "num_leaves": 3}, kind="discrete", rounds=5),
    make_case("ap_basic", "binary", {"metric": ["average_precision", "auc"]}),
    make_case("ap_weighted", "binary", {"metric": "average_precision"}, weighted=True),
    make_case("ap_imbalanced", "binary", {"metric": "average_precision"}, imbalance=0.9),
    make_case("ap_ties", "binary", {"metric": "average_precision", "num_leaves": 3}, kind="discrete", rounds=5),
    make_case("auc_weighted_ties", "binary", {"metric": ["auc", "average_precision"], "num_leaves": 4},
              weighted=True, rounds=8),
    make_case("auc_mu_basic", "multiclass", {"num_class": 3, "metric": ["auc_mu", "multi_logloss"]}),
    make_case("auc_mu_weighted", "multiclass", {"num_class": 4, "metric": "auc_mu"}, weighted=True),
    make_case("auc_mu_matrix", "multiclass",
              {"num_class": 3, "metric": "auc_mu", "auc_mu_weights": [0, 1, 2.5, 0.5, 3, 1, 2, 1.5, 0]}),
    make_case("auc_mu_matrix_weighted", "multiclass",
              {"num_class": 3, "metric": "auc_mu", "auc_mu_weights": [0, 2, 1, 1, 0, 3, 0.25, 1, 0]}, weighted=True),
    make_case("auc_mu_ova", "multiclassova", {"num_class": 3, "metric": "auc_mu"}),
    make_case("auc_mu_ties", "multiclass", {"num_class": 3, "metric": "auc_mu", "num_leaves": 3}, kind="discrete",
              rounds=4),
]


@pytest.mark.parametrize("case", METRIC_CASES, ids=[c.name for c in METRIC_CASES])
def test_metrics_r2_ap_auc_mu(case, recorder):
    """r2 / average_precision / auc_mu per iteration on weighted train and valid sets, plus early stopping."""
    rec = recorder(case.name)
    rng = np.random.default_rng(1)
    wv = rng.uniform(0.2, 3.0, size=len(case.yv)) if case.weight is not None else None
    out = {}
    for mod in (lgb_rs, lgb_up):
        ds = mod.Dataset(case.X, label=case.y, weight=case.weight, free_raw_data=False)
        valid = ds.create_valid(case.Xv, label=case.yv, weight=wv)
        hist = {}
        bst = mod.train(case.full_params, ds, num_boost_round=case.num_boost_round, valid_sets=[ds, valid],
                        valid_names=["train", "valid"],
                        callbacks=[mod.record_evaluation(hist), mod.early_stopping(3, verbose=False)])
        out[mod.__name__] = (hist, bst.best_iteration, bst.model_to_string())
    (h_rs, it_rs, m_rs), (h_up, it_up, m_up) = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("metric_names", "tree_structure", sorted(h_rs["valid"]), sorted(h_up["valid"]))
    for data in ("train", "valid"):
        for m in h_up[data]:
            rec.compare(f"{data}.{m}[per iteration]", "metrics", h_rs[data].get(m), h_up[data][m])
    rec.compare("best_iteration", "tree_structure", it_rs, it_up)
    rec.compare("model_text", "model_text", m_rs, m_up)
    # 4 threads: upstream's ParallelSort sorts 1024+ row chunks and merges them (reproduced
    # exactly); the OpenMP reductions of the pointwise metrics combine in thread-arrival order
    params = {**case.full_params, "num_threads": 4, "force_row_wise": False, "force_col_wise": True}
    mt = {}
    for mod in (lgb_rs, lgb_up):
        ds = mod.Dataset(case.X, label=case.y, weight=case.weight)
        hist = {}
        mod.train(params, ds, num_boost_round=case.num_boost_round, valid_sets=[ds], valid_names=["train"],
                  callbacks=[mod.record_evaluation(hist)])
        mt[mod.__name__] = hist["train"]
    for m in mt["lightgbm"]:
        rec.compare(f"train.{m}[4 threads]", "multithread", mt["lightgbm_rust"].get(m), mt["lightgbm"][m])
    rec.finish()


REFIT_CASES = [c for c in CASES if c.name in ("reg_basic", "reg_weighted", "bin_weighted", "mc_basic", "ova_basic",
                                              "l1_weighted", "quantile_basic", "poisson_basic", "bag_basic",
                                              "reg_init_score", "cat_basic", "cat_binary", "extra_trees")]


@pytest.mark.parametrize("decay_rate", [0.9, 0.0, 0.37])
@pytest.mark.parametrize("case", REFIT_CASES, ids=[c.name for c in REFIT_CASES])
def test_refit(case, decay_rate, recorder):
    """Booster.refit on the holdout data (weights too when the case has them), then continued training."""
    rec = recorder(f"{case.name}[decay={decay_rate}]")
    rng = np.random.default_rng(2)
    wv = rng.uniform(0.2, 3.0, size=len(case.yv)) if case.weight is not None else None
    out = {}
    for mod in (lgb_rs, lgb_up):
        bst = mod.train(case.full_params, mod.Dataset(case.X, label=case.y, weight=case.weight,
                                                      init_score=case.init_score),
                        num_boost_round=case.num_boost_round)
        new = bst.refit(case.Xv, case.yv, decay_rate=decay_rate, weight=wv)
        text = new.model_to_string()
        pred = new.predict(case.X, raw_score=True)
        new.update()
        out[mod.__name__] = (text, pred, new.model_to_string())
    (t_rs, p_rs, c_rs), (t_up, p_up, c_up) = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("model_text[refit]", "model_text", t_rs, t_up)
    rec.compare("raw_score[refit, train rows]", "predictions", p_rs, p_up)
    rec.compare("model_text[refit + 1 iteration]", "model_text", c_rs, c_up)
    rec.finish()


def test_refit_variants(recorder):
    """path_smooth on an in-memory booster, ranking with groups, pandas input with validate_features, kwargs."""
    import pandas as pd

    rec = recorder("refit_variants")
    reg = next(c for c in CASES if c.name == "reg_regularized")
    out = {}
    for mod in (lgb_rs, lgb_up):
        # upstream reads unset parent indices for trees loaded from text, so keep the trained booster
        bst = mod.train(reg.full_params, mod.Dataset(reg.X, label=reg.y), num_boost_round=reg.num_boost_round,
                        keep_training_booster=True)
        out[mod.__name__] = bst.refit(reg.Xv, reg.yv, decay_rate=0.5).model_to_string()
    rec.compare("model_text[path_smooth refit]", "model_text", out["lightgbm_rust"], out["lightgbm"])

    rank = RANK_CASES[1]
    for mod in (lgb_rs, lgb_up):
        bst = mod.train(rank.full_params, mod.Dataset(rank.X, label=rank.y, group=rank.group, weight=rank.weight),
                        num_boost_round=rank.rounds)
        out[mod.__name__] = bst.refit(rank.Xv, rank.yv, group=rank.groupv, decay_rate=0.2).model_to_string()
    rec.compare("model_text[lambdarank refit]", "model_text", out["lightgbm_rust"], out["lightgbm"])

    case = next(c for c in CASES if c.name == "mc_basic")
    cols = [f"f{i}" for i in range(case.X.shape[1])]
    df, dfv = pd.DataFrame(case.X, columns=cols), pd.DataFrame(case.Xv, columns=cols)
    for mod in (lgb_rs, lgb_up):
        bst = mod.train(case.full_params, mod.Dataset(df, label=case.y), num_boost_round=case.num_boost_round)
        new = bst.refit(dfv, case.yv, validate_features=True, dataset_params={"max_bin": 63}, num_threads=1)
        out[mod.__name__] = new.model_to_string()
        with pytest.raises(mod.basic.LightGBMError, match="Expected 'f0' at position 0 but found 'g0'"):
            bst.refit(dfv.rename(columns={"f0": "g0"}), case.yv, validate_features=True)
    rec.compare("model_text[pandas refit, dataset_params]", "model_text", out["lightgbm_rust"], out["lightgbm"])
    rec.finish()


def test_loaded_params():
    """Booster(model_str=...).params, i.e. GBDT::GetLoadedParam typing (6-digit doubles, vectors, strings)."""
    case = next(c for c in CASES if c.name == "cat_basic")
    params = {**case.full_params, "learning_rate": 0.123456789, "lambda_l2": 1234567.0, "min_gain_to_split": 1e-9,
              "metric": ["l2", "l1"], "eval_at": [3, 1], "label_gain": [0, 1.5, 3], "max_bin": 63}
    out = {}
    for mod in (lgb_rs, lgb_up):
        bst = mod.train(params, mod.Dataset(case.X, label=case.y), num_boost_round=3)
        loaded = mod.Booster(model_str=bst.model_to_string())
        out[mod.__name__] = (loaded.params, loaded._get_loaded_param())
    assert repr(out["lightgbm_rust"]) == repr(out["lightgbm"])


def test_refit_errors():
    case = next(c for c in CASES if c.name == "reg_basic")
    for mod in (lgb_rs, lgb_up):
        def fobj(preds, data):
            return preds - data.get_label(), np.ones_like(preds)

        bst = mod.train({**case.full_params, "objective": fobj}, mod.Dataset(case.X, label=case.y),
                        num_boost_round=2)
        with pytest.raises(mod.basic.LightGBMError, match="Cannot refit due to null objective function."):
            bst.refit(case.Xv, case.yv)
        bst = mod.train({**case.full_params, "categorical_feature": [1]}, mod.Dataset(case.X, label=case.y),
                        num_boost_round=2)
        with pytest.raises(mod.basic.LightGBMError, match="'categorical_feature' value passed to Booster.refit()"):
            bst.refit(case.Xv, case.yv, categorical_feature=[2])


def test_metric_errors():
    case = next(c for c in CASES if c.name == "mc_basic")
    for mod in (lgb_rs, lgb_up):
        ds = mod.Dataset(case.X, label=case.y)
        with pytest.raises(mod.basic.LightGBMError, match="auc_mu_weights must have 9 elements, but found 4"):
            mod.train({**case.full_params, "metric": "auc_mu", "auc_mu_weights": [0, 1, 1, 0]}, ds, 1)
        with pytest.raises(mod.basic.LightGBMError,
                           match="AUC-mu matrix must have non-zero values for non-diagonal entries. "
                                 "Found zero value in position 5 of auc_mu_weights."):
            mod.train({**case.full_params, "metric": "auc_mu", "auc_mu_weights": [0, 1, 1, 1, 0, 0, 1, 1, 0]},
                      mod.Dataset(case.X, label=case.y), 1)
        with pytest.raises(mod.basic.LightGBMError, match="Multiclass objective and metrics don't match"):
            mod.train({**case.full_params, "metric": "average_precision"}, mod.Dataset(case.X, label=case.y), 1)


def test_xent_errors():
    """upstream xentropy_objective.hpp / xentropy_metric.hpp Init checks, same messages."""
    case = next(c for c in CASES if c.name == "xent_basic")
    y_bad = case.y.copy()
    y_bad[7] = 1.25
    w_neg = np.ones(len(case.y))
    w_neg[3] = -0.5
    w_zero = np.ones(len(case.y))
    w_zero[5] = 0.0
    checks = [
        ({"objective": "cross_entropy"}, y_bad, None,
         r"\[cross_entropy\]: does not tolerate element \[#7 = 1.25\] outside \[0, 1\]"),
        ({"objective": "cross_entropy_lambda"}, y_bad, None,
         r"\[cross_entropy_lambda\]: does not tolerate element \[#7 = 1.25\] outside \[0, 1\]"),
        ({"objective": "cross_entropy"}, case.y, w_neg, r"\[cross_entropy\]: at least one weight is negative"),
        ({"objective": "cross_entropy"}, case.y, np.zeros(len(case.y)), r"\[cross_entropy\]: sum of weights is zero"),
        ({"objective": "cross_entropy_lambda"}, case.y, w_zero,
         r"\[cross_entropy_lambda\]: at least one weight is non-positive"),
        ({"objective": "regression", "metric": "kullback_leibler"}, y_bad, None,
         r"\[kullback_leibler\]: does not tolerate element \[#7 = 1.25\]"),
        ({"objective": "regression", "metric": "cross_entropy"}, case.y, w_neg,
         r"\[cross_entropy:Init\]: \(metric\) weights not allowed to be negative"),
        ({"objective": "regression", "metric": "cross_entropy_lambda"}, case.y, w_zero,
         r"\[cross_entropy_lambda:Init\]: \(metric\) all weights must be positive"),
        ({"objective": "regression", "metric": "kullback_leibler"}, case.y, w_neg,
         r"\[kullback_leibler:Init\]: \(metric\) at least one weight is negative"),
        ({"objective": "regression", "metric": "cross_entropy"}, case.y, np.zeros(len(case.y)),
         r"\[Init:cross_entropy\]: sum-of-weights = 0.000000 is non-positive"),
        ({"objective": "regression", "metric": "kullback_leibler"}, case.y, np.zeros(len(case.y)),
         r"\[kullback_leibler:Init\]: sum-of-weights = 0.000000 is non-positive"),
    ]
    for params, y, w, msg in checks:
        for mod in (lgb_rs, lgb_up):
            with pytest.raises(mod.basic.LightGBMError, match=msg):
                mod.train({**DETERMINISTIC, **params}, mod.Dataset(case.X, label=y, weight=w), 1)


def test_monotone_methods_differ_and_hold():
    """The three methods give different trees, and predictions respect each constraint."""
    case = next(c for c in CASES if c.name == "mono_basic")
    texts = {}
    for method in ("basic", "intermediate", "advanced"):
        bst = lgb_rs.train({**case.full_params, "monotone_constraints_method": method},
                           lgb_rs.Dataset(case.X, label=case.y), num_boost_round=case.num_boost_round)
        texts[method] = bst.model_to_string()
        grid = np.linspace(-3, 3, 41)
        base = case.Xv[:200]
        for f, sign in enumerate(case.params["monotone_constraints"]):
            if sign == 0:
                continue
            preds = []
            for v in grid:
                X = base.copy()
                X[:, f] = v
                preds.append(bst.predict(X))
            diffs = np.diff(np.asarray(preds), axis=0) * sign
            assert diffs.min() >= 0, (method, f)
    assert len(set(texts.values())) == 3


@pytest.mark.parametrize("name", ["ic_disjoint", "ic_overlap", "ic_partial", "ic_both", "ic_categorical"])
def test_interaction_constraints_hold(name):
    """Every root-to-leaf path splits only on features of a single constraint set."""
    case = next(c for c in CASES if c.name == name)
    sets = [set(s) for s in case.params["interaction_constraints"]]
    bst = lgb_rs.train(case.full_params, lgb_rs.Dataset(case.X, label=case.y, weight=case.weight),
                       num_boost_round=case.num_boost_round)

    def paths(node, used):
        if "split_feature" not in node:
            yield used
            return
        used = used | {node["split_feature"]}
        yield from paths(node["left_child"], used)
        yield from paths(node["right_child"], used)

    n_splits = 0
    for t in bst.dump_model()["tree_info"]:
        n_splits += t["num_leaves"] - 1
        for used in paths(t["tree_structure"], frozenset()):
            assert any(used <= s for s in sets), (name, sorted(used))
    assert n_splits > 0


def test_dart_custom_objective(recorder):
    """DART with a callable objective drops trees in ``__inner_predict``.

    Upstream reads ``DART::is_update_score_cur_iter_`` before initializing it, so its first
    custom-objective iteration may or may not draw from the drop generator (it varies between
    runs). With the flag cleared, the draws are those of the built-in objective, which upstream
    computes deterministically, so that is the reference.
    """
    rec = recorder("dart_custom_objective")
    case = next(c for c in CASES if c.name == "dart_no_skip")

    def fobj(preds, data):
        return preds - data.get_label(), np.ones_like(preds)

    params = {**case.full_params, "boost_from_average": False}
    rs = lgb_rs.train({**params, "objective": fobj}, lgb_rs.Dataset(case.X, label=case.y),
                      num_boost_round=case.num_boost_round)
    up = lgb_up.train(params, lgb_up.Dataset(case.X, label=case.y), num_boost_round=case.num_boost_round)
    rec.compare("raw_score[custom vs built-in l2]", "predictions", rs.predict(case.Xv), up.predict(case.Xv))
    rec.compare("trees[custom vs built-in l2]", "model_text", rs.model_to_string().split("Tree=0")[1].split("end of trees")[0],
                up.model_to_string().split("Tree=0")[1].split("end of trees")[0])
    rec.finish()


def test_dart_training_paths(recorder):
    """Learning-rate schedule (reset_parameter reseeds the drop generator), a train-set feval
    (extra drops in __inner_predict), continued training, Booster.update with rollback."""
    rec = recorder("dart_training_paths")
    case = next(c for c in CASES if c.name == "dart_no_skip")

    def feval(preds, data):
        return "my_l2", float(np.mean((preds - data.get_label()) ** 2)), False

    out = {}
    for mod in (lgb_rs, lgb_up):
        res = {}
        tr = mod.Dataset(case.X, label=case.y)
        b = mod.train(case.full_params, tr, num_boost_round=15,
                      callbacks=[mod.reset_parameter(learning_rate=lambda i: 0.1 * 0.95 ** i)])
        res["lr_schedule"] = b.model_to_string()
        tr = mod.Dataset(case.X, label=case.y)
        hist = {}
        b = mod.train(case.full_params, tr, num_boost_round=15, valid_sets=[tr, tr.create_valid(case.Xv, label=case.yv)],
                      valid_names=["train", "valid"], feval=feval, callbacks=[mod.record_evaluation(hist)])
        res["train_feval"] = b.model_to_string()
        res["train_feval_hist"] = hist
        first = mod.train({**case.full_params, "boosting": "gbdt"}, mod.Dataset(case.X, label=case.y), num_boost_round=5)
        tr = mod.Dataset(case.X, label=case.y)
        hist = {}
        b = mod.train(case.full_params, tr, num_boost_round=10, init_model=first,
                      valid_sets=[tr.create_valid(case.Xv, label=case.yv)], callbacks=[mod.record_evaluation(hist)])
        res["continued"] = b.model_to_string()
        res["continued_hist"] = hist
        b = mod.Booster(case.full_params, mod.Dataset(case.X, label=case.y))
        for _ in range(8):
            b.update()
        b.rollback_one_iter()
        b.update()
        res["update_rollback"] = b.model_to_string()
        out[mod.__name__] = res
    rs, up = out["lightgbm_rust"], out["lightgbm"]
    for key in ("lr_schedule", "train_feval", "continued", "update_rollback"):
        rec.compare(f"model_text[{key}]", "model_text", rs[key], up[key])
    for key in ("train_feval_hist", "continued_hist"):
        for data_name, metrics in up[key].items():
            for m, vals in metrics.items():
                rec.compare(f"{key}.{data_name}.{m}", "metrics", rs[key][data_name][m], vals)
    rec.finish()


def test_dart_early_stopping_is_disabled():
    case = next(c for c in CASES if c.name == "dart_basic")
    out = {}
    for mod in (lgb_rs, lgb_up):
        tr = mod.Dataset(case.X, label=case.y)
        with pytest.warns(UserWarning, match="Early stopping is not available in dart mode"):
            b = mod.train({**case.full_params, "early_stopping_round": 1}, tr, num_boost_round=12,
                          valid_sets=[tr.create_valid(case.Xv, label=case.yv)])
        out[mod.__name__] = (b.best_iteration, b.current_iteration())
    assert out["lightgbm_rust"] == out["lightgbm"] == (0, 12)


def test_rf_training_paths(recorder):
    """Random forest: Booster.update with rollbacks (scores re-averaged), a validation set added
    mid-training (scaled by 1/iterations), refit, prediction windows and contributions."""
    rec = recorder("rf_training_paths")
    out = {}
    # upstream's thread count is process-global and reset by each Dataset construction
    one = {"num_threads": 1}
    for name in ("rf_basic", "rf_multiclass", "rf_l1"):
        case = next(c for c in CASES if c.name == name)
        for mod in (lgb_rs, lgb_up):
            res = {}
            tr = mod.Dataset(case.X, label=case.y)
            b = mod.Booster(case.full_params, tr)
            b.add_valid(tr.create_valid(case.Xv, label=case.yv, params=one), "v")
            for _ in range(6):
                b.update()
            b.rollback_one_iter()
            b.rollback_one_iter()
            b.update()
            res["update_rollback"] = b.model_to_string()
            res["update_rollback_eval"] = [b.eval_train()[0][2], b.eval_valid()[0][2]]
            b2 = mod.Booster(case.full_params, tr)
            for _ in range(4):
                b2.update()
            b2.add_valid(tr.create_valid(case.Xv, label=case.yv, params=one), "v")
            res["late_valid_eval"] = [b2.eval_valid()[0][2]]
            b2.update()
            res["late_valid_eval"].append(b2.eval_valid()[0][2])
            r = b.refit(case.Xv, case.yv)
            res["refit"] = r.model_to_string()
            res["window"] = b.predict(case.Xv, start_iteration=1, num_iteration=3)
            res["contrib"] = np.asarray(b.predict(case.Xv, pred_contrib=True))
            out[mod.__name__] = res
        rs, up = out["lightgbm_rust"], out["lightgbm"]
        for key in ("update_rollback", "refit"):
            rec.compare(f"model_text[{name}.{key}]", "model_text", rs[key], up[key])
        for key in ("update_rollback_eval", "late_valid_eval"):
            rec.compare(f"{name}.{key}", "metrics", rs[key], up[key])
        for key in ("window", "contrib"):
            rec.compare(f"{name}.{key}", "predictions", rs[key], up[key])
    rec.finish()


def test_rf_errors():
    case = next(c for c in CASES if c.name == "rf_basic")

    def fobj(preds, data):
        return preds - data.get_label(), np.ones_like(preds)

    for mod in (lgb_rs, lgb_up):
        with pytest.raises(mod.basic.LightGBMError, match=re.escape(
                "Check failed: (config->bagging_freq > 0 && config->bagging_fraction < 1.0f && "
                "config->bagging_fraction > 0.0f) || (config->feature_fraction < 1.0f && "
                "config->feature_fraction > 0.0f)")):
            mod.train({**case.full_params, "bagging_freq": 0}, mod.Dataset(case.X, label=case.y), 1)
        with pytest.raises(mod.basic.LightGBMError,
                           match=re.escape("Check failed: (train_data->metadata().init_score()) == (nullptr)")):
            mod.train(case.full_params, mod.Dataset(case.X, label=case.y, init_score=np.zeros(len(case.y))), 1)
        with pytest.raises(mod.basic.LightGBMError, match="RF mode do not support custom objective function"):
            mod.train({**case.full_params, "objective": fobj}, mod.Dataset(case.X, label=case.y), 1)


def test_monotone_errors():
    case = next(c for c in CASES if c.name == "reg_basic")
    # Upstream v4.7.0 raises this Log::Fatal inside an OpenMP loop, which aborts the process
    # ("terminate called without an active exception"), so only the message is compared.
    with pytest.raises(lgb_rs.basic.LightGBMError,
                       match="The output cannot be monotone with respect to categorical features"):
        lgb_rs.train({**case.full_params, "monotone_constraints": [0, 1, 0, 0, 0, 0]},
                     lgb_rs.Dataset(case.X, label=case.y, categorical_feature=[1]), 1)
    for mod in (lgb_rs, lgb_up):
        with pytest.raises(mod.basic.LightGBMError, match=re.escape(
                "Check failed: (static_cast<size_t>(train_data_->num_total_features())) == "
                "(config->monotone_constraints.size())")):
            mod.train({**case.full_params, "monotone_constraints": [1, -1]}, mod.Dataset(case.X, label=case.y), 1)
        with pytest.raises(mod.basic.LightGBMError,
                           match="Cannot use ``monotone_constraints`` in regression_l1 objective, please disable it."):
            mod.train({**case.full_params, "objective": "l1", "monotone_constraints": [1, 0, 0, 0, 0, 0]},
                      mod.Dataset(case.X, label=case.y), 1)


def test_feature_contri_errors_and_params():
    case = next(c for c in CASES if c.name == "contri_basic")
    for mod in (lgb_rs, lgb_up):
        with pytest.raises(mod.basic.LightGBMError, match=re.escape(
                "Check failed: (static_cast<size_t>(train_data_->num_total_features())) == "
                "(config->feature_contri.size())")):
            mod.train({**case.full_params, "feature_contri": [0.5, 0.5]}, mod.Dataset(case.X, label=case.y), 1)
    # the model text keeps upstream's %.17g join, and the alias resolves to the same parameter
    contri = [0.1, 1 / 3, 1.0, 2.5, 1e-7, 1.0]
    params = {k: v for k, v in case.full_params.items() if k != "feature_contri"}
    texts = [mod.train({**params, "feature_penalty": contri}, mod.Dataset(case.X, label=case.y), 3)
             .model_to_string() for mod in (lgb_rs, lgb_up)]
    assert texts[0] == texts[1]
    assert "[feature_contri: 0.10000000000000001,0.33333333333333331,1,2.5,9.9999999999999995e-08,1]" in texts[1]


def test_bin_params_errors_and_warnings(tmp_path, capfd):
    case = next(c for c in CASES if c.name == "bins_by_feature")
    bad_json = {"not_array.json": '{"feature": 0}', "bad.json": "[{", "range.json": '[{"feature": 6}]'}
    for name, text in bad_json.items():
        (tmp_path / name).write_text(text)
    errors = [
        ({"max_bin_by_feature": [4, 4]},
         "Check failed: (static_cast<size_t>(num_col)) == (config_.max_bin_by_feature.size())"),
        ({"max_bin_by_feature": [4, 4, 1, 4, 4, 4]},
         "Check failed: (*(std::min_element(config_.max_bin_by_feature.begin(), "
         "config_.max_bin_by_feature.end()))) > (1)"),
        ({"forcedbins_filename": str(tmp_path / "not_array.json")}, "Check failed: forced_bins_json.is_array()"),
        ({"forcedbins_filename": str(tmp_path / "bad.json")}, "Check failed: forced_bins_json.is_array()"),
        ({"forcedbins_filename": str(tmp_path / "range.json")}, "Check failed: (feature_num) < (num_total_features)"),
    ]
    for extra, msg in errors:
        for mod in (lgb_rs, lgb_up):
            with pytest.raises(mod.basic.LightGBMError, match=re.escape(msg)):
                mod.Dataset(case.X, label=case.y, params={"verbosity": -1, **extra}).construct()
    # warnings: a missing file and forced bins on a categorical feature
    missing = str(tmp_path / "missing.json")
    cat = next(c for c in CASES if c.name == "bins_forced_categorical")
    logs = []
    for mod in (lgb_rs, lgb_up):
        capfd.readouterr()
        mod.Dataset(case.X, label=case.y, params={"forcedbins_filename": missing, "verbosity": 1}).construct()
        mod.Dataset(cat.X, label=cat.y, params={"verbosity": 1, "forcedbins_filename": cat.params["forcedbins_filename"]},
                    categorical_feature=[1, 3, 4]).construct()
        out = capfd.readouterr().out
        logs.append([ln for ln in out.splitlines() if "Will ignore" in ln])
    assert logs[0] == logs[1]
    assert logs[1] == [f"[LightGBM] [Warning] Could not open {missing}. Will ignore.",
                       "[LightGBM] [Warning] Feature 1 is categorical. Will ignore forced bins for this feature.",
                       "[LightGBM] [Warning] Feature 4 is categorical. Will ignore forced bins for this feature."]


def test_bin_params_reset_and_binary(tmp_path):
    case = next(c for c in CASES if c.name == "bins_by_feature_forced")
    params = {k: v for k, v in case.full_params.items() if k in _BIN_PARAMS}
    for mod in (lgb_rs, lgb_up):
        ds = mod.Dataset(case.X, label=case.y, params={**params, "verbosity": -1}).construct()
        with pytest.raises(mod.basic.LightGBMError,
                           match=re.escape("Cannot change max_bin_by_feature after constructed Dataset handle.")):
            ds._update_params({"max_bin_by_feature": [5, 3, 4, 300, 2, 7]})
        ds._update_params({"max_bin_by_feature": [5, 3, 4, 300, 2, 6]})
        path = tmp_path / f"{mod.__name__}.bin"
        ds.save_binary(path)
        with pytest.raises(mod.basic.LightGBMError,
                           match="Parameter max_bin_by_feature cannot be changed when loading from binary file."):
            mod.Dataset(str(path), params={k: v for k, v in params.items() if k != "max_bin_by_feature"}
                        ).construct()
        # a given value replaces the stored one
        back = mod.Dataset(str(path), params={**params, "max_bin_by_feature": [9, 9, 9, 9, 9, 9],
                                              "verbosity": -1}).construct()
        assert back.num_data() == len(case.y)


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
