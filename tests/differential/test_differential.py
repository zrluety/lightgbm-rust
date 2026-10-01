"""Differential tests: lightgbm-rust vs LightGBM 4.7.0 on the regression-family/binary/multiclass subset.

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
    REGRESSION_OBJECTIVES,
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
                                           "goss_no_subset", "sampling_combo")]


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
                                           "bin_init_score")]


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
                                             "goss_basic", "bin_init_score", "ova_basic")]


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
