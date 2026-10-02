"""Differential tests of the plotting module and
``Booster.get_split_value_histogram``: both packages plot models trained on the
same data, and the histograms, the graphviz source of the tree digraphs, and
the matplotlib artists (bars, lines, texts, labels, limits, rendered tree image)
are compared with lightgbm 4.7.0."""

from __future__ import annotations

import shutil
from typing import Any, Dict, List

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

graphviz = pytest.importorskip("graphviz")

BASE = {"verbosity": -1, "num_threads": 1, "deterministic": True, "force_row_wise": True, "seed": 3}


def _data(n: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 2][rng.uniform(size=n) < 0.15] = np.nan
    X[:, 4] = 0.0
    X[:, 5] = rng.integers(0, 6, size=n)
    y = 2.0 * X[:, 0] - X[:, 1] + np.nan_to_num(X[:, 2]) + (X[:, 5] == 2) + 0.3 * rng.normal(size=n)
    return X, y


def _regression(lgb):
    X, y = _data()
    names = [f"feat_{i}" for i in range(X.shape[1])]
    train = lgb.Dataset(X[:600], label=y[:600], feature_name=names, categorical_feature=[5], params=BASE)
    valid = lgb.Dataset(X[600:], label=y[600:], reference=train)
    evals: Dict[str, Any] = {}
    params = {**BASE, "objective": "regression", "num_leaves": 7, "metric": ["l2", "l1"]}
    bst = lgb.train(params, train, num_boost_round=6, valid_sets=[train, valid], valid_names=["train", "valid"],
                    callbacks=[lgb.record_evaluation(evals)])
    return bst, evals, X


def _multiclass(lgb):
    X, y = _data(seed=1)
    label = np.digitize(y, [-1.0, 1.0]).astype(float)
    params = {**BASE, "objective": "multiclass", "num_class": 3, "num_leaves": 5}
    return lgb.train(params, lgb.Dataset(X, label=label, categorical_feature=[5]), num_boost_round=2), X


MODELS = {"regression": _regression, "multiclass": _multiclass}
_CACHE: Dict[Any, Any] = {}


def _model(lgb, name):
    key = (lgb.__name__, name)
    if key not in _CACHE:
        _CACHE[key] = MODELS[name](lgb)
    return _CACHE[key]


def _axes(ax) -> str:
    legend = ax.get_legend()
    desc = {
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "xticks": ax.get_xticks().tolist(),
        "yticks": ax.get_yticks().tolist(),
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
        "texts": [(t.get_text(), t.get_position(), t.get_va()) for t in ax.texts],
        "patches": [(p.get_x(), p.get_y(), p.get_width(), p.get_height(), p.get_facecolor()) for p in ax.patches],
        "lines": [(ln.get_label(), list(ln.get_xdata()), list(ln.get_ydata())) for ln in ax.lines],
        "legend": None if legend is None else [t.get_text() for t in legend.get_texts()],
        "grid": [ln.get_visible() for ln in ax.get_xgridlines() + ax.get_ygridlines()],
        "images": [np.asarray(im.get_array()).tobytes().hex() for im in ax.images],
    }
    plt.close(ax.figure)
    return repr(desc)


def _outcome(fn) -> Any:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


HISTOGRAMS: List[Dict[str, Any]] = [
    {"feature": 0},
    {"feature": "feat_1", "bins": 4},
    {"feature": 2, "bins": [-3.0, -1.0, 0.0, 1.0, 3.0]},
    {"feature": "feat_0", "xgboost_style": True},
    {"feature": 1, "bins": 3, "xgboost_style": True},
    {"feature": 4},
    {"feature": 4, "xgboost_style": True},
    {"feature": 5},
]


@pytest.mark.parametrize("i", range(len(HISTOGRAMS)))
def test_split_value_histogram(i, recorder):
    kw = HISTOGRAMS[i]
    rec = recorder(f"hist_{i}")
    out = {}
    for lgb in (lgb_up, lgb_rs):
        bst = _model(lgb, "regression")[0]

        def run():
            res = bst.get_split_value_histogram(**kw)
            if isinstance(res, tuple):
                return [np.asarray(a).tolist() for a in res]
            return (type(res).__name__, np.asarray(res).tolist())

        out[lgb] = repr(_outcome(run))
    rec.compare("histogram", "plotting", out[lgb_rs], out[lgb_up])
    rec.finish()


DIGRAPHS: List[Dict[str, Any]] = [
    {"model": "regression", "tree_index": 0},
    {"model": "regression", "tree_index": 3, "precision": 2, "orientation": "vertical",
     "show_info": ["split_gain", "internal_value", "internal_count", "internal_weight", "leaf_count",
                   "leaf_weight", "data_percentage"]},
    {"model": "regression", "tree_index": 5, "example_case": 1, "show_info": ["leaf_count"]},
    {"model": "regression", "tree_index": 2, "example_case": 7, "max_category_values": 2,
     "name": "t2", "comment": "c", "graph_attr": {"color": "red"}},
    {"model": "multiclass", "tree_index": 4, "show_info": ["internal_value", "leaf_weight"], "precision": None},
    {"model": "multiclass", "tree_index": 1, "example_case": 3},
]


def _digraph_kwargs(spec, X):
    kw = {k: v for k, v in spec.items() if k != "model"}
    if "example_case" in kw:
        kw["example_case"] = X[kw["example_case"] : kw["example_case"] + 1]
    return kw


@pytest.mark.parametrize("i", range(len(DIGRAPHS)))
def test_create_tree_digraph(i, recorder):
    spec = DIGRAPHS[i]
    rec = recorder(f"digraph_{i}")
    out = {}
    for lgb in (lgb_up, lgb_rs):
        bst, X = (lambda m: (m[0], m[-1]))(_model(lgb, spec["model"]))
        out[lgb] = _outcome(lambda: lgb.create_tree_digraph(bst, **_digraph_kwargs(spec, X)).source)
    rec.compare("digraph_source", "plotting", out[lgb_rs], out[lgb_up])
    rec.finish()


PLOTS: List[Dict[str, Any]] = [
    {"fn": "plot_importance"},
    {"fn": "plot_importance", "importance_type": "gain", "precision": 2, "max_num_features": 3,
     "title": "gain", "xlabel": "x", "ylabel": "y", "height": 0.5, "grid": False, "color": "r"},
    {"fn": "plot_importance", "ignore_zero": False, "xlim": (0, 40), "ylim": (-1, 7), "figsize": (4, 3)},
    {"fn": "plot_split_value_histogram", "feature": 0},
    {"fn": "plot_split_value_histogram", "feature": "feat_1", "bins": 3, "width_coef": 0.5, "title": "f @feature@ @index/name@"},
    {"fn": "plot_split_value_histogram", "feature": 4},
    {"fn": "plot_metric", "source": "evals"},
    {"fn": "plot_metric", "source": "evals", "metric": "l1", "dataset_names": ["valid"], "grid": False},
    {"fn": "plot_metric", "source": "booster_evals", "title": "@metric@"},
]


@pytest.mark.parametrize("i", range(len(PLOTS)))
def test_plot_functions(i, recorder):
    spec = dict(PLOTS[i])
    rec = recorder(f"plot_{i}")
    fn = spec.pop("fn")
    source = spec.pop("source", "booster")
    out = {}
    for lgb in (lgb_up, lgb_rs):
        bst, evals, _ = _model(lgb, "regression")
        if source == "evals":
            arg = evals
        elif source == "booster_evals":
            arg = lgb.LGBMRegressor(**{**BASE, "n_estimators": 4, "num_leaves": 5}).fit(
                _data()[0], _data()[1], eval_set=[(_data()[0][:200], _data()[1][:200])], eval_metric="l1")
        else:
            arg = bst
        out[lgb] = _outcome(lambda: _axes(getattr(lgb, fn)(arg, **spec)))
    rec.compare("axes", "plotting", out[lgb_rs], out[lgb_up])
    rec.finish()


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz `dot` is needed to render trees")
@pytest.mark.parametrize("model,tree_index", [("regression", 1), ("multiclass", 5)])
def test_plot_tree(model, tree_index, recorder):
    rec = recorder(f"plot_tree_{model}")
    out = {}
    for lgb in (lgb_up, lgb_rs):
        bst = _model(lgb, model)[0]
        out[lgb] = _outcome(lambda: _axes(lgb.plot_tree(bst, tree_index=tree_index, show_info=["leaf_count"],
                                                        figsize=(6, 4), dpi=60)))
    rec.compare("axes", "plotting", out[lgb_rs], out[lgb_up])
    rec.finish()


def test_plotting_errors(recorder):
    rec = recorder("plotting_errors")
    calls = [
        lambda lgb, b: lgb.plot_importance(b, importance_type="bad"),
        lambda lgb, b: lgb.plot_importance(b, xlim=(1, 2, 3)),
        lambda lgb, b: lgb.plot_importance("not a booster"),
        lambda lgb, b: lgb.plot_split_value_histogram(b, feature=5),
        lambda lgb, b: lgb.plot_metric(b),
        lambda lgb, b: lgb.plot_metric({"a": {"l2": [1.0]}}, metric="l1"),
        lambda lgb, b: lgb.create_tree_digraph(b, tree_index=99),
        lambda lgb, b: lgb.create_tree_digraph(b, example_case=np.zeros((2, 6))),
        lambda lgb, b: b.get_split_value_histogram("missing_name"),
    ]
    for j, call in enumerate(calls):
        out = {}
        for lgb in (lgb_up, lgb_rs):
            bst = _model(lgb, "regression")[0]
            res = _outcome(lambda: call(lgb, bst))
            out[lgb] = res if isinstance(res, str) else "returned"
            plt.close("all")
        rec.compare(f"error_{j}", "plotting", out[lgb_rs], out[lgb_up])
    rec.finish()
