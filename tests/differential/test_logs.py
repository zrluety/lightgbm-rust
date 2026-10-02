"""Differential tests of the engine log: every line a registered logger
receives at ``verbosity=2`` (Info, Warning and Debug messages of dataset
construction, text loading, objectives, metrics, bagging, the tree learner
and boosting), compared with lightgbm 4.7.0 in text and order."""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

BASE = {"verbosity": 2, "num_threads": 1, "deterministic": True, "force_row_wise": True, "seed": 7}


class _Logger:
    def __init__(self) -> None:
        self.lines: List[str] = []

    def info(self, msg: str) -> None:
        self.lines.append(f"info | {msg}")

    def warning(self, msg: str) -> None:
        self.lines.append(f"warning | {msg}")


def _data(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    X[:, 5] = rng.integers(0, 4, size=n)
    y = 2.0 * X[:, 0] - X[:, 1] + 0.5 * rng.normal(size=n)
    return X, y, rng


def _regression_bagging(lgb, tmp):
    X, y, _ = _data()
    train = lgb.Dataset(X[:500], label=y[:500], params=BASE)
    valid = lgb.Dataset(X[500:], label=y[500:], reference=train)
    params = {**BASE, "objective": "regression", "bagging_fraction": 0.7, "bagging_freq": 2, "num_leaves": 15,
              "metric": ["l2", "l1"]}
    lgb.train(params, train, num_boost_round=5, valid_sets=[valid])


def _binary_goss(lgb, tmp):
    X, y, _ = _data()
    params = {**BASE, "objective": "binary", "data_sample_strategy": "goss", "metric": ["auc", "binary_logloss"]}
    lgb.train(params, lgb.Dataset(X, label=(y > 0).astype(float)), num_boost_round=4)


def _multiclass(lgb, tmp):
    X, y, _ = _data()
    label = np.digitize(y, [-1.0, 1.0]).astype(float)
    params = {**BASE, "objective": "multiclass", "num_class": 3, "feature_fraction": 0.8,
              "metric": ["multi_logloss", "auc_mu"], "auc_mu_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    lgb.train(params, lgb.Dataset(X, label=label), num_boost_round=3)


def _multiclassova(lgb, tmp):
    X, y, _ = _data()
    label = np.digitize(y, [-1.0, 1.0]).astype(float)
    params = {**BASE, "objective": "multiclassova", "num_class": 3}
    lgb.train(params, lgb.Dataset(X, label=label), num_boost_round=2)


def _l1_without_boost_from_average(lgb, tmp):
    X, y, _ = _data()
    params = {**BASE, "objective": "regression_l1", "boost_from_average": False}
    lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=2)


def _huber_sqrt(lgb, tmp):
    X, y, _ = _data()
    params = {**BASE, "objective": "huber", "reg_sqrt": True}
    lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=2)


def _cross_entropy(lgb, tmp):
    X, y, rng = _data()
    p = 1.0 / (1.0 + np.exp(-y))
    w = rng.uniform(0.5, 2.0, size=len(y))
    params = {**BASE, "objective": "cross_entropy", "metric": ["cross_entropy", "kullback_leibler"]}
    lgb.train(params, lgb.Dataset(X, label=p, weight=w), num_boost_round=2)


def _cross_entropy_lambda(lgb, tmp):
    X, y, rng = _data()
    p = 1.0 / (1.0 + np.exp(-y))
    w = rng.uniform(0.5, 2.0, size=len(y))
    params = {**BASE, "objective": "cross_entropy_lambda", "metric": ["cross_entropy_lambda"]}
    lgb.train(params, lgb.Dataset(X, label=p, weight=w), num_boost_round=2)


def _lambdarank_positions(lgb, tmp):
    X, y, rng = _data()
    label = np.digitize(y, [-1.0, 0.0, 1.0]).astype(float)
    group = [30] * 20
    position = np.tile(np.arange(30), 20)
    params = {**BASE, "objective": "lambdarank", "metric": ["ndcg", "map"], "eval_at": [3]}
    lgb.train(params, lgb.Dataset(X, label=label, group=group, position=position), num_boost_round=2)


def _no_split(lgb, tmp):
    X, y, _ = _data()
    params = {**BASE, "objective": "regression", "min_data_in_leaf": 400}
    lgb.train(params, lgb.Dataset(X, label=y, params={"min_data_in_leaf": 400}), num_boost_round=3)


def _random_forest(lgb, tmp):
    X, y, _ = _data()
    params = {**BASE, "boosting": "rf", "bagging_fraction": 0.6, "bagging_freq": 1, "objective": "binary"}
    lgb.train(params, lgb.Dataset(X, label=(y > 0).astype(float)), num_boost_round=3)


def _custom_objective(lgb, tmp):
    X, y, _ = _data()

    def l2(preds, data):
        return preds - data.get_label(), np.ones_like(preds)

    params = {**BASE, "objective": l2}
    lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=2)


def _reset_parameter(lgb, tmp):
    X, y, _ = _data()
    params = {**BASE, "objective": "regression", "data_sample_strategy": "goss"}
    lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=3,
              callbacks=[lgb.reset_parameter(learning_rate=[0.1, 0.05, 0.02])])


def _text_file(lgb, tmp):
    X, y, rng = _data()
    w = rng.uniform(0.5, 2.0, size=len(y))
    path = tmp / "train.csv"
    rows = ["y,w," + ",".join(f"f{j}" for j in range(X.shape[1]))]
    rows += [",".join(repr(float(v)) for v in (y[i], w[i], *X[i])) for i in range(len(y))]
    path.write_text("\n".join(rows) + "\n")
    (tmp / "train.csv.init").write_text("\n".join("0.25" for _ in y) + "\n")
    params = {**BASE, "header": True, "label_column": "name:y", "weight_column": "name:w", "two_round": True,
              "objective": "regression"}
    lgb.train(params, lgb.Dataset(str(path), params=params), num_boost_round=2)


def _sparse(seed: int = 3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(800, 40)) * (rng.uniform(size=(800, 40)) < 0.06)
    y = X[:, :5].sum(axis=1) + 0.1 * rng.normal(size=800)
    return X, y


def _sparse_col_wise(lgb, tmp):
    X, y = _sparse()
    params = {**BASE, "force_row_wise": False, "force_col_wise": True, "objective": "regression"}
    lgb.train(params, lgb.Dataset(X, label=y, params=params), num_boost_round=2)


def _sparse_row_wise(lgb, tmp):
    X, y = _sparse()
    params = {**BASE, "objective": "regression"}
    lgb.train(params, lgb.Dataset(X, label=y, params=params), num_boost_round=2)


def _valid_text_two_round(lgb, tmp):
    X, y, rng = _data()
    w = rng.uniform(0.5, 2.0, size=len(y))
    for name, rows_idx in (("train.csv", range(500)), ("valid.csv", range(500, 600))):
        rows = ["y,w," + ",".join(f"f{j}" for j in range(X.shape[1]))]
        rows += [",".join(repr(float(v)) for v in (y[i], w[i], *X[i])) for i in rows_idx]
        (tmp / name).write_text("\n".join(rows) + "\n")
    (tmp / "valid.csv.weight").write_text("\n".join("1.5" for _ in range(100)) + "\n")
    params = {**BASE, "header": True, "label_column": "name:y", "weight_column": "name:w", "two_round": True,
              "objective": "regression"}
    train = lgb.Dataset("train.csv", params=params)
    valid = lgb.Dataset("valid.csv", params=params, reference=train)
    bst = lgb.train(params, train, num_boost_round=2, valid_sets=[valid])
    bst.predict("valid.csv", data_has_header=True)


def _text_ranking_unterminated(lgb, tmp):
    X, y, rng = _data()
    label = np.digitize(y, [-1.0, 0.0, 1.0])
    rows = [",".join([str(label[i])] + [repr(float(v)) for v in X[i]]) for i in range(len(y))]
    (tmp / "rank.csv").write_text("\n".join(rows))
    (tmp / "rank.csv.query").write_text("\n".join(["30"] * 20))
    (tmp / "rank.csv.weight").write_text("\n".join(repr(float(w)) for w in rng.uniform(0.5, 2.0, size=len(y))))
    params = {**BASE, "objective": "lambdarank", "two_round": True}
    train = lgb.Dataset("rank.csv", params=params).construct()
    train.save_binary("rank.bin")
    lgb.train(params, lgb.Dataset("rank.bin", params=params), num_boost_round=1)


def _save_binary(lgb, tmp):
    X, y, _ = _data()
    ds = lgb.Dataset(X, label=y, init_score=np.zeros(len(y)), params=BASE).construct()
    ds.save_binary("data.bin")
    ds.save_binary("data.bin")


SCENARIOS: Dict[str, Callable[[Any, Any], None]] = {
    "regression_bagging": _regression_bagging,
    "binary_goss": _binary_goss,
    "multiclass_auc_mu": _multiclass,
    "multiclassova": _multiclassova,
    "l1_without_boost_from_average": _l1_without_boost_from_average,
    "huber_sqrt": _huber_sqrt,
    "cross_entropy": _cross_entropy,
    "cross_entropy_lambda": _cross_entropy_lambda,
    "lambdarank_positions": _lambdarank_positions,
    "no_split": _no_split,
    "random_forest": _random_forest,
    "custom_objective": _custom_objective,
    "reset_parameter": _reset_parameter,
    "text_file": _text_file,
    "sparse_col_wise": _sparse_col_wise,
    "sparse_row_wise": _sparse_row_wise,
    "valid_text_two_round": _valid_text_two_round,
    "text_ranking_unterminated": _text_ranking_unterminated,
    "save_binary": _save_binary,
}


def _normalize(lines: List[str], tmp) -> str:
    text = "\n".join(re.sub(r"\d+\.\d+ seconds", "<elapsed> seconds", ln) for ln in lines)
    return text.replace(str(tmp), "<tmp>")


def _run(lgb, scenario, tmp, monkeypatch) -> str:
    basic = lgb.basic
    saved = (basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME)
    logger = _Logger()
    lgb.register_logger(logger)
    tmp.mkdir()
    monkeypatch.chdir(tmp)
    try:
        scenario(lgb, tmp)
        lines = list(logger.lines)
    finally:
        # back to upstream's initial engine level (Info) before restoring the logger
        lgb.Dataset(np.zeros((2, 1)), label=np.zeros(2), params={"verbosity": 1}).construct()
        basic._LOGGER, basic._INFO_METHOD_NAME, basic._WARNING_METHOD_NAME = saved
    return _normalize(lines, tmp)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_engine_log(name, recorder, tmp_path, monkeypatch):
    rec = recorder(name)
    up = _run(lgb_up, SCENARIOS[name], tmp_path / "up", monkeypatch)
    rs = _run(lgb_rs, SCENARIOS[name], tmp_path / "rs", monkeypatch)
    rec.compare("log", "log", rs, up)
    rec.finish()
