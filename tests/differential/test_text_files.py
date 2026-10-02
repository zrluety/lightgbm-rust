"""Differential tests for CSV/TSV/LibSVM files: Dataset construction from text files (header, column
selectors, side files, two_round, sampling) and prediction from text files, vs LightGBM 4.7.0.

upstream: src/io/parser.cpp, src/io/dataset_loader.cpp, src/io/metadata.cpp, src/application/predictor.hpp
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional

import lightgbm as lgb_up
import numpy as np
import pytest

import lightgbm_rust as lgb_rs

BASE = {"verbose": -1, "num_leaves": 7, "min_data_in_leaf": 5, "num_threads": 1}
N = 500


def _fmt(v: float) -> str:
    return repr(float(v))


def _write(path: Path, rows, header=None, sep=",") -> str:
    with open(path, "w") as f:
        if header:
            f.write(sep.join(header) + "\n")
        for r in rows:
            f.write(sep.join(r) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def files(tmp_path_factory) -> Dict[str, str]:
    d = tmp_path_factory.mktemp("text")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, 5))
    X[rng.random((N, 5)) < 0.1] = 0.0
    X[:, 3] = rng.integers(0, 6, N)
    y = X[:, 0] * 2 + X[:, 1] - X[:, 3] * 0.3 + rng.normal(scale=0.1, size=N)
    yb = (y > 0).astype(int)
    q = np.repeat(np.arange(50), 10)
    w = rng.uniform(0.5, 2, N)

    def rows(lab, X):
        return [[_fmt(lab[i])] + [_fmt(v) for v in X[i]] for i in range(len(X))]

    out = {
        "csv": _write(d / "plain.csv", rows(y, X)),
        "tsv": _write(d / "plain.tsv", rows(y, X), sep="\t"),
        "libsvm": _write(d / "d.svm", [[_fmt(y[i])] + [f"{j}:{_fmt(v)}" for j, v in enumerate(X[i]) if v != 0]
                                       for i in range(N)], sep=" "),
        "header": _write(d / "hdr.csv", [[_fmt(X[i, 0]), _fmt(y[i])] + [_fmt(v) for v in X[i, 1:]]
                                         + [_fmt(w[i]), str(q[i])] for i in range(N)],
                         header=["a", "target", "b", "c", "d", "e", "wt", "qid"]),
        "side": _write(d / "side.csv", rows(yb, X)),
        "rank": _write(d / "rank.csv", rows(rng.integers(0, 4, N), X)),
        "nan": _write(d / "nan.csv", [r[:2] + ["na", "NULL"] + r[4:] if i % 7 == 0 else r
                                      for i, r in enumerate(rows(y, X))]),
        "valid": _write(d / "valid.csv", rows(y[:200], X[:200] + 0.01)),
        "valid_header": _write(d / "valid_hdr.csv", [[_fmt(v) for v in X[i, [4, 2, 0, 1, 3]]] for i in range(200)],
                               header=["e", "c", "a", "b", "d"]),
    }
    Path(out["side"] + ".weight").write_text("\n".join(_fmt(v) for v in w) + "\n")
    Path(out["side"] + ".init").write_text("\n".join(_fmt(v) for v in rng.normal(scale=0.1, size=N)) + "\n")
    Path(out["rank"] + ".query").write_text("10\n" * 50)
    Path(out["rank"] + ".position").write_text("".join(f"{i % 10}\n" for i in range(N)))
    return out


HEADER_NAMES = {"header": True, "label_column": "name:target", "weight_column": "name:wt",
                "ignore_column": "name:qid", "categorical_feature": "name:d"}

CASES: Dict[str, Any] = {
    "csv": ("csv", {}, None),
    "tsv": ("tsv", {}, None),
    "libsvm": ("libsvm", {}, None),
    "header_names": ("header", HEADER_NAMES, None),
    "header_index": ("header", {"header": True, "label_column": 1, "ignore_column": "5,6", "weight_column": 5}, None),
    "group_column": ("header", {"header": True, "label_column": "name:d", "group_column": "name:qid",
                                "objective": "lambdarank", "ignore_column": "name:target"}, None),
    "side_files": ("side", {"objective": "binary"}, None),
    "query_position_files": ("rank", {"objective": "lambdarank"}, None),
    "two_round": ("csv", {"two_round": True, "bin_construct_sample_cnt": 60}, None),
    "small_sample": ("csv", {"bin_construct_sample_cnt": 60}, None),
    "precise_float_parser": ("csv", {"precise_float_parser": True}, None),
    "categorical_index": ("csv", {"categorical_feature": "3"}, None),
    "valid": ("csv", {"metric": "l2"}, "valid"),
    "valid_two_round": ("csv", {"metric": "l2", "two_round": True}, "valid"),
}


def _run(mod, path: str, params: Dict[str, Any], vpath: Optional[str]):
    p = {**BASE, **params}
    ds = mod.Dataset(path, params=p, free_raw_data=False)
    h: Dict[str, Any] = {}
    vs = [mod.Dataset(vpath, reference=ds)] if vpath else []
    bst = mod.train(p, ds, num_boost_round=8, valid_sets=vs, callbacks=[mod.record_evaluation(h)] if vs else [])
    has_header = bool(p.get("header"))
    preds = {kind: bst.predict(path, data_has_header=has_header, **kw)
             for kind, kw in (("normal", {}), ("raw", {"raw_score": True}), ("leaf", {"pred_leaf": True}),
                              ("contrib", {"pred_contrib": True}))}
    if vpath:
        preds["valid[num_iteration=3]"] = bst.predict(vpath, num_iteration=3)
    fields = {f: ds.get_field(f) for f in ("label", "weight", "init_score", "group", "position")}
    return bst, h, preds, fields, ds


@pytest.mark.parametrize("name", list(CASES))
def test_text_file(name, files, recorder):
    """Training and validation sets loaded from text files, and predictions from text files."""
    rec = recorder(name)
    key, params, vkey = CASES[name]
    rs = _run(lgb_rs, files[key], params, files.get(vkey))
    up = _run(lgb_up, files[key], params, files.get(vkey))
    rec.compare("model_text", "model_text", rs[0].model_to_string(), up[0].model_to_string())
    for m in up[1].get("valid_0", {}):
        rec.compare(f"valid.{m}[per iteration]", "metrics", rs[1]["valid_0"][m], up[1]["valid_0"][m])
    for kind in up[2]:
        rec.compare(f"predict(file, {kind})", "predictions", rs[2][kind], up[2][kind])
    for f, v in up[3].items():
        if v is None:
            rec.compare(f"{f} is None", "tree_structure", rs[3][f] is None, True)
        else:
            rec.compare(f, "tree_structure", rs[3][f], v)
    rec.compare("num_data, num_feature", "tree_structure", [rs[4].num_data(), rs[4].num_feature()],
                [up[4].num_data(), up[4].num_feature()])
    rec.compare("feature_name", "tree_structure", rs[4].get_feature_name(), up[4].get_feature_name())
    rec.compare("feature_num_bin", "tree_structure", [rs[4].feature_num_bin(i) for i in range(rs[4].num_feature())],
                [up[4].feature_num_bin(i) for i in range(up[4].num_feature())])
    rec.finish()


def test_text_file_matches_in_memory(files, recorder):
    """With the precise parser, a CSV file and its in-memory array give the same bins, model and predictions."""
    rec = recorder("csv_vs_in_memory")
    data = np.loadtxt(files["csv"], delimiter=",")
    params = {**BASE, "precise_float_parser": True}
    from_file = lgb_rs.train(params, lgb_rs.Dataset(files["csv"]), num_boost_round=8)
    in_memory = lgb_rs.train(params, lgb_rs.Dataset(data[:, 1:], label=data[:, 0]), num_boost_round=8)
    rec.compare("model_text", "model_text", from_file.model_to_string(), in_memory.model_to_string())
    rec.compare("predict(file) vs predict(array)", "predictions", from_file.predict(files["csv"]),
                in_memory.predict(data[:, 1:]))
    rec.finish()


def test_text_file_continued_training_and_binary(files, recorder, tmp_path):
    """init_model on a file Dataset predicts the file (data_has_header from the params); a Dataset loaded
    from text and saved as binary keeps its label column index."""
    rec = recorder("text_init_model_binary")
    out = {}
    for mod in (lgb_rs, lgb_up):
        p = {**BASE, **HEADER_NAMES}
        first = mod.train(p, mod.Dataset(files["header"], params=p), num_boost_round=3)
        cont = mod.train(p, mod.Dataset(files["header"], params=p), num_boost_round=3, init_model=first)
        d = tmp_path / mod.__name__
        d.mkdir()
        mod.Dataset(files["header"], params=p).save_binary(d / "hdr.bin")
        # upstream runs SetHeader on binary files too, so name: selectors fail there
        try:
            mod.Dataset(d / "hdr.bin", params=p).construct()
            name_err = "no error"
        except mod.basic.LightGBMError as e:
            name_err = str(e)
        by_index = {**BASE, "header": True, "label_column": "1"}
        from_bin = mod.train(by_index, mod.Dataset(d / "hdr.bin", params=by_index), num_boost_round=3)
        out[mod.__name__] = (cont.model_to_string(), from_bin.model_to_string(),
                             cont.predict(files["valid_header"], data_has_header=True), name_err)
    rs, up = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("model_text(continued)", "model_text", rs[0], up[0])
    rec.compare("model_text(binary saved from text)", "model_text", rs[1], up[1])
    rec.compare("label_index after save_binary", "tree_structure",
                re.search(r"label_index=\d+", rs[1]).group(0), re.search(r"label_index=\d+", up[1]).group(0))
    rec.compare("predict(file with header, columns permuted)", "predictions", rs[2], up[2])
    rec.compare("name: selector on a binary file", "tree_structure",
                re.sub(r"^invalid parameter: ", "", rs[3]), up[3])
    rec.finish()


def test_predict_file_shape_and_formats(files, recorder, tmp_path):
    """The feature-count check and predict_disable_shape_check, LibSVM rows with extra columns, and a CSV
    file predicted with a LibSVM-trained model."""
    rec = recorder("predict_file_shapes")
    short = tmp_path / "short.csv"
    np.savetxt(short, np.loadtxt(files["csv"], delimiter=",")[:50, :4], delimiter=",", fmt="%.17g")
    wide = tmp_path / "wide.svm"
    wide.write_text("".join(f"0 0:{i * 0.1} 3:{i % 6} 9:{i}\n" for i in range(50)))
    out = {}
    for mod in (lgb_rs, lgb_up):
        bst = mod.train(BASE, mod.Dataset(files["libsvm"]), num_boost_round=5)
        res = []
        try:
            bst.predict(str(short))
            res.append("no error")
        except mod.basic.LightGBMError as e:
            res.append(re.sub(r"^invalid data: ", "", str(e)).splitlines()[0])
        res.append(bst.predict(str(short), predict_disable_shape_check=True).tolist())
        res.append(bst.predict(str(wide), predict_disable_shape_check=True).tolist())
        res.append(bst.predict(files["csv"], pred_leaf=True).tolist())
        out[mod.__name__] = res
    rs, up = out["lightgbm_rust"], out["lightgbm"]
    rec.compare("shape check message", "tree_structure", rs[0], up[0])
    rec.compare("predict(disable_shape_check)", "predictions", rs[1], up[1])
    rec.compare("predict(libsvm, extra columns)", "predictions", rs[2], up[2])
    rec.compare("predict(csv, leaf)", "predictions", rs[3], up[3])
    rec.finish()


def _error(mod, path: str, params: Dict[str, Any]) -> str:
    p = {**BASE, "min_data_in_leaf": 1, "min_data_in_bin": 1, **params}
    try:
        mod.train(p, mod.Dataset(path, params=p), num_boost_round=2)
    except mod.basic.LightGBMError as e:
        msg = re.sub(r"^(invalid data|invalid parameter): ", "", str(e).strip()).splitlines()[0]
        # upstream CHECK messages end with the source location
        return re.sub(r" at \S+dataset_loader\.cpp, line \d+ \.$", "", msg)
    return "no error"


GOOD = "".join(f"{i % 2},{i % 5},{(i * 3) % 7}\n" for i in range(40))
HDR = "y,a,b\n" + GOOD
ERROR_CASES = {
    "label_name_missing": (HDR, {"header": True, "label_column": "name:zz"}, {}),
    "label_not_number": (HDR, {"header": True, "label_column": "abc"}, {}),
    "ignore_name_missing": (HDR, {"header": True, "ignore_column": "name:zz"}, {}),
    "weight_name_missing": (HDR, {"header": True, "weight_column": "name:zz"}, {}),
    "group_name_missing": (HDR, {"header": True, "group_column": "name:zz"}, {}),
    "categorical_name_missing": (HDR, {"header": True, "categorical_feature": "name:zz"}, {}),
    "weight_not_number": (HDR, {"weight_column": "x"}, {}),
    "name_without_header": (GOOD, {"label_column": "name:y"}, {}),
    "unknown_token": (GOOD + "1,abc,2\n", {}, {}),
    "mixed_separators": (GOOD + "1\t2\t3\n", {}, {}),
    "bad_separator": (GOOD.replace("1,0,0\n", "1,0;0\n", 1), {}, {}),
    "empty_file": ("", {}, {}),
    "blank_lines": ("\n\n\n", {}, {}),
    "one_line": ("1,2,3\n", {}, {}),
    "libsvm_label_column": ("1 0:1 2:3\n0 1:2\n" * 10, {"label_column": "1"}, {}),
    "header_column_count": ("y,a\n" + GOOD, {"header": True}, {}),
    "weight_column_range": (GOOD, {"weight_column": "5"}, {}),
    "label_column_range": (GOOD, {"label_column": "7"}, {}),
    "weight_file_length": (GOOD, {}, {".weight": "1\n" * 39}),
    "query_file_sum": (GOOD, {"objective": "lambdarank"}, {".query": "10\n" * 3}),
    "position_file_length": (GOOD, {"objective": "lambdarank"}, {".position": "1\n" * 41, ".query": "10\n" * 4}),
    "precise_bad_token": (GOOD + "1,2x,3\n", {"precise_float_parser": True}, {}),
    "precise_trailing_space": (GOOD + "1,2 ,3\n", {"precise_float_parser": True}, {}),
    "legacy_trailing_space": (GOOD + "1,2 ,3\n", {}, {}),
    "inf_and_nan_tokens": (GOOD + "1,inf,-inf\n1,nan,NULL\n", {}, {}),
    "crlf": (GOOD.replace("\n", "\r\n"), {}, {}),
    "cr_only_header": (HDR.replace("\n", "\r"), {"header": True}, {}),
}


@pytest.mark.parametrize("name", list(ERROR_CASES))
def test_text_file_errors(name, recorder, tmp_path):
    """Upstream's messages for malformed files and selectors, and the inputs both accept."""
    rec = recorder(name)
    text, params, side = ERROR_CASES[name]
    path = tmp_path / "data.csv"
    path.write_text(text, newline="")
    for suffix, content in side.items():
        Path(str(path) + suffix).write_text(content)
    rec.compare("error", "tree_structure", _error(lgb_rs, str(path), params), _error(lgb_up, str(path), params))
    rec.finish()


def test_text_file_errors_without_upstream_counterpart(tmp_path):
    """A malformed .init file (upstream terminates the process from an OpenMP region) and parser_config_file."""
    path = tmp_path / "data.csv"
    path.write_text(GOOD)
    Path(str(path) + ".init").write_text("0.1\t0.2\n" * 39 + "0.1\n")
    assert _error(lgb_rs, str(path), {}) == "Invalid initial score file. Redundant or insufficient columns"
    Path(str(path) + ".init").unlink()
    with pytest.raises(lgb_rs.basic.LightGBMError, match="not supported by lightgbm-rust yet: parameter `parser_config_file"):
        lgb_rs.Dataset(str(path), params={"parser_config_file": "parser.json"}).construct()
