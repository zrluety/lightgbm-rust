"""Inventory upstream Python and C++ tests at the pinned commit.

Usage:
    uv run python scripts/inventory_upstream_tests.py

Writes docs/compat/upstream_tests.csv with one row per test function and an
initial feature-area mapping. The disposition column is a planning label, not
a result; results come from tests/upstream_runner/run.py.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "third_party" / "LightGBM" / "tests"
OUT = ROOT / "docs" / "compat" / "upstream_tests.csv"

# (regex on "file::test", area). First match wins.
AREA_RULES = [
    (r"dask", "distributed-dask"),
    (r"distributed", "distributed"),
    (r"cuda|gpu", "gpu"),
    (r"arrow", "input-arrow"),
    (r"polars", "input-polars"),
    (r"plot", "plotting"),
    (r"pandas|categor", "categorical-pandas"),
    (r"sparse|csr|csc|scipy", "input-sparse"),
    (r"sklearn", "sklearn"),
    (r"cv\b|_cv_|test_cv", "cv"),
    (r"callback|early_stop", "callbacks-early-stopping"),
    (r"save|load|dump|model_to|serializ|pickle|string|json", "persistence"),
    (r"refit|continue|init_model|init_score|keep_training", "continued-training"),
    (r"contrib|shap|leaf|pred", "prediction"),
    (r"importance", "feature-importance"),
    (r"monoton|interaction|forced|constraint", "constraints"),
    (r"dart|goss|rf\b|random_forest|bagging", "boosting-sampling"),
    (r"lambdarank|rank|xendcg|ndcg|map\b", "ranking"),
    (r"multiclass|softmax|ova", "multiclass"),
    (r"metric|auc|eval", "metrics"),
    (r"linear_tree|linear", "linear-tree"),
    (r"quantiz", "quantized-gradients"),
    (r"param|alias|config", "parameters"),
    (r"dataset|bin|construct|missing|zero|nan|weight", "dataset-binning"),
    (r"regression|binary|l1|l2|huber|fair|poisson|quantile|mape|gamma|tweedie|xentropy|cross_entropy", "objectives"),
    (r"consistency", "cli-consistency"),
    (r"single_row|array_args|chunked|byte_buffer|stream|common", "c-api-internals"),
]

UNSUPPORTED_AREAS = {"distributed-dask", "distributed", "gpu"}
DEFERRED_AREAS = {"input-arrow", "input-polars", "plotting", "ranking", "multiclass",
                  "linear-tree", "quantized-gradients", "cli-consistency", "c-api-internals",
                  "boosting-sampling", "constraints", "categorical-pandas", "input-sparse",
                  "sklearn", "cv"}


def area_for(key: str) -> str:
    low = key.lower()
    for pat, area in AREA_RULES:
        if re.search(pat, low):
            return area
    return "general"


def disposition(area: str) -> str:
    if area in UNSUPPORTED_AREAS:
        return "out-of-scope-m1-3"
    if area in DEFERRED_AREAS:
        return "deferred"
    return "target-m2-m4"


def python_tests() -> list[dict]:
    rows = []
    for path in sorted((UP / "python_package_test").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                params = [
                    ast.unparse(d) for d in node.decorator_list
                    if "parametrize" in ast.unparse(d) or "skip" in ast.unparse(d)
                ]
                key = f"{path.stem}::{node.name}"
                area = area_for(key)
                rows.append({
                    "suite": "python",
                    "file": f"tests/python_package_test/{path.name}",
                    "test": node.name,
                    "line": node.lineno,
                    "area": area,
                    "disposition": disposition(area),
                    "upstream_markers": " | ".join(p.replace("\n", " ")[:120] for p in params),
                })
    return rows


CPP_TEST_RE = re.compile(r"^TEST(?:_F)?\((\w+),\s*(\w+)\)", re.M)


def cpp_tests() -> list[dict]:
    rows = []
    for path in sorted((UP / "cpp_tests").glob("test_*.cpp")):
        text = path.read_text(encoding="utf-8")
        for m in CPP_TEST_RE.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            key = f"{path.stem}::{m.group(1)}.{m.group(2)}"
            area = area_for(key)
            rows.append({
                "suite": "cpp",
                "file": f"tests/cpp_tests/{path.name}",
                "test": f"{m.group(1)}.{m.group(2)}",
                "line": line,
                "area": area,
                "disposition": disposition(area),
                "upstream_markers": "",
            })
    return rows


def main() -> None:
    rows = python_tests() + cpp_tests()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    by_suite: dict[str, int] = {}
    for r in rows:
        by_suite[r["suite"]] = by_suite.get(r["suite"], 0) + 1
    print(f"wrote {len(rows)} tests to {OUT.relative_to(ROOT)}: {by_suite}")


if __name__ == "__main__":
    main()
