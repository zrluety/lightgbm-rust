"""pytest plugin that runs upstream LightGBM's Python tests against lightgbm_rust.

Loaded with ``-p shim_plugin`` before any test module is imported. It
aliases ``lightgbm`` (and its pure-Python submodules) to ``lightgbm_rust``,
so upstream test files run unmodified. Each test outcome is classified as:

* ``passed``      - the upstream assertions hold for lightgbm-rust;
* ``failed``      - an assertion or unexpected error: a behavioral difference
                    or a bug, to be investigated;
* ``unsupported`` - the test needs a feature lightgbm-rust reports as not
                    implemented (``LightGBMError: not supported by
                    lightgbm-rust yet``) or an API that does not exist yet
                    (missing attribute/module/keyword); this includes failures
                    whose error or preceding warning carries that message
                    (scikit-learn's model selection reports failed fits so);
* ``skipped``     - skipped by upstream's own markers/conditions (reason kept);
* ``adapted``     - listed in ``ADAPTATIONS`` below (none at present).

Results are written to ``tests/report/upstream_results.json``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

import lightgbm_rust
import lightgbm_rust.basic
import lightgbm_rust.callback
import lightgbm_rust.compat
import lightgbm_rust.engine
import lightgbm_rust.sklearn

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party" / "LightGBM"
RESULTS = ROOT / "tests" / "report" / "upstream_results.json"

# nodeid -> reason. Upstream tests are never edited; an entry here would
# describe a documented change of the test's expectations.
ADAPTATIONS: Dict[str, str] = {}


def _install_alias() -> None:
    existing = sys.modules.get("lightgbm")
    if existing is not None and existing is not lightgbm_rust:
        raise RuntimeError("upstream lightgbm was imported before the shim; results would be invalid")
    sys.modules["lightgbm"] = lightgbm_rust
    for sub in ("basic", "callback", "compat", "engine", "sklearn"):
        sys.modules[f"lightgbm.{sub}"] = getattr(lightgbm_rust, sub)


_install_alias()

_RECORDS: Dict[str, Dict[str, Any]] = {}
_COLLECTION_ERRORS: List[Dict[str, Any]] = []
_COLLECTION_SKIPS: List[Dict[str, Any]] = []
_START = time.time()

# Only errors that name a lightgbm module/class count as a missing API; any
# other AttributeError/TypeError is treated as a bug (``failed``).
_MISSING_API = [
    re.compile(r"module 'lightgbm(_rust)?(\.\w+)*' has no attribute"),
    re.compile(r"'(Booster|Dataset|CVBooster|EvalResult|Sequence|CallbackEnv)' object has no attribute"),
    re.compile(r"type object '(Booster|Dataset|CVBooster)' has no attribute"),
    re.compile(r"(Booster|Dataset|CVBooster)\.\w+\(\) got an unexpected keyword argument"),
    re.compile(r"^(train|cv|Booster|Dataset|__init__)\(\) got an unexpected keyword argument"),
    re.compile(r"No module named 'lightgbm\."),
    re.compile(r"cannot import name .* from 'lightgbm"),
]
_UNSUPPORTED_MARK = "not supported by lightgbm-rust yet"


def _classify_exception(excinfo: Any) -> tuple[str, str]:
    if excinfo is None:
        return "failed", ""
    exc = excinfo.value
    text = str(exc)
    msg = f"{type(exc).__name__}: {text}".splitlines()[0][:300] if text else type(exc).__name__
    if isinstance(exc, lightgbm_rust.LightGBMError) and _UNSUPPORTED_MARK in text:
        return "unsupported", msg
    # pytest.raises(match=...) on a feature we report as unsupported
    if isinstance(exc, AssertionError) and "Regex pattern did not match" in text and _UNSUPPORTED_MARK in text:
        inp = [ln.strip() for ln in text.splitlines() if _UNSUPPORTED_MARK in ln]
        return "unsupported", "LightGBMError: " + inp[0][:280]
    if isinstance(exc, (AttributeError, TypeError, ImportError)) and any(p.search(text) for p in _MISSING_API):
        return "unsupported", "missing API: " + msg
    # e.g. scikit-learn's model selection re-raising every failed fit's traceback in a ValueError
    marked = _marked_line(text)
    if marked:
        return "unsupported", f"{type(exc).__name__} wrapping {marked}"
    return "failed", msg


def _marked_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if _UNSUPPORTED_MARK in ln]
    return lines[0][:280] if lines else ""


# nodeid -> first warning line naming an unsupported feature (e.g. scikit-learn's FitFailedWarning)
_UNSUPPORTED_WARNINGS: Dict[str, str] = {}


def pytest_warning_recorded(warning_message: Any, when: str, nodeid: str, location: Any) -> None:
    marked = _marked_line(str(warning_message.message))
    if marked and nodeid:
        _UNSUPPORTED_WARNINGS.setdefault(nodeid, f"{warning_message.category.__name__} reporting {marked}")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo) -> Any:
    outcome = yield
    rep = outcome.get_result()
    nodeid = item.nodeid
    prev = _RECORDS.get(nodeid)
    if rep.when == "setup" and rep.passed:
        return
    if rep.when == "teardown" and (rep.passed or prev is not None and prev["category"] != "passed"):
        return
    file = str(Path(str(item.fspath)).relative_to(UPSTREAM)).replace("\\", "/")
    if rep.skipped:
        if hasattr(rep, "wasxfail"):
            category, reason = "skipped", f"xfail (upstream mark): {rep.wasxfail}"
        else:
            longrepr = rep.longrepr
            reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
            category = "skipped"
            if "not supported by lightgbm-rust yet" in reason:
                category = "unsupported"
    elif rep.failed:
        category, reason = _classify_exception(call.excinfo)
        if rep.when != "call":
            reason = f"[{rep.when}] {reason}"
    else:
        category, reason = "passed", ""
    if nodeid in ADAPTATIONS and category == "passed":
        category, reason = "adapted", ADAPTATIONS[nodeid]
    _RECORDS[nodeid] = {
        "nodeid": nodeid,
        "file": file,
        "test": item.originalname if hasattr(item, "originalname") else item.name,
        "category": category,
        "reason": reason,
        "duration": round(getattr(rep, "duration", 0.0), 3),
    }


def pytest_collectreport(report: pytest.CollectReport) -> None:
    if report.failed:
        text = str(report.longrepr)
        last = [ln for ln in text.splitlines() if ln.startswith("E ")]
        reason = (last[-1][2:].strip() if last else text.splitlines()[-1])[:300]
        _COLLECTION_ERRORS.append({"nodeid": report.nodeid, "reason": reason})
    elif report.skipped and report.nodeid.endswith(".py"):
        longrepr = report.longrepr
        reason = longrepr[2] if isinstance(longrepr, tuple) else str(longrepr)
        _COLLECTION_SKIPS.append({"nodeid": report.nodeid, "reason": reason[:300]})


def _inventory_counts() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    path = ROOT / "docs" / "compat" / "upstream_tests.csv"
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["suite"] == "python":
                counts[row["file"]] = counts.get(row["file"], 0) + 1
    return counts


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    import lightgbm  # the alias

    assert lightgbm is lightgbm_rust, "alias was replaced during the run"
    # a failure preceded by a warning that a fit hit an unsupported feature is attributed to that feature
    for nodeid, marked in _UNSUPPORTED_WARNINGS.items():
        rec = _RECORDS.get(nodeid)
        if rec is not None and rec["category"] == "failed":
            rec["category"] = "unsupported"
            rec["reason"] = f"{marked} (test then failed: {rec['reason'][:120]})"
    inventory = _inventory_counts()
    for err in _COLLECTION_ERRORS + _COLLECTION_SKIPS:
        err["nodeid"] = err["nodeid"].split("third_party/LightGBM/")[-1]
        err["upstream_test_functions"] = inventory.get(err["nodeid"], None)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "upstream_version": lightgbm_rust.UPSTREAM_VERSION,
        "upstream_commit": lightgbm_rust.UPSTREAM_COMMIT,
        "engine": "lightgbm_rust " + lightgbm_rust.__version__,
        "wall_seconds": round(time.time() - _START, 1),
        "adaptations": ADAPTATIONS,
        "records": sorted(_RECORDS.values(), key=lambda r: r["nodeid"]),
        "collection_errors": _COLLECTION_ERRORS,
        "collection_skips": _COLLECTION_SKIPS,
    }, indent=1))
