"""Build tests/report/summary.md from the machine-readable test results.

Inputs (each optional; missing inputs are reported as "not run"):
  tests/report/cargo_test.txt        output of `cargo test -p lgbm-core`
  tests/report/differential.json     written by tests/differential/conftest.py
  tests/report/upstream_results.json written by tests/upstream_runner/shim_plugin.py
  tests/report/bench.json            written by benches/bench_vs_upstream.py
  docs/compat/upstream_tests.csv     upstream test inventory
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "tests" / "report"
CATEGORIES = ["passed", "failed", "unsupported", "skipped", "adapted"]


def _load(name: str):
    p = REPORT / name
    return json.loads(p.read_text()) if p.exists() else None


def rust_section() -> list[str]:
    out = ["## Rust tests (`cargo test -p lgbm-core`)", ""]
    p = REPORT / "cargo_test.txt"
    if not p.exists():
        return out + ["Not run (no `tests/report/cargo_test.txt`).", ""]
    passed = failed = ignored = 0
    for m in re.finditer(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored", p.read_text()):
        passed += int(m[1])
        failed += int(m[2])
        ignored += int(m[3])
    return out + [f"- passed: {passed}", f"- failed: {failed}", f"- ignored: {ignored}", ""]


def cpp_section() -> list[str]:
    ported: dict[str, str] = {}
    pat = re.compile(r"// upstream: (tests/cpp_tests/\S+) TEST(?:_F)?\((\w+), (\w+)\)")
    for src in (ROOT / "crates").rglob("*.rs"):
        for m in pat.finditer(src.read_text(encoding="utf-8")):
            ported[f"{m[1]}::{m[2]}.{m[3]}"] = str(src.relative_to(ROOT)).replace("\\", "/")
    rows = []
    with open(ROOT / "docs" / "compat" / "upstream_tests.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["suite"] == "cpp"]
    n_ported = sum(1 for r in rows if r["file"] + "::" + r["test"] in ported)
    out = ["## Upstream C++ tests (ported to Rust)", "",
           f"{n_ported} of {len(rows)} upstream C++ tests are ported "
           "(they run as part of `cargo test`; traceability via `// upstream:` comments).", "",
           "| upstream test | status | location / reason |", "|---|---|---|"]
    for r in rows:
        key = f"{r['file']}::{r['test']}"
        if key in ported:
            out.append(f"| `{key}` | ported | `{ported[key]}` |")
        else:
            reason = {
                "c-api-internals": "tests a C++/C-API internal with no counterpart in lightgbm-rust",
                "input-arrow": "tests the deprecated chunk-array C API (`LGBM_*FromArrow`), which has no counterpart",
                "input-sparse": "tests the streaming push C API (`LGBM_DatasetPushRowsByCSR*`), which has no "
                                "counterpart; CSR/CSC input is covered by the differential `test_sparse_inputs`",
            }.get(r["area"], f"area `{r['area']}` not implemented yet")
            out.append(f"| `{key}` | not ported | {reason} |")
    return out + [""]


def differential_section() -> list[str]:
    data = _load("differential.json")
    out = ["## Differential tests vs LightGBM 4.7.0 (`tests/differential`)", ""]
    if data is None:
        return out + ["Not run.", ""]
    recs = data["records"]
    by_tol: dict[str, list] = defaultdict(list)
    for r in recs:
        by_tol[r["tolerance"]].append(r)
    cases = sorted({r["case"] for r in recs})
    out += [f"{len(recs)} comparisons over {len(cases)} cases; tolerances from `tests/tolerances.toml`.", "",
            "| quantity class | comparisons | within tolerance | bitwise exact | max abs diff | max rel diff |",
            "|---|---|---|---|---|---|"]
    for tol, rs in sorted(by_tol.items()):
        ok = sum(r["passed"] for r in rs)
        ex = sum(r["exact"] for r in rs)
        mabs = max(r["max_abs"] for r in rs)
        mrel = max(r["max_rel"] for r in rs)
        out.append(f"| {tol} | {len(rs)} | {ok} | {ex} | {mabs:.3g} | {mrel:.3g} |")
    bad = [r for r in recs if not r["passed"]]
    out += ["", f"Cases: {', '.join(cases)}.", ""]
    if bad:
        out += ["Out-of-tolerance comparisons:", ""]
        out += [f"- `{r['case']}` {r['quantity']}: max_abs={r['max_abs']:.3g}, max_rel={r['max_rel']:.3g} {r['note']}"
                for r in bad[:50]]
        out.append("")
    notes = sorted({r["note"] for r in recs if r["note"] and r["passed"]})
    if notes:
        out += ["Notes: " + "; ".join(notes), ""]
    return out


def upstream_section() -> list[str]:
    data = _load("upstream_results.json")
    out = ["## Upstream Python test suite (`tests/upstream_runner`)", ""]
    if data is None:
        return out + ["Not run.", ""]
    recs = data["records"]
    total = Counter(r["category"] for r in recs)
    out += [f"Upstream `tests/python_package_test` at {data['upstream_commit'][:7]} (v{data['upstream_version']}), "
            f"run unmodified with `lightgbm` aliased to `lightgbm_rust` ({data['wall_seconds']} s).", "",
            "| category | tests |", "|---|---|"]
    for c in CATEGORIES:
        out.append(f"| {c} | {total.get(c, 0)} |")
    out.append(f"| **total collected** | {len(recs)} |")
    errs = data["collection_errors"]
    mod_skips = data.get("collection_skips", [])
    if errs:
        n_fn = sum(e.get("upstream_test_functions") or 0 for e in errs)
        out.append(f"| not collected (module import failed) | {len(errs)} modules, {n_fn} test functions |")
    if mod_skips:
        n_fn = sum(e.get("upstream_test_functions") or 0 for e in mod_skips)
        out.append(f"| module skipped by upstream (`importorskip` etc.) | {len(mod_skips)} modules, {n_fn} test functions |")
    out += ["", "Adaptations: " + ("none (no upstream test is modified)" if not data["adaptations"] else
                                   str(len(data["adaptations"]))), ""]

    per_file: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        per_file[r["file"]][r["category"]] += 1
    out += ["### By file", "", "| file | " + " | ".join(CATEGORIES) + " |", "|---|" + "---|" * len(CATEGORIES)]
    for f in sorted(per_file):
        out.append(f"| `{f}` | " + " | ".join(str(per_file[f].get(c, 0)) for c in CATEGORIES) + " |")
    if errs:
        out += ["", "### Modules that could not be imported", ""]
        out += [f"- `{e['nodeid']}` ({e.get('upstream_test_functions')} test functions): {e['reason']}" for e in errs]
    if mod_skips:
        out += ["", "### Modules skipped as a whole by upstream conditions", ""]
        out += [f"- `{e['nodeid']}` ({e.get('upstream_test_functions')} test functions): {e['reason']}" for e in mod_skips]

    def grouped(cat: str, limit: int) -> list[str]:
        reasons = Counter()
        for r in recs:
            if r["category"] == cat:
                key = re.sub(r"0x[0-9a-f]+", "0x..", r["reason"])[:200]
                reasons[key] += 1
        return [f"- {n} x {k or '(no message)'}" for k, n in reasons.most_common(limit)]

    out += ["", "### By feature area (areas from `docs/compat/upstream_tests.csv`)", "",
            "Counts are test *cases* (parametrizations), except the last two columns, which count test functions.", "",
            "| area | " + " | ".join(CATEGORIES) + " | not collected | module skipped |",
            "|---|" + "---|" * (len(CATEGORIES) + 2)]
    area_of: dict[tuple[str, str], str] = {}
    with open(ROOT / "docs" / "compat" / "upstream_tests.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["suite"] == "python":
                area_of[(row["file"], row["test"])] = row["area"]
    per_area: dict[str, Counter] = defaultdict(Counter)
    for r in recs:
        per_area[area_of.get((r["file"], r["test"]), "unmapped")][r["category"]] += 1
    uncollected = {e["nodeid"] for e in errs}
    skipped_mods = {e["nodeid"] for e in mod_skips}
    for (file, _test), area in area_of.items():
        if file in uncollected:
            per_area[area]["not collected"] += 1
        elif file in skipped_mods:
            per_area[area]["module skipped"] += 1
    for a in sorted(per_area):
        cols = CATEGORIES + ["not collected", "module skipped"]
        out.append(f"| `{a}` | " + " | ".join(str(per_area[a].get(c, 0)) for c in cols) + " |")

    out += ["", "### Unsupported: most common reasons", ""] + grouped("unsupported", 40)
    out += ["", "### Skipped: reasons", ""] + grouped("skipped", 20)
    failed = [r for r in recs if r["category"] == "failed"]
    out += ["", f"### Failed ({len(failed)})", "",
            "Each failure is a behavioral difference or a bug to investigate; none is suppressed.", ""]
    out += [f"- `{r['nodeid'].split('python_package_test/')[-1]}`: {r['reason'] or '(assertion)'}" for r in failed]
    return out + [""]


def bench_section() -> list[str]:
    data = _load("bench.json")
    out = ["## Benchmark (`benches/bench_vs_upstream.py`)", ""]
    if data is None:
        return out + ["Not run.", ""]
    out += [f"{data['rows']:,} rows, {data['iterations']} iterations, best of {data['repeats']} runs; "
            f"host: {data['host']}.", "", "Tasks:", ""]
    for name, t in data["tasks"].items():
        extra = ", ".join(f"{k}={v}" for k, v in t["params"].items())
        shape = f"{t.get('rows', data['rows']):,} rows, {t['cols']} features"
        if t.get("density") is not None:
            shape += f", scipy CSR with density {t['density']}"
        out.append(f"- `{name}`: objective `{t['objective']}`, {shape}" + (f", {extra}" if extra else ""))
    out += ["", "Times in seconds (upstream / rust); ratio = rust / upstream train time. Memory is the peak RSS "
            "above the RSS after generating the input, in MB (upstream / rust).", "",
            "| task | threads | construct | train | ratio | predict | memory | max abs prediction diff |",
            "|---|---|---|---|---|---|---|---|"]
    pairs: dict[tuple[str, int], dict[str, dict]] = {}
    for r in data["results"]:
        pairs.setdefault((r["task"], r["threads"]), {})["rs" if "rust" in r["engine"] else "up"] = r

    def mem(r: dict) -> str:
        return f"{r['peak_rss_mb'] - r['data_rss_mb']:.0f}" if "peak_rss_mb" in r else "–"

    for (task, t), d in pairs.items():
        u, s = d["up"], d["rs"]
        out.append(f"| {task} | {t} | {u['construct_s']:.2f} / {s['construct_s']:.2f} | "
                   f"{u['train_s']:.2f} / {s['train_s']:.2f} | {s['train_s'] / u['train_s']:.2f} | "
                   f"{u['predict_s']:.2f} / {s['predict_s']:.2f} | {mem(u)} / {mem(s)} | "
                   f"{s['max_abs_prediction_diff']:.1e} |")
    return out + ["", data.get("note", ""), ""]


def main() -> None:
    lines = ["# lightgbm-rust test report", "",
             "Generated by `tests/report/make_summary.py`. Do not edit by hand.", ""]
    for section in (rust_section, cpp_section, differential_section, upstream_section, bench_section):
        lines += section()
    (REPORT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT / 'summary.md'}")


if __name__ == "__main__":
    main()
