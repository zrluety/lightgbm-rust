"""Run upstream LightGBM's Python test suite against lightgbm_rust and write a report.

    uv run python tests/upstream_runner/run.py [extra pytest args]

Upstream tests run unmodified from third_party/LightGBM/tests/python_package_test
with ``lightgbm`` aliased to ``lightgbm_rust`` (see shim_plugin.py). Results go to
tests/report/upstream_results.json and tests/report/summary.md. The exit code is
0 even when upstream tests fail: failures are data for the report, not a gate.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "third_party" / "LightGBM" / "tests" / "python_package_test"


def main(argv: list[str]) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), env.get("PYTHONPATH", "")])
    cmd = [
        sys.executable, "-m", "pytest", str(SUITE),
        "-p", "shim_plugin",
        "-p", "no:cacheprovider",
        "--import-mode=importlib",
        "--continue-on-collection-errors",
        "-o", "addopts=",
        "--timeout=300",
        "-q", "--tb=no", "-rN",
        *argv,
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    subprocess.run([sys.executable, str(ROOT / "tests" / "report" / "make_summary.py")], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
