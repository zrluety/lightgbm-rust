# Upstream compatibility target

| Item | Value |
|---|---|
| Project | LightGBM (`https://github.com/lightgbm-org/LightGBM`, formerly `microsoft/LightGBM`) |
| Release | `v4.7.0` (published 2026-07-18) |
| Commit | `8f7036f03627054d5a54a6f965b13f4b9ff2cb63` ("release v4.7.0 (#7129)") |
| License | MIT (copy in [`NOTICE`](../NOTICE)) |
| Source location | git submodule `third_party/LightGBM` |
| Reference Python engine | PyPI `lightgbm==4.7.0` (dev dependency group only) |

## How the pin is used

- **Source inspection.** Ported algorithms cite the upstream file and function in an `upstream:` comment.
- **Parameter inventory.** `scripts/extract_params.py` parses `include/LightGBM/config.h` and writes `docs/compat/params.json`. The result was checked against the reference engine's `LGBM_DumpParamAliases` output: both list 141 parameters, and every alias set matches.
- **Test inventory.** `scripts/inventory_upstream_tests.py` lists every upstream Python test function and C++ `TEST`/`TEST_F` case. The output is `docs/compat/upstream_tests.csv`, which covers 346 Python tests and 34 C++ tests.
- **Upstream Python tests.** These run from the submodule, unmodified, through `tests/upstream_runner/run.py`.
- **Differential tests.** These import the PyPI `lightgbm==4.7.0` wheel as the reference engine. They never import it from `lightgbm_rust`.

## Updating the pin

1. Run `git -C third_party/LightGBM fetch --tags`, then `git -C third_party/LightGBM checkout <new tag>`.
2. Update the reference engine version in the `pyproject.toml` dev group, then run `uv sync`.
3. Re-run both inventory scripts and diff `docs/compat/*`.
4. Re-run the full harness (see [TESTING.md](TESTING.md)), then update [COMPATIBILITY.md](COMPATIBILITY.md).
