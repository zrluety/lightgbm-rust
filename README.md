# lightgbm-rust

A pure-Rust implementation of [LightGBM](https://github.com/lightgbm-org/LightGBM) gradient boosting, with a LightGBM-like Python API.

**Status: pre-alpha.** The implemented subset is CPU-only. It covers:
- GBDT, DART and random forest
- the regression, binary, multiclass and ranking objectives
- numerical and categorical features
- monotone and interaction constraints
- `refit`, `cv` and the scikit-learn estimators
- input from arrays, pandas, pyarrow, polars, scipy.sparse, and binary or CSV/TSV/LibSVM files

Within that subset, on every single-threaded differential test case run so far, results are bitwise identical to upstream LightGBM 4.7.0:
- bins
- gradients
- trees
- predictions
- metrics
- model text

Everything else raises `LightGBMError("not supported by lightgbm-rust yet: ...")`. lightgbm-rust does **not** have full parity with LightGBM. See [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the per-feature status.

The engine does not wrap, link to, or call the upstream C++ library. Upstream LightGBM is used only as a reference in two places: source inspection (git submodule `third_party/LightGBM`, pinned to v4.7.0 / `8f7036f`), and differential testing (`lightgbm==4.7.0` from PyPI, dev environment only).

## Layout

| path | contents |
|---|---|
| `crates/lgbm-core` | the engine: pure Rust, no Python dependency, usable as a Rust library |
| `crates/lgbm-py` | PyO3 bindings (`lightgbm_rust._lightgbm_rust`, abi3, Python ≥ 3.11) |
| `python/lightgbm_rust` | the Python API: `Dataset`, `Booster`, `train`, `cv`, callbacks, the scikit-learn estimators, `register_logger` |
| `tests/differential` | comparisons against LightGBM 4.7.0 (tolerances in `tests/tolerances.toml`) |
| `tests/upstream_runner` | runs upstream's Python test suite, unmodified, against `lightgbm_rust` |
| `tests/report/summary.md` | latest generated test and benchmark report |
| `benches/` | benchmark vs upstream (`bench_vs_upstream.py`) and a criterion micro-benchmark |
| `docs/` | [COMPATIBILITY](docs/COMPATIBILITY.md), [ARCHITECTURE](docs/ARCHITECTURE.md), [TESTING](docs/TESTING.md), [UPSTREAM](docs/UPSTREAM.md), [hazard design note](docs/hazard/design-note.md) |

## Install

```bash
pip install lightgbm-rust
```

Wheels are published for CPython 3.11–3.14 on Windows x64, macOS (arm64 and x86_64), and Linux (x86_64 and aarch64). Each platform wheel is built with Python 3.11 and tested on 3.11, 3.12, 3.13, and 3.14. `pip` downloads the matching wheel, so Rust is not required. Rust is required to build from the source distribution or to develop the package locally.

## Quick start

```bash
uv sync --python 3.12
uv run --no-sync maturin develop --release
uv run --no-sync python examples/train_binary.py
```

```python
import numpy as np
import lightgbm_rust as lgb

X = np.random.default_rng(0).normal(size=(1000, 5))
y = (X[:, 0] + X[:, 1] > 0).astype(float)
booster = lgb.train({"objective": "binary"}, lgb.Dataset(X, label=y), num_boost_round=50)
p = booster.predict(X)
booster.save_model("model.txt")  # loadable by upstream lightgbm.Booster(model_file=...)
```

See [docs/TESTING.md](docs/TESTING.md) for the full test commands. The differential suite was verified on Linux (WSL 2). Each release smoke-tests a small training run on native Windows, macOS, and Linux before publishing.

## License

MIT. Ported algorithms follow LightGBM (MIT, © Microsoft Corporation and the LightGBM developers); see [NOTICE](NOTICE).
