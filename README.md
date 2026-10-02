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

## Changelog

See [CHANGELOG.md](CHANGELOG.md) (Keep a Changelog format).

## Roadmap

Open work is tracked in [GitHub issues](https://github.com/zrluety/lightgbm-rust/issues), labeled by `difficulty:` (small, medium, large), `area:` and `priority:`, and grouped into milestones. Items are listed in the suggested order.

**Phase 1: verification and performance**

- [#1](https://github.com/zrluety/lightgbm-rust/issues/1) Verify wheels natively on Windows and macOS (medium)
- [#2](https://github.com/zrluety/lightgbm-rust/issues/2) Fix benchmark noise: pin CPUs and add repeats (small)
- [#3](https://github.com/zrluety/lightgbm-rust/issues/3) Speed up `wide_sparse` and `sparse` prediction (small)
- [#4](https://github.com/zrluety/lightgbm-rust/issues/4) Reduce memory overhead (medium)
- [#5](https://github.com/zrluety/lightgbm-rust/issues/5) Speed up `wide_sparse` construction (small)
- [#6](https://github.com/zrluety/lightgbm-rust/issues/6) Speed up `wide_sparse` training at 4 and 8 threads (medium)
- [#7](https://github.com/zrluety/lightgbm-rust/issues/7) Add missing benchmarks (small)
- [#8](https://github.com/zrluety/lightgbm-rust/issues/8) Larger differential datasets and exact 4-thread metrics (medium)
- [#9](https://github.com/zrluety/lightgbm-rust/issues/9) Port remaining upstream C++ tests (small)

**Phase 2: API gaps**

- [#10](https://github.com/zrluety/lightgbm-rust/issues/10) `Sequence` Dataset input, which unblocks 27 upstream tests (medium)
- [#11](https://github.com/zrluety/lightgbm-rust/issues/11) `Dataset.add_features_from` (medium)
- [#12](https://github.com/zrluety/lightgbm-rust/issues/12) `Booster.set_leaf_output`, `shuffle_models`, `set_network` (small)
- [#13](https://github.com/zrluety/lightgbm-rust/issues/13) Private and plugin hooks used by upstream tests (small)

**Phase 3: parity edges**

- [#14](https://github.com/zrluety/lightgbm-rust/issues/14) Choose histogram layout by timing, like upstream (medium)
- [#15](https://github.com/zrluety/lightgbm-rust/issues/15) Decide behavior for gated combinations (medium)
- [#16](https://github.com/zrluety/lightgbm-rust/issues/16) Match upstream error messages and log lines (small)
- [#17](https://github.com/zrluety/lightgbm-rust/issues/17) Minor storage parity (small)
- [#18](https://github.com/zrluety/lightgbm-rust/issues/18) Upstream-compatible `.bin` Dataset files (large)
- [#19](https://github.com/zrluety/lightgbm-rust/issues/19) Bagging subset-copy mode and parallel GOSS (medium)

**Phase 4: distributed training**

- [#20](https://github.com/zrluety/lightgbm-rust/issues/20) Socket network layer and data-parallel learner (large)
- [#21](https://github.com/zrluety/lightgbm-rust/issues/21) Feature-parallel and voting-parallel learners (large)
- [#22](https://github.com/zrluety/lightgbm-rust/issues/22) `lightgbm.dask` (large)

**Phase 5: GPU and deferred**

- [#23](https://github.com/zrluety/lightgbm-rust/issues/23) GPU histogram offload with an exact quantized mode (large)
- [#24](https://github.com/zrluety/lightgbm-rust/issues/24) Full CUDA learner (large)
- [#25](https://github.com/zrluety/lightgbm-rust/issues/25) Hazard objective and panel loss (medium, deferred)

### Resuming development with an agent

- Read [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) and [docs/TESTING.md](docs/TESTING.md) first, then the issue.
- Dev pattern: `bash scripts/wsl-dev.sh ...`. Full verification is `target/run_all.sh`; launch it in the background and poll its log.
- Non-negotiables: bitwise parity with LightGBM 4.7.0 where upstream is deterministic, no weakened assertions or silent skips, no parity claims without evidence, signed commits, and update `docs/COMPATIBILITY.md` and `CHANGELOG.md` with each change.
- Out of scope for now: native Windows/macOS verification (#1 tracks it) and the hazard panel loss (#25).

## License

MIT. Ported algorithms follow LightGBM (MIT, © Microsoft Corporation and the LightGBM developers); see [NOTICE](NOTICE).
