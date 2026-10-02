#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install dist/*.whl
python - <<'PY'
import numpy as np
import lightgbm_rust as lgb

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 5))
y = (X[:, 0] + X[:, 1] > 0).astype(float)
booster = lgb.train({"objective": "binary"}, lgb.Dataset(X, label=y), num_boost_round=5)
pred = booster.predict(X)
if pred.shape != (100,):
    raise SystemExit(f"unexpected prediction shape {pred.shape}")
PY
