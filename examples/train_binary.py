"""Train a binary classifier with lightgbm-rust and compare it with LightGBM 4.7.0.

    uv run python examples/train_binary.py

Upstream ``lightgbm`` is only imported for the comparison; skip it with
``--no-compare``.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np

import lightgbm_rust as lgbr


def make_data(n: int, p: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    X[rng.random(size=X.shape) < 0.02] = np.nan
    logit = 1.5 * np.nan_to_num(X[:, 0]) - np.nan_to_num(X[:, 1]) ** 2 + 0.5 * np.nan_to_num(X[:, 2] * X[:, 3])
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.float64)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    X, y = make_data(20_000, 10, seed=0)
    Xv, yv = make_data(5_000, 10, seed=1)
    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "learning_rate": 0.1,
        "num_leaves": 31,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "verbosity": -1,
    }

    train = lgbr.Dataset(X, label=y)
    valid = train.create_valid(Xv, label=yv)
    history: dict = {}
    bst = lgbr.train(
        params,
        train,
        num_boost_round=200,
        valid_sets=[valid],
        callbacks=[lgbr.early_stopping(20, verbose=False), lgbr.log_evaluation(50), lgbr.record_evaluation(history)],
    )
    print(f"lightgbm-rust: best_iteration={bst.best_iteration} best_score={dict(bst.best_score['valid_0'])}")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "model.txt"
        bst.save_model(path)
        reloaded = lgbr.Booster(model_file=path)
        p_rust = bst.predict(Xv)
        assert np.array_equal(p_rust, reloaded.predict(Xv)), "save/load round trip changed predictions"

        if args.no_compare:
            return
        import lightgbm as lgb  # reference only

        up = lgb.train(
            params,
            lgb.Dataset(X, label=y),
            num_boost_round=200,
            valid_sets=[lgb.Dataset(Xv, label=yv)],
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        p_up = up.predict(Xv)
        print(f"lightgbm 4.7.0: best_iteration={up.best_iteration} best_score={dict(up.best_score['valid_0'])}")
        print(f"max |p_rust - p_upstream| = {np.max(np.abs(p_rust - p_up)):.3e}")

        # Cross-loading: upstream reads our model file and predicts the same values.
        up_from_ours = lgb.Booster(model_file=str(path))
        print(f"max |upstream(our model) - ours| = {np.max(np.abs(up_from_ours.predict(Xv) - p_rust)):.3e}")


if __name__ == "__main__":
    main()
