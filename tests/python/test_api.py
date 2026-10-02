"""Tests of the lightgbm_rust Python API itself (no upstream comparison)."""

import pickle
import threading
import time

import numpy as np
import pytest

import lightgbm_rust as lgb

BASE = {"verbosity": -1, "num_threads": 1, "deterministic": True}


def _data(n=500, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float64)
    return X, y


def test_numpy_layouts_give_identical_models():
    X, y = _data()
    ref = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(X, label=y), 10)
    expected = ref.predict(X)
    bst = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(np.asfortranarray(X), label=y), 10)
    np.testing.assert_array_equal(bst.predict(X), expected)
    # float32 input is widened to double exactly, like upstream
    X32 = X.astype(np.float32)
    a = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(X32, label=y), 10)
    b = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(X32.astype(np.float64), label=y), 10)
    assert a.model_to_string() == b.model_to_string()
    # non-contiguous views are copied, not rejected
    wide = np.repeat(X, 2, axis=1)[:, ::2]
    assert not wide.flags["C_CONTIGUOUS"] and not wide.flags["F_CONTIGUOUS"]
    np.testing.assert_array_equal(ref.predict(wide), expected)


def test_pandas_input_uses_column_names():
    pd = pytest.importorskip("pandas")
    X, y = _data()
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    bst = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(df, label=pd.Series(y)), 5)
    assert bst.feature_name() == ["a", "b", "c", "d"]
    np.testing.assert_array_equal(bst.predict(df), bst.predict(X))


def test_unsupported_features_raise_lightgbm_error(tmp_path):
    X, y = _data()
    for params in ({"objective": "cross_entropy"}, {"cegb_tradeoff": 0.5}, {"boosting": "dart"}):
        with pytest.raises(lgb.LightGBMError, match="not supported by lightgbm-rust yet"):
            lgb.train({**BASE, **params}, lgb.Dataset(X, label=y), 2)
    text = tmp_path / "train.csv"
    text.write_text("1,0.5,0.25\n0,0.1,0.2\n")
    with pytest.raises(lgb.LightGBMError, match="not supported by lightgbm-rust yet: training from files"):
        lgb.Dataset(text).construct()


def test_invalid_parameter_message_matches_upstream_check():
    X, y = _data()
    with pytest.raises(lgb.LightGBMError, match=r"Check failed: \(num_leaves\) > \(1\)"):
        lgb.train({**BASE, "num_leaves": 1}, lgb.Dataset(X, label=y), 2)


@pytest.mark.parametrize(
    ("params", "label_scale", "message"),
    [
        ({"objective": "huber", "reg_sqrt": True}, 1.0, "Cannot use sqrt transform in huber Regression, will auto disable it"),
        ({"objective": "mape"}, 0.5, "Some label values are < 1 in absolute value. MAPE is unstable with such values"),
    ],
)
def test_objective_warnings_match_upstream(capsys, params, label_scale, message):
    X, y = _data()
    lgb.train({**BASE, "verbosity": 0, **params}, lgb.Dataset(X, label=y * label_scale), 1)
    assert f"[LightGBM] [Warning] {message}" in capsys.readouterr().out


def test_feature_count_mismatch_on_predict():
    X, y = _data()
    bst = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(X, label=y), 2)
    with pytest.raises(lgb.LightGBMError, match=r"The number of features in data \(3\) is not the same as it was in training data \(4\)"):
        bst.predict(X[:, :3])


def test_dataset_param_change_after_free_raises():
    X, y = _data()
    ds = lgb.Dataset(X, label=y, params={"max_bin": 63}).construct()
    with pytest.raises(lgb.LightGBMError, match="Cannot change max_bin after constructed Dataset handle."):
        lgb.train({**BASE, "max_bin": 15}, ds, 2)


def test_custom_objective_matches_builtin_l2():
    X, _ = _data()
    y = X[:, 0] * 2 + np.sin(X[:, 1])

    def l2(preds, data):
        return preds - data.get_label(), np.ones_like(preds)

    params = {**BASE, "boost_from_average": False}
    builtin = lgb.train({**params, "objective": "regression"}, lgb.Dataset(X, label=y), 10)
    custom = lgb.train({**params, "objective": l2}, lgb.Dataset(X, label=y), 10)
    np.testing.assert_allclose(custom.predict(X), builtin.predict(X), rtol=1e-6, atol=1e-6)


def test_feval_record_and_early_stopping():
    X, y = _data(2000)
    Xv, yv = _data(500, seed=1)
    train = lgb.Dataset(X, label=y)
    valid = train.create_valid(Xv, label=yv)

    def err(preds, data):
        return "my_err", float(np.mean((preds > 0.5) != data.get_label())), False

    hist: dict = {}
    bst = lgb.train(
        {**BASE, "objective": "binary", "metric": "binary_logloss"},
        train,
        200,
        valid_sets=[train, valid],
        valid_names=["train", "val"],
        feval=err,
        callbacks=[lgb.record_evaluation(hist), lgb.early_stopping(5, verbose=False)],
    )
    assert set(hist) == {"train", "val"}
    assert set(hist["val"]) == {"binary_logloss", "my_err"}
    assert 0 < bst.best_iteration < 200
    assert set(bst.best_score["val"]) == {"binary_logloss", "my_err"}


def test_pickle_and_model_string_round_trip(tmp_path):
    X, y = _data()
    bst = lgb.train({**BASE, "objective": "binary"}, lgb.Dataset(X, label=y), 7)
    p = bst.predict(X)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(bst)).predict(X), p)
    path = tmp_path / "m.txt"
    bst.save_model(path)
    loaded = lgb.Booster(model_file=path)
    np.testing.assert_array_equal(loaded.predict(X), p)
    assert loaded.params["objective"] == "binary"
    assert loaded.model_to_string() == bst.model_to_string()
    leaves = loaded.predict(X, pred_leaf=True)
    assert leaves.shape == (X.shape[0], 7) and leaves.dtype == np.int32
    np.testing.assert_array_equal(loaded.predict(X, raw_score=True, num_iteration=3),
                                  bst.predict(X, raw_score=True, num_iteration=3))


def test_gil_released_during_training():
    X, y = _data(200_000, 20)
    ds = lgb.Dataset(X, label=y).construct()
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    bst = lgb.Booster({**BASE, "objective": "binary", "num_threads": 1}, ds)
    t.start()
    start = time.perf_counter()
    for _ in range(20):
        bst.update()
    elapsed = time.perf_counter() - start
    stop.set()
    t.join()
    # if the GIL were held, the ticker could only run between iterations
    assert elapsed > 0.05
    assert len([x for x in ticks if start <= x <= start + elapsed]) > 20


def test_panic_is_mapped_not_fatal():
    from lightgbm_rust import _lightgbm_rust as _rs

    with pytest.raises(lgb.LightGBMError):
        _rs.RsBooster.from_model_string("tree\nversion=v4\nnum_class=1\n")
