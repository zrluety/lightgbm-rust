"""Training routines, following upstream ``python-package/lightgbm/engine.py`` (4.7.0).

Upstream code is Copyright Microsoft Corporation, MIT License (see NOTICE).
"""

import copy
import json
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from . import callback
from .basic import (
    Booster,
    Dataset,
    EvalResult,
    LightGBMError,
    _choose_param_value,
    _ConfigAliases,
    _log_warning,
    _unsupported,
)
from .compat import SKLEARN_INSTALLED, _LGBMGroupKFold, _LGBMStratifiedKFold

__all__ = ["CVBooster", "cv", "train"]


def _choose_num_iterations(*, num_boost_round_kwarg: int, params: Dict[str, Any]) -> Dict[str, Any]:
    num_iteration_configs_provided = {
        alias: params[alias] for alias in _ConfigAliases.get("num_iterations") if alias in params
    }
    params = _choose_param_value(main_param_name="num_iterations", params=params, default_value=num_boost_round_kwarg)
    if len(num_iteration_configs_provided) <= 1:
        return params
    if len(set(num_iteration_configs_provided.values())) <= 1:
        return params
    value_string = ", ".join(f"{alias}={val}" for alias, val in num_iteration_configs_provided.items())
    _log_warning(
        f"Found conflicting values for num_iterations provided via 'params': {value_string}. "
        f"LightGBM will perform up to {params['num_iterations']} boosting rounds. "
        "To be confident in the maximum number of boosting rounds LightGBM will perform and to "
        "suppress this warning, modify 'params' so that only one of those is present."
    )
    return params


def train(
    params: Dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    feval: Any = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    keep_training_booster: bool = False,
    callbacks: Optional[List[Any]] = None,
) -> Booster:
    """Perform the training with given parameters (same contract as ``lightgbm.train``)."""
    if not isinstance(train_set, Dataset):
        raise TypeError(f"train() only accepts Dataset object, train_set has type '{type(train_set).__name__}'.")
    if isinstance(valid_sets, list):
        for i, valid_item in enumerate(valid_sets):
            if not isinstance(valid_item, Dataset):
                raise TypeError(
                    "Every item in valid_sets must be a Dataset object. "
                    f"Item {i} has type '{type(valid_item).__name__}'."
                )
    if init_model is not None:
        raise _unsupported("init_model / continued training")

    params = copy.deepcopy(params)
    params = _choose_param_value(main_param_name="objective", params=params, default_value=None)
    fobj = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = "none"

    params = _choose_num_iterations(num_boost_round_kwarg=num_boost_round, params=params)
    num_boost_round = params["num_iterations"]
    if num_boost_round <= 0:
        raise ValueError(f"Number of boosting rounds must be greater than 0. Got {num_boost_round}.")

    params = _choose_param_value(main_param_name="early_stopping_round", params=params, default_value=None)
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    first_metric_only = params.get("first_metric_only", False)

    init_iteration = 0
    train_set._update_params(params)._set_predictor(None)

    is_valid_contain_train = False
    train_data_name = "training"
    reduced_valid_sets = []
    name_valid_sets = []
    if valid_sets is not None:
        if isinstance(valid_sets, Dataset):
            valid_sets = [valid_sets]
        if isinstance(valid_names, str):
            valid_names = [valid_names]
        for i, valid_data in enumerate(valid_sets):
            if valid_data is train_set:
                is_valid_contain_train = True
                if valid_names is not None:
                    train_data_name = valid_names[i]
                continue
            reduced_valid_sets.append(valid_data._update_params(params).set_reference(train_set))
            if valid_names is not None and len(valid_names) > i:
                name_valid_sets.append(valid_names[i])
            else:
                name_valid_sets.append(f"valid_{i}")

    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault("order", i - len(callbacks))
        callbacks_set = set(callbacks)

    if callback._should_enable_early_stopping(params.get("early_stopping_round", 0)):
        callbacks_set.add(
            callback.early_stopping(
                stopping_rounds=params["early_stopping_round"],
                first_metric_only=first_metric_only,
                min_delta=params.get("early_stopping_min_delta", 0.0),
                verbose=_choose_param_value(main_param_name="verbosity", params=params, default_value=1).pop("verbosity")
                > 0,
            )
        )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, "before_iteration", False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter("order"))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter("order"))

    try:
        booster = Booster(params=params, train_set=train_set)
        if is_valid_contain_train:
            booster.set_train_data_name(train_data_name)
        for valid_set, name_valid_set in zip(reduced_valid_sets, name_valid_sets):
            booster.add_valid(valid_set, name_valid_set)
    finally:
        train_set._reverse_update_params()
        for valid_set in reduced_valid_sets:
            valid_set._reverse_update_params()
    booster.best_iteration = 0

    evaluation_result_list: List[EvalResult] = []
    for i in range(init_iteration, init_iteration + num_boost_round):
        for cb in callbacks_before_iter:
            cb(
                callback.CallbackEnv(
                    model=booster,
                    params=params,
                    iteration=i,
                    begin_iteration=init_iteration,
                    end_iteration=init_iteration + num_boost_round,
                    evaluation_result_list=None,
                )
            )

        booster.update(fobj=fobj)

        evaluation_result_list = []
        if valid_sets is not None:
            if is_valid_contain_train:
                evaluation_result_list.extend(booster.eval_train(feval))
            evaluation_result_list.extend(booster.eval_valid(feval))
        try:
            for cb in callbacks_after_iter:
                cb(
                    callback.CallbackEnv(
                        model=booster,
                        params=params,
                        iteration=i,
                        begin_iteration=init_iteration,
                        end_iteration=init_iteration + num_boost_round,
                        evaluation_result_list=evaluation_result_list,
                    )
                )
        except callback.EarlyStopException as earlyStopException:
            booster.best_iteration = earlyStopException.best_iteration + 1
            evaluation_result_list = earlyStopException.best_score
            break
    booster.best_score = defaultdict(OrderedDict)
    for result in evaluation_result_list:
        booster.best_score[result.dataset_name][result.metric_name] = result.metric_value
    if not keep_training_booster:
        booster.model_from_string(booster.model_to_string()).free_dataset()
    return booster


class CVBooster:
    """Holds the per-fold boosters of ``cv()`` and redirects method calls to all of them.

    Same contract as ``lightgbm.CVBooster``: every method except
    ``model_from_string``, ``model_to_string`` and ``save_model`` is called on
    each booster and the results are returned as a list.
    """

    def __init__(self, model_file: Optional[Union[str, Path]] = None):
        self.boosters: List[Booster] = []
        self.best_iteration = -1
        if model_file is not None:
            with open(model_file, "r") as file:
                self._from_dict(json.load(file))

    def _from_dict(self, models: Dict[str, Any]) -> None:
        self.best_iteration = models["best_iteration"]
        self.boosters = []
        for model_str in models["boosters"]:
            self.boosters.append(Booster(model_str=model_str))

    def _to_dict(self, *, num_iteration: Optional[int], start_iteration: int, importance_type: str) -> Dict[str, Any]:
        models_str = []
        for booster in self.boosters:
            models_str.append(
                booster.model_to_string(
                    num_iteration=num_iteration, start_iteration=start_iteration, importance_type=importance_type
                )
            )
        return {"boosters": models_str, "best_iteration": self.best_iteration}

    def __getattr__(self, name: str) -> Callable[[Any, Any], List[Any]]:
        def handler_function(*args: Any, **kwargs: Any) -> List[Any]:
            ret = []
            for booster in self.boosters:
                ret.append(getattr(booster, name)(*args, **kwargs))
            return ret

        return handler_function

    def __getstate__(self) -> Dict[str, Any]:
        return vars(self)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        vars(self).update(state)

    def model_from_string(self, model_str: str) -> "CVBooster":
        self._from_dict(json.loads(model_str))
        return self

    def model_to_string(
        self, num_iteration: Optional[int] = None, start_iteration: int = 0, importance_type: str = "split"
    ) -> str:
        return json.dumps(
            self._to_dict(num_iteration=num_iteration, start_iteration=start_iteration, importance_type=importance_type)
        )

    def save_model(
        self,
        filename: Union[str, Path],
        num_iteration: Optional[int] = None,
        start_iteration: int = 0,
        importance_type: str = "split",
    ) -> "CVBooster":
        with open(filename, "w") as file:
            json.dump(
                self._to_dict(
                    num_iteration=num_iteration, start_iteration=start_iteration, importance_type=importance_type
                ),
                file,
            )
        return self


def _make_n_folds(
    *,
    full_data: Dataset,
    folds: Any,
    nfold: int,
    params: Dict[str, Any],
    seed: int,
    fpreproc: Any,
    stratified: bool,
    shuffle: bool,
    eval_train_metric: bool,
) -> CVBooster:
    """Make a n-fold list of Booster from random indices."""
    full_data = full_data.construct()
    num_data = full_data.num_data()
    if folds is not None:
        if not hasattr(folds, "__iter__") and not hasattr(folds, "split"):
            raise AttributeError(
                "folds should be a generator or iterator of (train_idx, test_idx) tuples "
                "or scikit-learn splitter object with split method"
            )
        if hasattr(folds, "split"):
            group_info = full_data.get_group()
            if group_info is not None:
                group_info = np.asarray(group_info, dtype=np.int32)
                flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            else:
                flatted_group = np.zeros(num_data, dtype=np.int32)
            folds = folds.split(X=np.empty(num_data), y=full_data.get_label(), groups=flatted_group)
    else:
        if any(
            params.get(obj_alias, "")
            in {"lambdarank", "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
            for obj_alias in _ConfigAliases.get("objective")
        ):
            if not SKLEARN_INSTALLED:
                raise LightGBMError("scikit-learn is required for ranking cv")
            group_info = np.asarray(full_data.get_group(), dtype=np.int32)
            flatted_group = np.repeat(range(len(group_info)), repeats=group_info)
            group_kfold = _LGBMGroupKFold(n_splits=nfold)
            folds = group_kfold.split(X=np.empty(num_data), groups=flatted_group)
        elif stratified:
            if not SKLEARN_INSTALLED:
                raise LightGBMError("scikit-learn is required for stratified cv")
            skf = _LGBMStratifiedKFold(n_splits=nfold, shuffle=shuffle, random_state=seed)
            folds = skf.split(X=np.empty(num_data), y=full_data.get_label())
        else:
            if shuffle:
                randidx = np.random.RandomState(seed).permutation(num_data)
            else:
                randidx = np.arange(num_data)
            test_id = np.array_split(randidx, nfold)
            train_id = [np.concatenate([test_id[i] for i in range(nfold) if k != i]) for k in range(nfold)]
            folds = zip(train_id, test_id, strict=True)

    ret = CVBooster()
    for train_idx, test_idx in folds:
        train_set = full_data.subset(sorted(train_idx))
        valid_set = full_data.subset(sorted(test_idx))
        if fpreproc is not None:
            train_set, valid_set, tparam = fpreproc(train_set, valid_set, params.copy())
        else:
            tparam = params
        booster_for_fold = Booster(tparam, train_set)
        if eval_train_metric:
            booster_for_fold.add_valid(train_set, "train")
        booster_for_fold.add_valid(valid_set, "valid")
        ret.boosters.append(booster_for_fold)
    return ret


def _agg_cv_result(raw_results: List[List[EvalResult]]) -> List[EvalResult]:
    """Aggregate cross-validation results."""
    metric_types: Dict[Tuple[str, str], bool] = OrderedDict()
    metric_values: Dict[Tuple[str, str], List[float]] = OrderedDict()
    for result_list in raw_results:
        for result in result_list:
            key = (result.dataset_name, result.metric_name)
            metric_types[key] = result.maximize
            metric_values.setdefault(key, [])
            metric_values[key].append(result.metric_value)
    return [
        EvalResult(
            dataset_name=k[0],
            metric_name=k[1],
            metric_value=float(np.mean(v)),
            maximize=metric_types[k],
            metric_std_dev=float(np.std(v)),
        )
        for k, v in metric_values.items()
    ]


def cv(
    params: Dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    folds: Optional[Union[Iterable[Tuple[np.ndarray, np.ndarray]], Any]] = None,
    nfold: int = 5,
    stratified: bool = True,
    shuffle: bool = True,
    metrics: Optional[Union[str, List[str]]] = None,
    feval: Any = None,
    init_model: Optional[Union[str, Path, Booster]] = None,
    fpreproc: Any = None,
    seed: int = 0,
    callbacks: Optional[List[Callable]] = None,
    eval_train_metric: bool = False,
    return_cvbooster: bool = False,
) -> Dict[str, Union[List[float], CVBooster]]:
    """Perform the cross-validation with given parameters (same contract as ``lightgbm.cv``)."""
    if not isinstance(train_set, Dataset):
        raise TypeError(f"cv() only accepts Dataset object, train_set has type '{type(train_set).__name__}'.")

    params = copy.deepcopy(params)
    params = _choose_param_value(main_param_name="objective", params=params, default_value=None)
    fobj = None
    if callable(params["objective"]):
        fobj = params["objective"]
        params["objective"] = "none"

    params = _choose_num_iterations(num_boost_round_kwarg=num_boost_round, params=params)
    num_boost_round = params["num_iterations"]
    if num_boost_round <= 0:
        raise ValueError(f"Number of boosting rounds must be greater than 0. Got {num_boost_round}.")

    params = _choose_param_value(main_param_name="early_stopping_round", params=params, default_value=None)
    if params["early_stopping_round"] is None:
        params.pop("early_stopping_round")
    first_metric_only = params.get("first_metric_only", False)

    if init_model is not None:
        raise _unsupported("init_model / continued training")

    if metrics is not None:
        for metric_alias in _ConfigAliases.get("metric"):
            params.pop(metric_alias, None)
        params["metric"] = metrics

    train_set._update_params(params)._set_predictor(None)

    results = defaultdict(list)
    cvbooster = _make_n_folds(
        full_data=train_set,
        folds=folds,
        nfold=nfold,
        params=params,
        seed=seed,
        fpreproc=fpreproc,
        stratified=stratified,
        shuffle=shuffle,
        eval_train_metric=eval_train_metric,
    )

    if callbacks is None:
        callbacks_set = set()
    else:
        for i, cb in enumerate(callbacks):
            cb.__dict__.setdefault("order", i - len(callbacks))
        callbacks_set = set(callbacks)

    if callback._should_enable_early_stopping(params.get("early_stopping_round", 0)):
        callbacks_set.add(
            callback.early_stopping(
                stopping_rounds=params["early_stopping_round"],
                first_metric_only=first_metric_only,
                min_delta=params.get("early_stopping_min_delta", 0.0),
                verbose=_choose_param_value(main_param_name="verbosity", params=params, default_value=1).pop("verbosity")
                > 0,
            )
        )

    callbacks_before_iter_set = {cb for cb in callbacks_set if getattr(cb, "before_iteration", False)}
    callbacks_after_iter_set = callbacks_set - callbacks_before_iter_set
    callbacks_before_iter = sorted(callbacks_before_iter_set, key=attrgetter("order"))
    callbacks_after_iter = sorted(callbacks_after_iter_set, key=attrgetter("order"))

    for i in range(num_boost_round):
        for cb in callbacks_before_iter:
            cb(
                callback.CallbackEnv(
                    model=cvbooster,
                    params=params,
                    iteration=i,
                    begin_iteration=0,
                    end_iteration=num_boost_round,
                    evaluation_result_list=None,
                )
            )
        cvbooster.update(fobj=fobj)
        evaluation_result_list = _agg_cv_result(cvbooster.eval_valid(feval))
        for result in evaluation_result_list:
            results[f"{result.dataset_name} {result.metric_name}-mean"].append(result.metric_value)
            results[f"{result.dataset_name} {result.metric_name}-stdv"].append(result.metric_std_dev)
        try:
            for cb in callbacks_after_iter:
                cb(
                    callback.CallbackEnv(
                        model=cvbooster,
                        params=params,
                        iteration=i,
                        begin_iteration=0,
                        end_iteration=num_boost_round,
                        evaluation_result_list=evaluation_result_list,
                    )
                )
        except callback.EarlyStopException as earlyStopException:
            cvbooster.best_iteration = earlyStopException.best_iteration + 1
            for bst in cvbooster.boosters:
                bst.best_iteration = cvbooster.best_iteration
            for k in results:
                results[k] = results[k][: cvbooster.best_iteration]
            break

    if return_cvbooster:
        results["cvbooster"] = cvbooster

    return dict(results)
