"""Training routines, following upstream ``python-package/lightgbm/engine.py`` (4.7.0).

Upstream code is Copyright Microsoft Corporation, MIT License (see NOTICE).
"""

import copy
from collections import OrderedDict, defaultdict
from operator import attrgetter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from . import callback
from .basic import Booster, Dataset, EvalResult, _choose_param_value, _ConfigAliases, _log_warning, _unsupported

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
    """Placeholder for ``lightgbm.CVBooster``; ``cv()`` is not implemented yet."""

    def __init__(self, model_file: Optional[Union[str, Path]] = None):
        raise _unsupported("cv() / CVBooster")


def cv(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Not implemented yet (milestone 4)."""
    raise _unsupported("cv()")
