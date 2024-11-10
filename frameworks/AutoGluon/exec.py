import logging
import os
import shutil
import warnings
import sys
import tempfile
from typing import Union

warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result, measure_inference_times, output_subdir
from frameworks.shared.utils import Timer

from ag_utils.save_artifacts import ArtifactSaver
from ag_utils.zs_portfolio import get_hyperparameters_from_zeroshot_framework

log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")

    metrics_mapping = dict(
        acc=metrics.accuracy,
        auc=metrics.roc_auc,
        f1=metrics.f1,
        logloss=metrics.log_loss,
        mae=metrics.mean_absolute_error,
        mse=metrics.mean_squared_error,
        r2=metrics.r2,
        rmse=metrics.root_mean_squared_error,
    )

    perf_metric = metrics_mapping[config.metric] if config.metric in metrics_mapping else None
    if perf_metric is None:
        # TODO: figure out if we are going to blindly pass metrics through, or if we use a strict mapping
        log.warning("Performance metric %s not supported.", config.metric)

    is_classification = config.type == 'classification'
    training_params = {k: v for k, v in config.framework_params.items() if not k.startswith('_')}

    if callback_settings := training_params.get("callbacks", {}):
        training_params["callbacks"] = initialize_callbacks(callback_settings)

    time_limit = config.max_runtime_seconds
    presets = training_params.get("presets", [])
    presets = presets if isinstance(presets, list) else [presets]
    if (preset_with_refit_full := (set(presets) & {"good_quality", "high_quality"})) and (time_limit is not None):
        preserve = 0.9
        preset = next(iter(preset_with_refit_full))
        msg = (
            f"Detected `{preset}` preset, reducing `max_runtime_seconds` "
            f"from {config.max_runtime_seconds}s to "
            f"{preserve * config.max_runtime_seconds}s to account for `refit_full` "
            f"call after fit, which can take up to ~15% of total time. "
            "See https://auto.gluon.ai/stable/api/autogluon.tabular.TabularPredictor.refit_full.html"
        )
        log.info(msg)
        time_limit = preserve * config.max_runtime_seconds

    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    """
    The _include_test_during_fit flag enables the test_data to be passed into AutoGluon's predictor object
    during the fit call. If enabled, it is ensured that the test_data is seperated from all training and validation
    data. It is never seen by the models, nor does it influence the training process in any way.

    One might want to use this flag when generating learning curves. If this flag is enabled and learning_curves
    have been turned on, then your learning curve artifacts will also include curves for your test dataset.
    """
    _include_test_during_fit = config.framework_params.get('_include_test_during_fit', False)
    if _include_test_during_fit:
        training_params["test_data"] = test_path

    # whether to generate learning curves (VERY EXPENSIVE. Do not enable for benchmark comparisons.)
    if "learning_curves" in training_params:
        lc = training_params["learning_curves"]
        _curve_metrics = config.framework_params.get('_curve_metrics', {})
        if isinstance(lc, dict) and "metrics" not in lc and problem_type in _curve_metrics:
            training_params["learning_curves"]["metrics"] = _curve_metrics[problem_type]

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    # FIXME: Either remove or cleanup, this is very rarely used.
    _zeroshot_framework = config.framework_params.get('_zeroshot_framework', None)
    if _zeroshot_framework is not None:
        _zeroshot_include_defaults = config.framework_params.get('_zeroshot_include_defaults', False)
        _hyperparameters = get_hyperparameters_from_zeroshot_framework(
            zeroshot_framework=_zeroshot_framework,
            config=config,
            include_defaults=_zeroshot_include_defaults,
        )
        if _hyperparameters is not None:
            training_params['hyperparameters'] = _hyperparameters

    with Timer() as training:
        predictor = TabularPredictor(
            label=label,
            eval_metric=perf_metric.name,
            path=models_dir,
            problem_type=problem_type,
        ).fit(
            train_data=train_path,
            time_limit=time_limit,
            **training_params
        )

    artifact_saver = ArtifactSaver(predictor=predictor, config=config)

    log.info(f"Finished fit in {training.duration}s.")

    # Log failures
    if hasattr(predictor, 'model_failures'):  # version >0.8.2
        model_failures_df = predictor.model_failures()
        num_model_failures = len(model_failures_df)
        log.info(f'num_model_failures: {num_model_failures}')
        if num_model_failures > 0:
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                log.info(model_failures_df)
    else:
        model_failures_df = None
    artifact_saver.cache_post_fit(model_failures_df=model_failures_df)

    # Persist model in memory that is going to be predicting to get correct inference latency
    if hasattr(predictor, 'persist'):  # autogluon>=1.0
        predictor.persist('best')
    else:
        predictor.persist_models('best')

    def inference_time_classification(data: Union[str, pd.DataFrame]):
        return None, predictor.predict_proba(data, as_multiclass=True)

    def inference_time_regression(data: Union[str, pd.DataFrame]):
        return predictor.predict(data, as_pandas=False), None

    infer = inference_time_classification if is_classification else inference_time_regression
    inference_times = {}
    if config.measure_inference_time:
        inference_times["file"] = measure_inference_times(infer, dataset.inference_subsample_files)
        test_data = pd.read_parquet(dataset.test.path)
        inference_times["df"] = measure_inference_times(
            infer,
            [(1, test_data.sample(1, random_state=i)) for i in range(100)],
        )
        log.info(f"Finished inference time measurements.")

    test_data = TabularDataset(test_path)
    with Timer() as predict:
        predictions, probabilities = infer(test_data)
    if is_classification:
        if hasattr(predictor, 'predict_from_proba'):  # autogluon>=1.0
            predictions = predictor.predict_from_proba(probabilities).to_numpy()
        else:
            predictions = probabilities.idxmax(axis=1).to_numpy()

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None
    log.info(f"Finished predict in {predict.duration}s.")

    learning_curves = predictor.learning_curves() if training_params.get("learning_curves", None) else None
    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(extra_info=_leaderboard_extra_info)
    # Disabled leaderboard test data input by default to avoid long running computation, remove 7200s timeout limitation to re-enable
    if _leaderboard_test:
        leaderboard_kwargs['data'] = test_data

    leaderboard = predictor.leaderboard(**leaderboard_kwargs)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        log.info(leaderboard)

    num_models_trained = len(leaderboard)
    if predictor._trainer.model_best is not None:
        num_models_ensemble = len(predictor._trainer.get_minimum_model_set(predictor._trainer.model_best))
    else:
        num_models_ensemble = 1

    # FIXME: caching artifacts doesn't work correctly in local mode, because all dataset/folds will overwrite eachother.
    #  It does work fine in AWS mode though. Refer to https://github.com/openml/automlbenchmark/issues/647
    artifact_saver.cache_post_predict(leaderboard=leaderboard, test_data=test_data)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration,
                  inference_times=inference_times,)


def initialize_callbacks(callback_settings):
    callbacks = []
    try:
        import autogluon.core.callbacks
    except ImportError:
        raise ValueError("Callbacks are only available for AutoGluon>=1.1.2")
    for callback, hyperparameters in callback_settings.items():
        callback_cls = getattr(autogluon.core.callbacks, callback, None)
        if not callback_cls:
            raise ValueError(f"Callback {callback} is not a valid callback")
        callbacks.append(callback_cls(**hyperparameters))
    return callbacks


if __name__ == '__main__':
    call_run(run)
