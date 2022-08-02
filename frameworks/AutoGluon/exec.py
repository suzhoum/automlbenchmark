import logging
import os
import shutil
import warnings
import sys
import tempfile
warnings.simplefilter("ignore")

if sys.platform == 'darwin':
    os.environ['OMP_NUM_THREADS'] = '1'

import matplotlib
import pandas as pd
matplotlib.use('agg')  # no need for tk

from autogluon.tabular import TabularPredictor, TabularDataset
import autogluon.core.metrics as metrics
from autogluon.tabular.version import __version__

from frameworks.shared.callee import call_run, result
from frameworks.shared.utils import Timer

from ag_utils.save_artifacts import save_artifacts
from ag_utils.zs_portfolio import get_hyperparameters_from_zeroshot_framework


log = logging.getLogger(__name__)


def run(dataset, config):
    log.info(f"\n**** AutoGluon [v{__version__}] ****\n")
    log_pip_freeze()

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

    train_path, test_path = dataset.train.path, dataset.test.path
    label = dataset.target.name
    problem_type = dataset.problem_type

    models_dir = tempfile.mkdtemp() + os.sep  # passed to AG

    _zeroshot_framework = config.framework_params.get('_zeroshot_framework', None)
    if _zeroshot_framework is not None:
        _hyperparameters = get_hyperparameters_from_zeroshot_framework(
            zeroshot_framework=_zeroshot_framework,
            config=config,
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
            time_limit=config.max_runtime_seconds,
            **training_params
        )

    test_data = TabularDataset(test_path)
    # Persist model in memory that is going to be predicting to get correct inference latency
    predictor.persist_models('best', max_memory=0.4)

    if is_classification:
        with Timer() as predict:
            probabilities = predictor.predict_proba(test_data, as_multiclass=True)
        predictions = probabilities.idxmax(axis=1).to_numpy()
    else:
        with Timer() as predict:
            predictions = predictor.predict(test_data, as_pandas=False)
        probabilities = None

    prob_labels = probabilities.columns.values.astype(str).tolist() if probabilities is not None else None

    _leaderboard_extra_info = config.framework_params.get('_leaderboard_extra_info', False)  # whether to get extra model info (very verbose)
    _leaderboard_test = config.framework_params.get('_leaderboard_test', False)  # whether to compute test scores in leaderboard (expensive)
    leaderboard_kwargs = dict(silent=True, extra_info=_leaderboard_extra_info)
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

    save_artifacts(predictor, leaderboard, config, test_data=test_data)
    shutil.rmtree(predictor.path, ignore_errors=True)

    return result(output_file=config.output_predictions_file,
                  predictions=predictions,
                  probabilities=probabilities,
                  probabilities_labels=prob_labels,
                  target_is_encoded=False,
                  models_count=num_models_trained,
                  models_ensemble_count=num_models_ensemble,
                  training_duration=training.duration,
                  predict_duration=predict.duration)


def log_pip_freeze():
    try:
        from pip._internal.operations import freeze
        pip_dependencies = freeze.freeze()
        log_pip_str = '\n===== pip freeze =====\n'
        for p in pip_dependencies:
            log_pip_str += f'{p}\n'
        log_pip_str += '======================\n'
        log.info(log_pip_str)
    except:
        pass


if __name__ == '__main__':
    call_run(run)
