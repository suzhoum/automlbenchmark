import copy
import logging
from ast import literal_eval
from typing import Dict, List, Any

import pandas as pd

from autogluon.common.loaders import load_pd, load_json

log = logging.getLogger(__name__)

PORTFOLIO = 'config_selected'


def convert_config_name_bag_to_norm(config_name_bag: str):
    config_name = config_name_bag.rsplit('_BAG_', 1)
    assert len(config_name) == 2
    config_name = config_name[0]
    return config_name


class ZeroshotHyperparametersVendor:
    def __init__(self,
                 config_hyperparameters_dict: dict,
                 zeroshot_results_df: pd.DataFrame,
                 framework: str,
                 framework_column: str = "framework",
                 convert_from_bag: bool = True,
                 portfolio_column: str = PORTFOLIO):
        self.config_hyperparameters_dict = config_hyperparameters_dict
        self.zeroshot_results_df = copy.deepcopy(zeroshot_results_df[zeroshot_results_df[framework_column] == framework])
        assert len(self.zeroshot_results_df) > 0, f'framework="{framework}" missing from zeroshot_results_df!'
        self.convert_from_bag = convert_from_bag
        self.portfolio_column = portfolio_column
        self.framework = framework

    def get_ag_hyperparameters_from_portfolio(self, portfolio: List[str], include_defaults=False) -> Dict[str, Any]:
        if include_defaults:
            portfolio = [p for p in portfolio if "_c" not in p]
        priority = -1
        ag_hyperparameters = {}
        for m in portfolio:
            assert m in self.config_hyperparameters_dict, m
            hyperparameters = self.config_hyperparameters_dict[m]['hyperparameters']
            model_type = self.config_hyperparameters_dict[m]['model_type']
            hyperparameters_w_priority = copy.deepcopy(hyperparameters)
            if 'ag_args' in hyperparameters_w_priority:
                hyperparameters_w_priority['ag_args']['priority'] = priority
            else:
                hyperparameters_w_priority['ag_args'] = {'priority': priority}
            priority -= 1
            if model_type in ag_hyperparameters:
                ag_hyperparameters[model_type].append(hyperparameters_w_priority)
            else:
                ag_hyperparameters[model_type] = [hyperparameters_w_priority]

        if include_defaults:
            from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
            default_hyperparameters = get_hyperparameter_config(config_name="default")
            final_hyperparameters = copy.deepcopy(default_hyperparameters)
            for k in ag_hyperparameters:
                if k not in final_hyperparameters:
                    final_hyperparameters[k] = []
                if not isinstance(final_hyperparameters[k], list):
                    final_hyperparameters[k] = [final_hyperparameters[k]]
                final_hyperparameters[k] = final_hyperparameters[k] + ag_hyperparameters[k]
            ag_hyperparameters = final_hyperparameters

        return ag_hyperparameters

    def get_portfolio_for_dataset(self,
                                  dataset: str,
                                  fold: int):
        row_for_dataset = self.zeroshot_results_df[
            (self.zeroshot_results_df['dataset'] == dataset) & (self.zeroshot_results_df['fold'] == fold)]

        return self._get_portfolio_from_row(row_for_dataset=row_for_dataset)

    def get_portfolio_for_tid(self,
                              tid: int,
                              fold: int) -> List[str]:
        row_for_dataset = self.zeroshot_results_df[
            (self.zeroshot_results_df['tid'] == tid) & (self.zeroshot_results_df['fold'] == fold)]

        return self._get_portfolio_from_row(row_for_dataset=row_for_dataset)

    def _get_portfolio_from_row(self, row_for_dataset: pd.DataFrame):
        if len(row_for_dataset) == 1:  # task and fold present in zeroshot results
            portfolio = row_for_dataset.iloc[0][self.portfolio_column]
        elif len(row_for_dataset) == 0:  # task and fold not present in zeroshot results, use an arbitrary portfolio.
            print(f'NOTE: DATASET MISSING IN ZEROSHOT! Using default.')
            # FIXME: Technically should use a dedicated portfolio for this
            portfolio = self.zeroshot_results_df.iloc[0][self.portfolio_column]
        else:
            print(row_for_dataset)
            raise AssertionError(f'Found more than 1 row ({len(row_for_dataset)}) '
                                 f'framework={self.framework}.')

        portfolio = literal_eval(portfolio)
        assert isinstance(portfolio, list)

        if self.convert_from_bag:
            portfolio = [convert_config_name_bag_to_norm(m) for m in portfolio]
        return portfolio

    def get_ag_hyperparameters_for_tid(self,
                                       tid: int,
                                       fold: int) -> Dict[str, Any]:
        portfolio = self.get_portfolio_for_tid(tid=tid, fold=fold)
        return self.get_ag_hyperparameters_from_portfolio(portfolio=portfolio)

    def get_ag_hyperparameters_for_dataset(self,
                                           dataset: str,
                                           fold: int,
                                           include_defaults=False) -> Dict[str, Any]:
        portfolio = self.get_portfolio_for_dataset(dataset=dataset, fold=fold)
        return self.get_ag_hyperparameters_from_portfolio(portfolio=portfolio, include_defaults=include_defaults)


def get_zs_hpo_vendor(
    framework: str,
    zeroshot_results_path='s3://automl-benchmark-ag/ec2/zs_data_v2/simulation/D244_F3_C1416_200_ALL/results.csv',
    config_hyperparameters_dict_path='s3://automl-benchmark-ag/ec2/zs_data_v2/configs.json',
):
    zeroshot_results_df = load_pd.load(path=zeroshot_results_path)
    config_hyperparameters_dict = load_json.load(path=config_hyperparameters_dict_path)
    return ZeroshotHyperparametersVendor(
        framework=framework,
        zeroshot_results_df=zeroshot_results_df,
        config_hyperparameters_dict=config_hyperparameters_dict
    )


def get_hyperparameters_from_zeroshot_framework(
    zeroshot_framework: str,
    config,
    include_defaults=False,
    **kwargs,
) -> dict:
    log.info(f'ZEROSHOT FRAMEWORK: {zeroshot_framework}')
    zs_vendor = get_zs_hpo_vendor(framework=zeroshot_framework, **kwargs)
    dataset_name = config.name
    fold = config.fold
    log.info(f'fold={fold}, dataset_name={dataset_name}')
    portfolio = zs_vendor.get_portfolio_for_dataset(dataset=dataset_name, fold=fold)
    log.info(f'Zeroshot Portfolio: {portfolio}')
    hyperparameters = zs_vendor.get_ag_hyperparameters_for_dataset(dataset=dataset_name, fold=fold, include_defaults=include_defaults)
    log.info(hyperparameters)
    return hyperparameters
