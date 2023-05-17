import copy
from ast import literal_eval
from typing import Dict, List, Any

import pandas as pd

from autogluon.common.loaders import load_pd, load_pkl

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
                 convert_from_bag: bool = True,
                 portfolio_column: str = PORTFOLIO):
        self.config_hyperparameters_dict = config_hyperparameters_dict
        self.zeroshot_results_df = copy.deepcopy(zeroshot_results_df[zeroshot_results_df['framework'] == framework])
        assert len(self.zeroshot_results_df) > 0, f'framework="{framework}" missing from zeroshot_results_df!'
        self.convert_from_bag = convert_from_bag
        self.portfolio_column = portfolio_column
        self.framework = framework

    def get_ag_hyperparameters_from_portfolio(self, portfolio: List[str]) -> Dict[str, Any]:
        priority = len(portfolio)
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
                                       fold: int) -> Dict[str, Any]:
        portfolio = self.get_portfolio_for_dataset(dataset=dataset, fold=fold)
        return self.get_ag_hyperparameters_from_portfolio(portfolio=portfolio)



def get_zs_hpo_vendor(
    framework: str,
    zeroshot_results_path='s3://autogluon-zeroshot/config_results/zs_Bag244_test.csv',
    config_hyperparameters_dict_path='s3://autogluon-zeroshot/config_dict.pkl',
):
    zeroshot_results_df = load_pd.load(path=zeroshot_results_path)
    config_hyperparameters_dict = load_pkl.load(path=config_hyperparameters_dict_path)
    return ZeroshotHyperparametersVendor(
        framework=framework,
        zeroshot_results_df=zeroshot_results_df,
        config_hyperparameters_dict=config_hyperparameters_dict
    )
