from typing import List

import pandas as pd

from autogluon.tabular import TabularPredictor


def get_infer_speed_real(predictor: TabularPredictor,
                         test_data: pd.DataFrame,
                         batch_sizes: List[int] = None,
                         repeats: int = None) -> pd.DataFrame:
    # Lazy import to keep compatibility with old versions of AutoGluon
    from autogluon.core.utils.infer_utils import get_model_true_infer_speed_per_row_batch

    if batch_sizes is None:
        batch_sizes = [
            1,
            10,
            100,
            1000,
            10000,
            # 100000,  # Too big to safely fit into 32 GB memory on datasets with 10,000+ columns.
        ]

    best_model = predictor.get_model_best()

    infer_dfs = dict()
    for batch_size in batch_sizes:
        if repeats is None:
            repeat = 2 if batch_size <= 1000 else 1
        else:
            repeat = repeats
        infer_df, time_per_row_transform = get_model_true_infer_speed_per_row_batch(data=test_data, predictor=predictor, batch_size=batch_size, repeats=repeat)
        infer_df_best = infer_df[infer_df.index == best_model].copy()
        assert len(infer_df_best) == 1
        infer_df_best.index = ['best']

        infer_df_transform = pd.Series({
            'pred_time_test': time_per_row_transform,
            'pred_time_test_marginal': time_per_row_transform,
            'pred_time_test_with_transform': time_per_row_transform,
        }, name='transform_features').to_frame().T
        infer_df_transform.index.rename('model', inplace=True)

        infer_df = pd.concat([infer_df, infer_df_best, infer_df_transform])
        infer_df.index.name = 'model'
        infer_dfs[batch_size] = infer_df
    for key in infer_dfs.keys():
        infer_dfs[key] = infer_dfs[key].reset_index()
        infer_dfs[key]['batch_size'] = key
    infer_speed_df = pd.concat([infer_dfs[key] for key in infer_dfs.keys()])
    return infer_speed_df
