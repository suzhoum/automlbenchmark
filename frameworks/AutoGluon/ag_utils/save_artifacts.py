import logging
import os
import shutil

import pandas as pd

from autogluon.common.savers import save_pd, save_pkl
from autogluon.tabular import TabularPredictor

from frameworks.shared.callee import touch
from frameworks.shared.utils import zip_path

from calc_infer_speed import get_infer_speed_real
from zs_artifact import get_zeroshot_artifact

log = logging.getLogger(__name__)


def get_save_path(config, suffix: str, create_dir: bool = True, as_dir: bool = False) -> str:
    path = os.path.join(config.output_dir, suffix)
    if create_dir:
        touch(path, as_dir=as_dir)
    return path


def save_artifacts(predictor: TabularPredictor, leaderboard: pd.DataFrame, config, test_data: pd.DataFrame):
    artifacts = config.framework_params.get('_save_artifacts', ['leaderboard'])
    try:
        if 'leaderboard' in artifacts:
            save_pd.save(path=get_save_path(config, "leaderboard.csv"), df=leaderboard)

        if 'info' in artifacts:
            info_path = get_save_path(config, 'info', as_dir=True)
            ag_info = predictor.info()
            ag_size_df = predictor.get_size_disk_per_file().to_frame().reset_index(names='file')
            save_pd.save(path=os.path.join(info_path, "file_sizes.csv"), df=ag_size_df)
            save_pkl.save(path=os.path.join(info_path, "info.pkl"), object=ag_info)

        if 'infer_speed' in artifacts:
            infer_speed_df = get_infer_speed_real(predictor=predictor, test_data=test_data)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                log.info(infer_speed_df)
            save_pd.save(path=get_save_path(config, "infer_speed.csv"), df=infer_speed_df)

        if 'zeroshot' in artifacts:
            zeroshot_path = get_save_path(config, 'zeroshot', as_dir=True)
            zeroshot_dict = get_zeroshot_artifact(predictor=predictor, test_data=test_data)
            save_pkl.save(path=os.path.join(zeroshot_path, "zeroshot_metadata.pkl"), object=zeroshot_dict)

        if 'models' in artifacts:
            shutil.rmtree(os.path.join(predictor.path, "utils"), ignore_errors=True)
            zip_path(predictor.path, get_save_path(config, "models.zip"))
    except Exception:
        log.warning("Error when saving artifacts.", exc_info=True)
