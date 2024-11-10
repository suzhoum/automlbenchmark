import pandas as pd
from autogluon.tabular import TabularPredictor


def get_zeroshot_artifact(predictor: TabularPredictor, test_data: pd.DataFrame) -> dict:
    models = predictor.model_names(can_infer=True)

    if predictor.can_predict_proba:
        pred_proba_dict_val = predictor.predict_proba_multi(inverse_transform=False, as_multiclass=False, models=models)
        pred_proba_dict_test = predictor.predict_proba_multi(test_data, inverse_transform=False, as_multiclass=False, models=models)
    else:
        pred_proba_dict_val = predictor.predict_multi(inverse_transform=False, models=models)
        pred_proba_dict_test = predictor.predict_multi(test_data, inverse_transform=False, models=models)

    val_data_source = 'val' if predictor._trainer.has_val else 'train'
    _, y_val = predictor.load_data_internal(data=val_data_source, return_X=False, return_y=True)
    y_test = test_data[predictor.label]
    y_test = predictor.transform_labels(y_test, inverse=False)

    zeroshot_dict = dict(
        pred_proba_dict_val=pred_proba_dict_val,
        pred_proba_dict_test=pred_proba_dict_test,
        y_val=y_val,
        y_test=y_test,
        eval_metric=predictor.eval_metric.name,
        problem_type=predictor.problem_type,
        ordered_class_labels=predictor._learner.label_cleaner.ordered_class_labels,
        ordered_class_labels_transformed=predictor._learner.label_cleaner.ordered_class_labels_transformed,
        problem_type_transform=predictor._learner.label_cleaner.problem_type_transform,
        num_classes=predictor._learner.label_cleaner.num_classes,
        label=predictor.label,
    )

    return zeroshot_dict
