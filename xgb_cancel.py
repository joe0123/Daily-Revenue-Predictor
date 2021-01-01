import os
import sys
import json
from xgboost import XGBClassifier

from dataset import Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _ = dataset.get_cancel_data(onehot_x=True)
    print(train_x.shape)
    groups = dataset.get_groups("train")
    #models = [weighted(XGBClassifier, 1.2)(random_state=0, n_jobs=-1)]
    models = [XGBClassifier(objective="binary:logistic", eval_metric="error", 
                            tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, use_label_encoder=False)]
    params_grids = [{
        "n_estimators": [100],
        "learning_rate": [0.1],
        "min_child_weight": [10],
        "max_depth": [3],
        "gamma": [1],
        "subsample": [0.6],
        "colsample_bytree": [0.9],
        "reg_lambda": [1],
        "reg_alpha": [1e-1],
    }]
    print(params_grids, flush=True)
    splits = range(2, 5)

    for model, params_grid in zip(models, params_grids):
        cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
        results = single_search_cv(x=train_x, y=train_y, groups=groups, model=model, params_grid=params_grid, cv=cv, \
                            scoring="accuracy", n_iter=None, random_state=0, n_jobs=1)
        
        print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4)
        
        print("\nmean_score:", print_format(sort_key=lambda item: item["mean_score"]))
        for i, split in enumerate(splits):
            print("split{}_score:".format(split), print_format(sort_key=lambda item: item["valid_scores"][i]))
        print('\n')
