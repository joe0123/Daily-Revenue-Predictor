import os
import sys
import json
from xgboost import XGBClassifier
from dataset import Dataset
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _, _ = dataset.get_cancel_data(onehot_x=True)
    print(train_x.shape)
    groups = dataset.get_groups("train")
    model = XGBClassifier(objective="binary:logistic", eval_metric="error", 
            tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, use_label_encoder=False)
    params_grid = [{
        "n_estimators": [250],
        "learning_rate": [0.08],
        "min_child_weight": [18],
        "max_depth": [3],
        "gamma": [0],
        "subsample": [0.7],
        "colsample_bytree": [1.0],
        "reg_lambda": [1e-1],
        "reg_alpha": [1e-2],
        #"learning_rate": [0.2, 0.1, 0.08, 0.05, 0.02, 0.01],
        #"min_child_weight": [8, 10, 12, 15, 18, 20],
        #"max_depth": [2, 3, 5, 6, 8, 10],
        #"gamma": [0, 1, 2, 3, 5, 8, 10, 15],
        #"subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        #"colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        #"reg_lambda": [1, 1e-1, 1e-2, 1e-3],
        #"reg_alpha": [1, 1e-1, 1e-2, 1e-3, 0],
    }]

    print(params_grid)
    #cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
    split_groups = ['-'.join(g.split('-')[:2]) for g in groups]
    cv = sliding_monthly_split(train_x, split_groups=split_groups, start_group="2016-05", group_window=5, step=2, soft=True)
    results = single_search_cv(x=train_x, y=train_y, model=model, params_grid=params_grid, cv=cv, \
                        scoring="accuracy", n_iter=None, random_state=0, n_jobs=1)
    
    print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4) 
    print("\nmean_score:", print_format(sort_key=lambda item: item["mean_score"]))
