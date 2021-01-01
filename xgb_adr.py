import os
import sys
import json
from xgboost import XGBRegressor

from dataset import Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _ = dataset.get_adr_data(onehot_x=False)
    print(train_x.shape)
    groups = dataset.get_groups("train")
    #models = [weighted(XGBRegressor, 1.2)(random_state=0, n_jobs=-1)]
    models = [XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", random_state=0)]
    params_grids = [{
        "n_estimators": [100],
        "learning_rate": [0.1],
        "min_child_weight": [2],
        "max_depth": [5],
        "gamma": [0],
        "subsample": [0.7],
        "colsample_bytree": [0.6],
        "reg_lambda": [1e+1, 1, 1e-1, 1e-2, 1e-3],
        "reg_alpha": [1, 1e-1, 1e-2, 1e-3, 0],
    }]
    
    print(params_grids)
    splits = range(2, 5)
    for model, params_grid in zip(models, params_grids):
        cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
        results = single_search_cv(x=train_x, y=train_y, groups=groups, model=model, params_grid=params_grid, cv=cv, \
                            scoring="neg_mean_absolute_error", n_iter=None, random_state=0, n_jobs=1)
        
        print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4)
        
        print("\nmean_score:", print_format(sort_key=lambda item: item["mean_score"]))
        for i, split in enumerate(splits):
            print("split{}_score:".format(split), print_format(sort_key=lambda item: item["valid_scores"][i]))
        print('\n')
