import os
import sys
import json
from functools import partial
from xgboost import XGBRegressor
import optuna

from dataset import Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'


def optuna_obj(trial, train_x, train_y, groups, splits):
    model = XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", eval_metric="mae", random_state=0)
    weight_ratio = 1
    params = {
        "n_estimators": 200, 
        "learning_rate": trial.suggest_float("learning_rate", 1e-2, 2e-1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "gamma": 0,
        "subsample": trial.suggest_float("subsample", 0.5, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-4, 1),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-4, 1),
    }
    cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
    
    result = single_cv(x=train_x, y=train_y, groups=groups, model=model, weight_ratio=weight_ratio, \
                    params=params, cv=cv, scoring="neg_mean_absolute_error")
    
    return result["mean_score"]



if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _ = dataset.get_adr_data(onehot_x=True)
    print(train_x.shape)
    groups = np.array(dataset.get_groups("train"))
    splits = range(2, 5)
    
    optuna_obj = partial(optuna_obj, train_x=train_x, train_y=train_y, groups=groups, splits=splits)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True))
    study.optimize(optuna_obj, n_trials=500)
    trials = [{"score": t.value, "params": t.params} for t in study.get_trials()]
    
    print('\n', json.dumps(sorted(trials, key=lambda item: item["score"], reverse=True), indent=4), sep='')
