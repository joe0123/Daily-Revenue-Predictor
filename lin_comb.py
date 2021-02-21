import numpy as np
import os
import pickle
import itertools
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from dataset import Dataset
from utils import *


if __name__ == "__main__":
# Initialization
    dataset = Dataset("./data")
    adr_x, adr_y, test_adr_x, _ = dataset.get_adr_data(onehot_x=True, case="linear", scale=True)
    print(adr_x.shape)
    cancel_x, cancel_y, test_cancel_x, _ = dataset.get_cancel_data(onehot_x=True, case="linear", scale=True)
    print(cancel_x.shape)
    groups = np.array(dataset.get_groups("train"))
    total_nights = np.array(dataset.get_stay_nights("train"))
    labels_df = dataset.train_label_df
    test_groups = np.array(dataset.get_groups("test"))
    test_total_nights = np.array(dataset.get_stay_nights("test"))
    
    adr_model = Pipeline([("feature_selection", SelectFromModel(Lasso(max_iter=1e+8))), \
                    ("regression", Ridge(max_iter=1e+8))])
    cancel_model = Pipeline([("feature_selection", SelectFromModel(LogisticRegression(max_iter=1e+8, penalty="l1", \
                                                                                solver="liblinear", random_state=0))), 
    model = DailyRevenueEstimator(adr_model, cancel_model)
    
    params_grid = {
        ("adr", "feature_selection__estimator__alpha"): [1, 1e-1, 1e-2, 1e-3],
        ("adr", "regression__alpha"): [1, 1e-1, 1e-2, 1e-3],
        ("cancel", "feature_selection__estimator__C"): [1, 1e+1, 1e+2, 1e+3],
        ("cancel", "classification__C"): [1, 1e+1, 1e+2, 1e+3]}
    print(params_grid)

    #cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=range(2, 5))
    split_groups = ['-'.join(g.split('-')[:2]) for g in groups]
    cv = sliding_monthly_split(adr_x, split_groups=split_groups, start_group="2016-05", group_window=5, step=2, soft=True)
    cv_result = [i for i in cv]

    results = comb_search_cv(x=(adr_x, cancel_x), y=(adr_y, cancel_y), groups=groups, total_nights=total_nights, 
                labels_df=labels_df, model=model, params_grid=params_grid, cv=cv_result, n_iter=50, random_state=0, n_jobs=8)
    
    results_ = []
    for result in results:
        params_ = dict()
        for param, value in result["params"].items():
            params_[', '.join(param)] = value
        result["params"] = params_
        results_.append(result)
    print_format = lambda sort_key: json.dumps(sorted(results_, key=sort_key), indent=4)
    print("\nmean_score:", print_format(sort_key=lambda item: item["mean_score"]))
