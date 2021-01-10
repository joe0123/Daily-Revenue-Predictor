import sys
import json
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from dataset import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _, _ = dataset.get_cancel_data(onehot_x=True, case="linear", scale=True)
    print(train_x.shape)
    groups = dataset.get_groups("train")
    
    model = Pipeline([("feature_selection", SelectFromModel(LogisticRegression(max_iter=1e+8, penalty="l1", \
                                                                                solver="liblinear", random_state=0))), 
                    ("classification", LogisticRegression(max_iter=1e+8))])
    params_grid = {
        "feature_selection__estimator__C": [1, 1e+1, 1e+2],
        "classification__C": [1, 1e+1, 1e+2, 1e+3]}
    print(params_grid)

    #splits = range(2, 5)
    #cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
    split_groups = ['-'.join(g.split('-')[:2]) for g in groups]
    cv = sliding_monthly_split(train_x, split_groups=split_groups, start_group="2016-05", group_window=5, step=2, soft=True)
    results = single_search_cv(x=train_x, y=train_y, model=model, params_grid=params_grid, cv=cv, \
                        scoring="accuracy", n_iter=None, random_state=0, n_jobs=-1)
    
    print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4) 
    print("\nmean_score:", print_format(sort_key=lambda item: item["mean_score"]))
