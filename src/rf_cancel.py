import sys
import json
from sklearn.ensemble import RandomForestClassifier

from dataset import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset("../data")
    train_x, train_y, _, _ = dataset.get_cancel_data(onehot_x=True)
    print(train_x.shape)
    groups = dataset.get_groups("train")
    model = RandomForestClassifier(random_state=0, n_jobs=4)
    params_grid = {
        "criterion": ["entropy"],
        "n_estimators": [300],
        "max_features": ["auto", None],
        "max_depth": [8, 10, 20, 40, 60, 80, 100, None],
        "min_samples_leaf": [2, 3, 5, 8, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3], }
    print(params_grid)
    #splits = range(2, 5)
    #cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
    split_groups = ['-'.join(g.split('-')[:2]) for g in groups]
    cv = sliding_monthly_split(train_x, split_groups=split_groups, start_group="2016-05", group_window=5, step=2, soft=True)
    results = single_search_cv(x=train_x, y=train_y, model=model, params_grid=params_grid, cv=cv, \
                        scoring="accuracy", n_iter=50, random_state=0, n_jobs=4)
    
    print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4) 
    print("\nmean_score:", print_format(sort_key=lambda item: item["mean_score"]))
