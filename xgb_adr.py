import sys
import json
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from dataset import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _ = dataset.get_adr_data(onehot_x=True)
    groups = dataset.get_groups("train")
    #models = [weighted(XGBRegressor, 1.2)(random_state=0, criterion="mse", n_jobs=-1)]
    models = [XGBRegressor(criterion="mse", random_state=0, n_jobs=-1)]
    params_grids = [{
        "n_estimators": [600, 700, 800, 900, 1000, 1200, 1500],
        "max_features": ["auto"],
        "max_depth": [40, 50, 60, 70, 80, 90, 100, None],
        #"min_samples_split": [2, 5, 2e-4, 1e-3, 2e-3],
        "min_samples_leaf": [1, 2, 1e-4, 5e-4, 1e-3]}]

    print(params_grids)
    splits = range(2, 5)
    for model, params_grid in zip(models, params_grids):
        cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
        results = single_search_cv(x=train_x, y=train_y, groups=groups, model=model, params_grid=params_grid, cv=cv, \
                            scoring="neg_mean_absolute_error", n_iter=1, random_state=0, n_jobs=4)
        
        print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4)
        
        print("mean_score:", print_format(sort_key=lambda item: item["mean_score"]))
        for i, split in enumerate(splits):
            print("split{}_score:".format(split), print_format(sort_key=lambda item: item["valid_scores"][i]))
        print('\n')
