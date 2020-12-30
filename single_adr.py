import sys
import json
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor

from dataset import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, _ = dataset.get_adr_data()
    groups = dataset.get_groups("train")
    models = [weighted(RandomForestRegressor, 1.5)(criterion='mse', n_jobs=4)]
    #models = [RandomForestRegressor(criterion='mse', n_jobs=4)]
    params_grids = [{
        'random_state': [0, 42],
        'n_estimators': [10],#0, 200, 400, 600, 800, 1000],
        'max_features': ['auto'],
        'max_depth': [10, 20, 40, 60, 80, 100, None],
        'min_samples_split': [2, 5, 2e-4, 1e-3, 2e-3],
        'min_samples_leaf': [1, 2, 1e-4, 1e-3], }]
    print(params_grids)
    splits = range(2, 5)
    for model, params_grid in zip(models, params_grids):
        cv = GroupTimeSeriesSplit(n_splits=5).split(train_x, groups=groups, select_splits=splits)
        results = single_search_cv(x=train_x, y=train_y, groups=groups, model=model, params_grid=params_grid, cv=cv, \
                            scoring="neg_mean_absolute_error", n_iter=5, random_state=0, n_jobs=4)
        
        print_format = lambda sort_key: json.dumps(sorted(results, key=sort_key, reverse=True), indent=4)
        
        print("mean_score:", print_format(sort_key=lambda item: item["mean_score"]))
        for i, split in enumerate(splits):
            print("split{}_score:".format(split), print_format(sort_key=lambda item: item["scores"][i]))
        print('\n')
