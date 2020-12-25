import numpy as np
from tqdm import tqdm
import itertools
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from datasets import Dataset
from utils import *

#MODEL_ADR = Lasso(alpha=0.01, max_iter=1e+5)
#MODEL_CANCEL = LogisticRegression(max_iter=1e+5)
MODEL_ADR = Pipeline([("feature_selection", SelectFromModel(Lasso(max_iter=1e+8))), \
                        ("regression", Ridge(max_iter=1e+8))])
MODEL_CANCEL = Pipeline([("feature_selection", SelectFromModel(Lasso(max_iter=1e+8))), \
                        ("classifier", LogisticRegression(max_iter=1e+8))])
PARAM_GRID = {("adr", "feature_selection__estimator__alpha"): [1e-1, 1e-2, 1e-3], \
                ("adr", "regression__alpha"): [1, 1e-1, 1e-2, 1e-3], \
                ("cancel", "feature_selection__estimator__alpha"): [1e-1, 1e-2, 1e-3], \
                ("cancel", "classifier__C"): [1e+2, 1e+1, 1, 1e-1, 1e-2]}

if __name__ == "__main__":
# Initialization
    dataset = Dataset("./data")
    adr_x, adr_y, test_adr_x = dataset.get_adr_data()
    cancel_x, cancel_y, test_cancel_x = dataset.get_cancel_data()
    groups = np.array(dataset.get_groups("train"))
    total_nights = np.array(dataset.get_stay_nights("train"))
    labels = dataset.train_label_df["label"].to_numpy()
    test_groups = np.array(dataset.get_groups("test"))
    test_total_nights = np.array(dataset.get_stay_nights("test"))
    
    model = DailyRevenueEstimator(MODEL_ADR, MODEL_CANCEL)
    
    cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=[2], return_group_i=True)
    param_grid = ParameterGrid(PARAM_GRID)

# Start grid search
    record = []
    for params in tqdm(param_grid):
        errs = []
        model.set_params(params)
        cv, cv_run = itertools.tee(cv, 2)
        for train_i, train_group_i, valid_i, valid_group_i in cv_run:
            train_adr_x, train_adr_y = adr_x[train_i], adr_y[train_i]
            valid_adr_x, _ = adr_x[valid_i], adr_y[valid_i]
            train_cancel_x, train_cancel_y = cancel_x[train_i], cancel_y[train_i]
            valid_cancel_x, _ = cancel_x[valid_i], cancel_y[valid_i]
            valid_labels = labels[valid_group_i]

            model = model.fit(train_adr_x, train_adr_y, train_cancel_x, train_cancel_y)

            err = model.score(valid_adr_x, valid_cancel_x, valid_labels, total_nights[valid_i], groups[valid_i])
            errs.append(err)
        
        record.append((params, errs))
    
    print(record, flush=True)
    best_params, best_scores = min(record, key=lambda t: sum(t[1]))
    print("Best Params: {} \nBest Error: {:.2f} {}".format(best_params, np.mean(best_scores), best_scores), flush=True)

# Start re-training
    model.set_params(best_params)
    model = model.fit(adr_x, adr_y, cancel_x, cancel_y)

# Start testing and Write the result file
    result = model.predict(test_adr_x, test_cancel_x, test_total_nights, test_groups)
    df = dataset.test_nolabel_df
    df["label"] = result
    df.to_csv("result.csv", index=False)
