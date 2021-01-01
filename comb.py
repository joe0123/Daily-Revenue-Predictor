import numpy as np
from tqdm import tqdm
import itertools
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, Lasso, Ridge, HuberRegressor, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

from dataset import Dataset
from utils import *

MODEL_ADR = RandomForestRegressor(n_estimators=800, min_samples_split=2, min_samples_leaf=1, max_features="auto", max_depth=60, criterion="mse", random_state=0, n_jobs=-1)
print(MODEL_ADR)

MODEL_CANCEL = RandomForestClassifier(n_estimators=400, min_samples_split=2e-3, min_samples_leaf=5e-4, max_features=None, max_depth=10, criterion="entropy", random_state=0, n_jobs=-1)
print(MODEL_CANCEL)

if __name__ == "__main__":
# Initialization
    dataset = Dataset("./data")
    adr_x, adr_y, test_adr_x = dataset.get_adr_data(onehot_x=True)
    print(adr_x.shape)
    cancel_x, cancel_y, test_cancel_x = dataset.get_cancel_data(onehot_x=True)
    print(cancel_x.shape)
    groups = np.array(dataset.get_groups("train"))
    total_nights = np.array(dataset.get_stay_nights("train"))
    labels = dataset.train_label_df["label"].to_numpy()
    test_groups = np.array(dataset.get_groups("test"))
    test_total_nights = np.array(dataset.get_stay_nights("test"))
    
    model = DailyRevenueEstimator(MODEL_ADR, MODEL_CANCEL)
    
    #cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=[2])
    cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=range(2, 5))

# Start CV
    errs = []
    for train_i, (train_gi, _), valid_i, (valid_gi, _) in tqdm(cv):
        train_adr_x, train_adr_y = adr_x[train_i], adr_y[train_i]
        valid_adr_x, _ = adr_x[valid_i], adr_y[valid_i]
        train_cancel_x, train_cancel_y = cancel_x[train_i], cancel_y[train_i]
        valid_cancel_x, _ = cancel_x[valid_i], cancel_y[valid_i]
        valid_labels = labels[valid_gi]

        model = model.fit(train_adr_x, train_adr_y, train_cancel_x, train_cancel_y)

        err = model.score(valid_adr_x, valid_cancel_x, valid_labels, total_nights[valid_i], groups[valid_i])
        errs.append(err)
    
    print("Errors: {:.2f} {}".format(np.mean(errs), errs), flush=True)

# Start re-training
    model = model.fit(adr_x, adr_y, cancel_x, cancel_y)

# Start testing and Write the result file
    result = model.predict(test_adr_x, test_cancel_x, test_total_nights, test_groups)
    df = dataset.test_nolabel_df
    df["label"] = result
    df.to_csv("result.csv", index=False)
