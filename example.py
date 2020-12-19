import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset
from utils import *

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
    gtss = GroupTimeSeriesSplit(n_splits=5)

    a = np.unique(groups)

# Start time-series validation
    errs = []
    for train_i, train_group_i, valid_i, valid_group_i in gtss.split(adr_x, groups=groups, pass_splits=0):
    # Train adr model (regression)
        train_adr_x, train_adr_y = adr_x[train_i], adr_y[train_i]
        valid_adr_x, valid_adr_y = adr_x[valid_i], adr_y[valid_i]
        model_adr = Lasso(alpha=0.01, max_iter=1e+5).fit(train_adr_x, train_adr_y)
    
    # Train is_cancel model (classification)
        train_cancel_x, train_cancel_y = cancel_x[train_i], cancel_y[train_i]
        valid_cancel_x, valid_cancel_y = cancel_x[valid_i], cancel_y[valid_i]
        model_cancel = LogisticRegression(max_iter=1e+5).fit(train_cancel_x, train_cancel_y)
    
    # Calc daily revenue's error
        result = predict_daily_revenue(model_adr, model_cancel, valid_adr_x, valid_cancel_x, \
                                        total_nights[valid_i], groups[valid_i])
        err = np.mean(np.abs(result - labels[valid_group_i]))
        errs.append(err)
    
    print("Time Series Validation Error: {} {}".format(np.mean(errs), errs))

# Start training
    model_adr = Lasso(alpha=0.01, max_iter=1e+5).fit(adr_x, adr_y)
    model_cancel = LogisticRegression(max_iter=1e+5).fit(cancel_x, cancel_y)

# Start testing and Write the result file
    result = predict_daily_revenue(model_adr, model_cancel, test_adr_x, test_cancel_x, \
                                    test_total_nights, test_groups)
    df = dataset.test_nolabel_df
    df["label"] = result
    df.to_csv("result.csv", index=False)
