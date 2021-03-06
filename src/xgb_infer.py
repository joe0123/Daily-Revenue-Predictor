import numpy as np
import os
import json
import itertools
from xgboost import XGBRegressor, XGBClassifier

from dataset import Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


adr_model = XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, \
                        n_estimators=200, learning_rate=1e-1, min_child_weight=8, max_depth=8, gamma=0, \
                        subsample=0.7, colsample_bytree=0.8, reg_lambda=1, reg_alpha=1e-3)
cancel_model = XGBClassifier(objective="binary:logistic", eval_metric="error", \
                            tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, use_label_encoder=False, \
                            n_estimators=250, learning_rate=0.08, min_child_weight=8, max_depth=3, gamma=0, \
                            subsample=0.7, colsample_bytree=0.8, reg_lambda=1, reg_alpha=1e-3)

model = DailyRevenueEstimator(adr_model, cancel_model)

if __name__ == "__main__":
# Initialization
    dataset = Dataset("../data")
    adr_x, adr_y, test_adr_x, _ = dataset.get_adr_data(onehot_x=True)
    print(adr_x.shape)
    cancel_x, cancel_y, test_cancel_x, _ = dataset.get_cancel_data(onehot_x=True)
    print(cancel_x.shape)
    groups = np.array(dataset.get_groups("train"))
    total_nights = np.array(dataset.get_stay_nights("train"))
    labels_df = dataset.train_label_df
    test_groups = np.array(dataset.get_groups("test"))
    test_total_nights = np.array(dataset.get_stay_nights("test"))
    
    #cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=range(2, 5))
    split_groups = ['-'.join(g.split('-')[:2]) for g in groups]
    cv = sliding_monthly_split(adr_x, split_groups=split_groups, start_group="2016-05", group_window=5, step=2, soft=True)
    cv_result = [i for i in cv]

    single_cv(x=adr_x, y=adr_y, model=adr_model, params={}, cv=cv_result, scoring="neg_mean_absolute_error")
    single_cv(x=cancel_x, y=cancel_y, model=cancel_model, params={}, cv=cv_result, scoring="accuracy")

# Start CV 
    result = comb_cv(x=(adr_x, cancel_x), y=(adr_y, cancel_y), groups=groups, total_nights=total_nights, \
                    labels_df=labels_df, model=model, cv=cv_result, params={})

# Start re-training
    model = model.fit((adr_x, cancel_x), (adr_y, cancel_y))

# Start testing and Write the result file
    result = dict(model.predict((test_adr_x, test_cancel_x), test_total_nights, test_groups).values)
    df = dataset.test_nolabel_df
    df["label"] = df["arrival_date"].map(result)
    df.to_csv("result.csv", index=False)
