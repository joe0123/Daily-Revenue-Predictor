import numpy as np
import os
import json
import itertools
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import VotingRegressor, VotingClassifier

from dataset import Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

with open("xgb_outputs/ohfeat/adr_opt_s/trials_0.json", 'r') as f:
    all_adr_params = [item["params"] for item in json.load(f)[:10]]
adr_model = VotingRegressor([(str(i), XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", eval_metric="mae", 
                    random_state=0, n_estimators=200, gamma=0, **adr_params)) for i, adr_params in enumerate(all_adr_params)])

with open("xgb_outputs/ohfeat/cancel_opt_s/trials_0.json", 'r') as f:
    all_cancel_params = [item["params"] for item in json.load(f)[:10]]
cancel_model = VotingClassifier([(str(i), XGBClassifier(objective="binary:logistic", eval_metric="error", 
                    tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, use_label_encoder=False,
                    n_estimators=250, gamma=0).set_params(**cancel_params)) for i, cancel_params in enumerate(all_cancel_params)])

model = DailyRevenueEstimator(adr_model, cancel_model)

if __name__ == "__main__":
# Initialization
    dataset = Dataset("./data")
    adr_x, adr_y, test_adr_x = dataset.get_adr_data(onehot_x=True)
    print(adr_x.shape)
    cancel_x, cancel_y, test_cancel_x = dataset.get_cancel_data(onehot_x=True)
    print(cancel_x.shape)
    groups = np.array(dataset.get_groups("train"))
    total_nights = np.array(dataset.get_stay_nights("train"))
    labels_df = dataset.train_label_df
    test_groups = np.array(dataset.get_groups("test"))
    test_total_nights = np.array(dataset.get_stay_nights("test"))
    
    #cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=range(2, 5))
    split_groups = ['-'.join(g.split('-')[:2]) for g in groups]
    cv = sliding_monthly_split(adr_x, split_groups=split_groups, start_group="2016-05", group_window=5, step=2, soft=False)
    cv_result = [i for i in cv]

    #single_cv(x=adr_x, y=adr_y, model=adr_model, params={}, cv=cv_result, scoring="neg_mean_absolute_error")
    #single_cv(x=cancel_x, y=cancel_y, model=cancel_model, params={}, cv=cv_result, scoring="accuracy")
# Start CV 
    result = comb_cv((adr_x, cancel_x), (adr_y, cancel_y), groups, total_nights, labels_df, model, cv_result)

# Start re-training
    model = model.fit((adr_x, cancel_x), (adr_y, cancel_y))

# Start testing and Write the result file
    result = dict(model.predict((test_adr_x, test_cancel_x), test_total_nights, test_groups).values)
    df = dataset.test_nolabel_df
    df["label"] = df["arrival_date"].map(result)
    df.to_csv("result.csv", index=False)
