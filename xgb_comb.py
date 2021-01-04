import numpy as np
import os
from tqdm import tqdm
import itertools
from xgboost import XGBRegressor, XGBClassifier

from dataset import Dataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = '3'

model_adr = XGBRegressor(tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, \
                        n_estimators=200, learning_rate=1e-1, min_child_weight=5, max_depth=5, gamma=0, \
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=1, reg_alpha=0)

model_cancel = XGBClassifier(objective="binary:logistic", eval_metric="error", \
                            tree_method="gpu_hist", predictor="gpu_predictor", random_state=0, use_label_encoder=False, \
                            n_estimators=300, learning_rate=0.2, min_child_weight=2, max_depth=2, gamma=12, \
                            subsample=0.8, colsample_bytree=0.8, reg_lambda=0.01, reg_alpha=1)
    
model = DailyRevenueEstimator(model_adr, model_cancel)

weight_ratio = 1

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
    
    cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=range(2, 5))

# Start CV
    result = comb_cv(adr_x, adr_y, cancel_x, cancel_y, groups, total_nights, labels, model, cv, weight_ratio)    

# Start re-training
    if weight_ratio != 1:
        focus = find_focus(groups, test_groups)
        model = model.fit((adr_x, cancel_x), (adr_y, cancel_y), sample_weight=np.where(focus, weight_ratio, 1))
    else:
        model = model.fit((adr_x, cancel_x), (adr_y, cancel_y))

# Start testing and Write the result file
    result = model.predict((test_adr_x, test_cancel_x), test_total_nights, test_groups)
    df = dataset.test_nolabel_df
    df["label"] = result
    df.to_csv("result.csv", index=False)
