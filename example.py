import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset
from utils import *

if __name__ == "__main__":
# Initialize dataset
    dataset = Dataset("./data")
    
# Train adr model (regression)
    train_x, train_y, test_x = dataset.get_adr_data()
    model_adr = Lasso(alpha=0.01, max_iter=1e+5).fit(train_x, train_y)
    train_x, train_y, test_x = dataset.get_cancel_data()
    
# Train is_cancel model (classification)
    model_cancel = LogisticRegression(max_iter=1e+5).fit(train_x, train_y)
    pred_adr = model_adr.predict(test_x) 
    pred_cancel = model_cancel.predict(test_x)
    
# Calculate one-day revenue
    total_nights = dataset.get_stay_nights("test")
    result = group_sum(pred_adr * (1 - pred_cancel) * total_nights, dataset.get_groups("test"))
    result = [np.clip(i // 10000, 0, 9) for i in result]

# Write the result file
    df = dataset.test_nolabel_df
    df["label"] = result
    df.to_csv("result.csv", index=False)
