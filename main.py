import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from datasets import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset("./data", dur=["2015-07-01", "2016-08-31"])
    train_x, train_y, test_x = dataset.get_adr_data()
    cv = GroupTimeSeriesSplit(n_splits=5)
    for train_i, valid_i in cv.split(train_x, groups=dataset.train_group):
        print(train_i, valid_i)
