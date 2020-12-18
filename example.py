import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datasets import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset("./data")
    train_x, train_y, test_x = dataset.get_adr_data()
    cv = GroupTimeSeriesSplit(n_splits=5)
    for train_i, valid_i in cv.split(train_x, train_y, groups=dataset.train_group, pass_splits=2):
        print("Train start:", dataset.train_group[train_i[0]])
        print("Train end:", dataset.train_group[train_i[-1]])
        print("Valid start:", dataset.train_group[valid_i[0]])
        print("Valid end:", dataset.train_group[valid_i[-1]], '\n')
