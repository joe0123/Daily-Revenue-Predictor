import numpy as np
from tqdm import tqdm
import itertools
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression, Lasso, Ridge, HuberRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.pipeline import Pipeline

from datasets import Dataset
from utils import *


if __name__ == "__main__":
    dataset = Dataset("./data")
    adr_x, adr_y, _ = dataset.get_adr_data()
    print(adr_x.shape)
    exit()
    cols = dataset.train_feat_df.columns.tolist()
    
    model = RandomForestRegressor(n_estimators=800, min_samples_split=10, min_samples_leaf=1, max_features="auto", max_depth=40, criterion='mse', random_state=0, n_jobs=-1)
    
