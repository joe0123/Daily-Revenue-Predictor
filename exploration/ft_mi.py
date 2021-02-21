import os
import sys
from sklearn.feature_selection import mutual_info_regression
import numpy as np
import pandas as pd

root = ".."
sys.path.append(root)
from dataset import Dataset
from utils import *

if __name__ == "__main__":
    dataset = Dataset(os.path.join(root, "data"))
    adr_x, adr_y, test_adr_x, ohfeat_cols = dataset.get_adr_data(onehot_x=True)
    df = pd.DataFrame(adr_x, columns=ohfeat_cols)
    mi = mutual_info_regression(df, adr_y, random_state=0)
    mi_df = pd.DataFrame(mi, columns=["mi"], index=ohfeat_cols)
    mi_df.to_excel("adr_mi.xlsx")
    
    dataset = Dataset(os.path.join(root, "data"))
    cancel_x, cancel_y, test_cancel_x, ohfeat_cols = dataset.get_cancel_data(onehot_x=True)
    df = pd.DataFrame(cancel_x, columns=ohfeat_cols)
    mi = mutual_info_regression(df, cancel_y, random_state=0)
    mi_df = pd.DataFrame(mi, columns=["mi"], index=ohfeat_cols)
    mi_df.to_excel("cancel_mi.xlsx")

