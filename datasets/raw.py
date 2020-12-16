import os
import pandas as pd

from .dataset import Dataset

class Raw_Dataset(Dataset):
    def __init__(self, data_dir):
       super(Raw_Dataset, self).__init__(data_dir)

    def get_adr_data(self, numpy=True):
        train_x, train_y = self.train_feat_df, self.train_raw_df["adr"]
        test_x = self.test_feat_df
        if numpy:
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            test_x = test_x.to_numpy()

        return train_x, train_y, test_x
    
    def get_cancel_data(self, numpy=True):
        train_x, train_y = self.train_feat_df, self.train_raw_df["is_canceled"]
        test_x = self.test_feat_df
        if numpy:
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            test_x = test_x.to_numpy()

        return train_x, train_y, test_x

