import os
import pandas as pd

from .dataset import Dataset

class Raw_Dataset(Dataset):
    def __init__(self, data_dir, drop_cols=["ID", "arrival_date_year", "agent", "company", "country"], durs=dict()):
        super(Raw_Dataset, self).__init__(data_dir, drop_cols, durs)

    def get_adr_data(self, numpy=True):
        train_x, train_y = self.train_feat_df, self.train_raw_df["adr"]
        if self.valid_dur:
            valid_x, valid_y = self.valid_feat_df, self.valid_raw_df["adr"]
        test_x = self.test_feat_df
        
        if numpy:
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            if self.valid_dur:
                valid_x = valid_x.to_numpy()
                valid_y = valid_y.to_numpy()
            test_x = test_x.to_numpy()

        if self.valid_dur:
            return train_x, train_y, valid_x, valid_y, test_x
        else:
            return train_x, train_y, test_x

    
    def get_cancel_data(self, numpy=True):
        train_x, train_y = self.train_feat_df, self.train_raw_df["is_canceled"]
        if self.valid_dur:
            valid_x, valid_y = self.valid_feat_df, self.valid_raw_df["is_canceled"]
        test_x = self.test_feat_df
        
        if numpy:
            train_x = train_x.to_numpy()
            train_y = train_y.to_numpy()
            if self.valid_dur:
                valid_x = valid_x.to_numpy()
                valid_y = valid_y.to_numpy()
            test_x = test_x.to_numpy()

        if self.valid_dur:
            return train_x, train_y, valid_x, valid_y, test_x
        else:
            return train_x, train_y, test_x

