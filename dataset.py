import os
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class Dataset:
    def __init__(self, data_dir, drop_cols=["ID", "company"], dur=None):
        self.data_dir = data_dir
        self.load_df()
        if dur:
            self.cut_df(dur)
        self.create_feats(drop_cols)
        self.create_groups()

    def load_df(self):
        self.train_raw_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.train_label_df = pd.read_csv(os.path.join(self.data_dir, "train_label.csv"))
        self.test_raw_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.test_nolabel_df = pd.read_csv(os.path.join(self.data_dir, "test_nolabel.csv"))
    
    def _build_date_group(self, df):
        date_group = df["arrival_date_year"].astype("str").str.cat(  \
                    [df["arrival_date_month"].apply(lambda i: "{:02d}".format(i)), 
                    df["arrival_date_day_of_month"].apply(lambda i: "{:02d}".format(i))], sep='-').tolist()
        assert all(date_group[i] <= date_group[i + 1] for i in range(len(date_group) - 1))
        
        return date_group

    def create_groups(self):
        self.train_group = self._build_date_group(self.train_feat_df)
        self.test_group = self._build_date_group(self.test_feat_df)

    def _is_fit_date(self, date_df, dur):
        date_df = pd.to_datetime(date_df)
        dur = pd.to_datetime(dur)
        
        return (date_df >= dur[0]) & (date_df <= dur[1])

    def cut_df(self, dur):
        self.train_group = self.train_group[self._is_fit_date(self.train_group, dur)]
        self.train_raw_df = self.train_raw_df.iloc[:len(self.train_group)]
        self.train_label_df = self.train_label_df[self._is_fit_date(self.train_label_df["arrival_date"], dur)]

    def _build_feats(self, df, drop_cols):
        df = df[self.test_raw_df.columns]
        df = df.drop(columns=drop_cols, errors="ignore")
        if "arrival_date_month" in df:
            month_int = dict(zip(["January", "February", "March", "April", "May", "June", "July", "August", \
                "September", "October", "November", "December"], range(1, 13)))
            df["arrival_date_month"] = df["arrival_date_month"].apply(month_int.get)
        if "agent" in df:
            df["agent"] = df["agent"].fillna(value=0)
            df["agent"] = df["agent"].astype("object")
        if "children" in df:
            df["children"] = df["children"].fillna(value=0)
        if "country" in df:
            df["country"] = df["country"].fillna(value="others")
        
        return df, pd.get_dummies(df)

    def _align_feats(self, train_df, test_df):
        feat_cols = sorted(train_df.columns)
        #feat_cols = sorted(list(set(train_df.columns) & set(test_df.columns)))
        train_df = train_df.reindex(feat_cols, fill_value=0, axis=1)
        test_df = test_df.reindex(feat_cols, fill_value=0, axis=1)
        
        return train_df, test_df, feat_cols

    def create_feats(self, drop_cols):
        train_feat_df, train_ohfeat_df = self._build_feats(self.train_raw_df, drop_cols)
        test_feat_df, test_ohfeat_df = self._build_feats(self.test_raw_df, drop_cols)
        self.train_feat_df, self.test_feat_df, self.feat_cols = self._align_feats(train_feat_df, test_feat_df)
        self.train_ohfeat_df, self.test_ohfeat_df, self.ohfeat_cols = self._align_feats(train_ohfeat_df, test_ohfeat_df)

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        for col in train_feat_df.select_dtypes(exclude="number").columns:
            self.train_feat_df[col] = encoder.fit_transform(train_feat_df[col].to_numpy().reshape(-1, 1)).reshape(-1)
            self.test_feat_df[col] = encoder.transform(test_feat_df[col].to_numpy().reshape(-1, 1)).reshape(-1)

    def get_adr_data(self, onehot_x=False):
        if onehot_x:
            train_x = self.train_ohfeat_df
            test_x = self.test_ohfeat_df
        else:
            train_x = self.train_feat_df
            test_x = self.test_feat_df
        train_y = self.train_raw_df["adr"]
        
        return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy()
 
    def get_cancel_data(self, onehot_x=False):
        if onehot_x:
            train_x = self.train_ohfeat_df
            test_x = self.test_ohfeat_df
        else:
            train_x = self.train_feat_df
            test_x = self.test_feat_df
        train_y = self.train_raw_df["is_canceled"]
        
        return train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy()
 
    def get_groups(self, case):
        return getattr(self, "{}_group".format(case))
    
    def get_stay_nights(self, case):
        df = getattr(self, "{}_raw_df".format(case))
        return (df.stays_in_weekend_nights.to_numpy() + df.stays_in_week_nights.to_numpy()).tolist()
