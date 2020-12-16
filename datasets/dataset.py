import os
import pandas as pd

class Dataset:
    def __init__(self, data_dir, drop_cols):
        self.data_dir = data_dir
        self.load_df()
        self.drop_cols = drop_cols
        self.create_feats()

    def load_df(self):
        self.train_raw_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.train_label_df = pd.read_csv(os.path.join(self.data_dir, "train_label.csv"))
        self.test_raw_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.test_nolabel_df = pd.read_csv(os.path.join(self.data_dir, "test_nolabel.csv"))
    
    def _build_feats(self, df):
        df = df[self.test_raw_df.columns]
        df = df.drop(columns=self.drop_cols, errors="ignore")
        if "agent" in df:
            df["agent"].astype("object")
        if "children" in df:
            df["children"] = df["children"].fillna(value=0)
        df = pd.get_dummies(df)

        return df

    def create_feats(self):
        train_feat_df = self._build_feats(self.train_raw_df)
        test_feat_df = self._build_feats(self.test_raw_df)
        cols = sorted(list(set(train_feat_df.columns) | set(test_feat_df.columns)))
        self.train_feat_df = train_feat_df.reindex(cols, fill_value=0, axis=1)
        self.test_feat_df = test_feat_df.reindex(cols, fill_value=0, axis=1)
        assert self.train_feat_df.columns.equals(self.test_feat_df.columns)
