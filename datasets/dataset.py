import os
import pandas as pd


class Dataset:
    def __init__(self, data_dir, drop_cols, durs):
        self.data_dir = data_dir
        self.load_df()
        self.train_dur, self.valid_dur = durs.get("train"), durs.get("valid")
        self.split_df()
        self.drop_cols = drop_cols
        self.create_feats()

    def load_df(self):
        self.train_raw_df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))
        self.train_label_df = pd.read_csv(os.path.join(self.data_dir, "train_label.csv"))
        self.test_raw_df = pd.read_csv(os.path.join(self.data_dir, "test.csv"))
        self.test_nolabel_df = pd.read_csv(os.path.join(self.data_dir, "test_nolabel.csv"))
    
    def _is_fit_date(self, df, dur):
        if len(df.columns) == 2:
            month_int = dict(zip(["January", "February", "March", "April", "May", "June", "July", "August", \
                    "September", "October", "November", "December"], ["{:02d}".format(i) for i in range(1, 13)]))
            date_df = df.iloc[:, 0].astype("str").str.cat(df.iloc[:, 1].apply(month_int.get), sep='-')
        else:
            date_df = df.iloc[:, 0].apply(lambda s: '-'.join(s.split('/')[:2]))

        return (date_df >= dur[0]) & (date_df <= dur[1])


    def split_df(self):
        if self.train_dur:
            train_raw_df = self.train_raw_df[self._is_fit_date(self.train_raw_df[["arrival_date_year", "arrival_date_month"]], \
                                    self.train_dur)]
            train_label_df = self.train_label_df[self._is_fit_date(self.train_label_df[["arrival_date"]], \
                                    self.train_dur)]
        else:
            train_raw_df = self.train_raw_df
            train_label_df = self.train_raw_df
        
        if self.valid_dur:
            valid_raw_df = self.train_raw_df[self._is_fit_date(self.train_raw_df[["arrival_date_year", "arrival_date_month"]], \
                                    self.valid_dur)]
            valid_label_df = self.train_label_df[self._is_fit_date(self.train_label_df[["arrival_date"]], \
                                    self.valid_dur)]
        else:
            print("Note: No valid data selected", flush=True)

        self.train_raw_df = train_raw_df
        self.train_label_df = train_label_df
        if self.valid_dur:
            self.valid_raw_df = valid_raw_df
            self.valid_label_df = valid_label_df

    
    def _build_feats(self, df):
        df = df[self.test_raw_df.columns]
        df = df.drop(columns=self.drop_cols, errors="ignore")
        if "agent" in df:
            df["agent"] = df["agent"].astype("object")
        if "children" in df:
            df["children"] = df["children"].fillna(value=0)
        df = pd.get_dummies(df)

        return df

    def create_feats(self):
        train_feat_df = self._build_feats(self.train_raw_df)
        test_feat_df = self._build_feats(self.test_raw_df)
        cols = sorted(train_feat_df.columns)   #TODO intersect valid?
        self.train_feat_df = train_feat_df.reindex(cols, fill_value=0, axis=1)
        self.test_feat_df = test_feat_df.reindex(cols, fill_value=0, axis=1)
        if self.valid_dur:
            valid_feat_df = self._build_feats(self.valid_raw_df)
            self.valid_feat_df = valid_feat_df.reindex(cols, fill_value=0, axis=1)

