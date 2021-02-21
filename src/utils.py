import numpy as np
import pandas as pd
import json
import time
import itertools
from functools import partial
from joblib import Parallel, delayed
from sklearn.utils import indexable
from sklearn.utils import shuffle
from sklearn.utils.validation import _num_samples
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._split import _BaseKFold
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import get_scorer

def single_cv(params, x, y, model, cv, scoring, fit_params=dict()):
    start_time = time.time()
    train_scores = []
    valid_scores = []
    model.set_params(**params)
    for train_i, valid_i in cv:
        train_x, train_y = x[train_i], y[train_i]
        valid_x, valid_y = x[valid_i], y[valid_i]
        
        if "early_stopping_rounds" in fit_params: # for xgboost and lightgbm
            fit_params["eval_set"] = [(valid_x, valid_y)]
        model = model.fit(train_x, train_y, **fit_params)
        
        scorer = get_scorer(scoring)
        train_score = scorer(model, train_x, train_y)
        train_scores.append(np.around(train_score, decimals=4))
        valid_score = scorer(model, valid_x, valid_y)
        valid_scores.append(np.around(valid_score, decimals=4))

    dur = np.around(time.time() - start_time, decimals=2)
    result = {"mean_score": np.around(np.mean(valid_scores), decimals=4), \
                "valid_scores": valid_scores, "train_scores": train_scores, "params": params}
    print("\n{} sec -- ".format(dur), result, sep='', flush=True)
    return result

def single_search_cv(x, y, model, params_grid, cv, scoring, fit_params=dict(), \
                    n_iter=None, random_state=0, n_jobs=1):
    params_grid = shuffle(ParameterGrid(params_grid), random_state=random_state, n_samples=n_iter)
    cv_result = [i for i in cv]
    single_cv_ = partial(single_cv, x=x, y=y, model=model, cv=cv_result, \
                            scoring=scoring, fit_params=fit_params)
    results = Parallel(n_jobs=n_jobs)(delayed(single_cv_)(params) for params in params_grid)
    
    return results

def comb_cv(params, x, y, groups, total_nights, labels_df, model, cv):
    start_time = time.time()
    adr_x, cancel_x = x
    adr_y, cancel_y = y
    train_scores = []
    valid_scores = []
    model.set_params(params)
    for train_i, valid_i in cv:
        train_adr_x, train_adr_y = adr_x[train_i], adr_y[train_i]
        valid_adr_x, _ = adr_x[valid_i], adr_y[valid_i]
        train_cancel_x, train_cancel_y = cancel_x[train_i], cancel_y[train_i]
        valid_cancel_x, _ = cancel_x[valid_i], cancel_y[valid_i]

        model = model.fit((train_adr_x, train_cancel_x), (train_adr_y, train_cancel_y))

        train_score = model.score((train_adr_x, train_cancel_x), total_nights[train_i], groups[train_i], labels_df)
        train_scores.append(train_score)
        valid_score = model.score((valid_adr_x, valid_cancel_x), total_nights[valid_i], groups[valid_i], labels_df)
        valid_scores.append(valid_score)
    
    dur = np.around(time.time() - start_time, decimals=2)
    result = {"mean_score": np.around(np.mean(valid_scores), decimals=4), \
                "valid_scores": valid_scores, "train_scores": train_scores, "params": params}
    print("\n{} sec -- ".format(dur), result, sep='', flush=True)
    return result

def comb_search_cv(x, y, groups, total_nights, labels_df, model, params_grid, cv, \
                    n_iter=None, random_state=0, n_jobs=1):
    params_grid = shuffle(ParameterGrid(params_grid), random_state=random_state, n_samples=n_iter)
    cv_result = [i for i in cv]
    comb_cv_ = partial(comb_cv, x=x, y=y, groups=groups, total_nights=total_nights, labels_df=labels_df, model=model, cv=cv_result)
    results = Parallel(n_jobs=n_jobs)(delayed(comb_cv_)(params) for params in params_grid)
    
    return results


class DailyRevenueEstimator(BaseEstimator):
    def __init__(self, adr_model, cancel_model):
        self.adr_model = adr_model
        self.cancel_model = cancel_model

    def fit(self, x, y):
        adr_x, cancel_x = x
        adr_y, cancel_y = y
        self.adr_model = self.adr_model.fit(adr_x, adr_y)
        self.cancel_model = self.cancel_model.fit(cancel_x, cancel_y)
        return self

    def predict(self, x, total_nights, groups):
        adr_x, cancel_x = x
        result_df = predict_daily_revenue(self.adr_model, self.cancel_model, adr_x, cancel_x, total_nights, groups)
        return result_df

    def score(self, x, total_nights, groups, labels_df):
        labels = dict(labels_df.values)
        result_df = self.predict(x, total_nights, groups)
        pred_labels = result_df["label"].to_numpy()
        true_labels = result_df["arrival_date"].map(labels).to_numpy()
        err = np.mean(np.abs(pred_labels - true_labels))
        return err

    def set_params(self, params):
        for (case, param), value in params.items():
            if case == "adr":
                self.adr_model.set_params(**{param: value})
            elif case == "cancel":
                self.cancel_model.set_params(**{param: value})
            else:
                print("Invalid parameter case!")
                exit()

def sliding_monthly_split(X, split_groups, start_group, group_window, step=1, soft=True):
    X, split_groups = indexable(X, split_groups)
    n_samples = len(X)
    indices = np.arange(n_samples)
    group_names, group_counts = np.unique(split_groups, return_counts=True)
    group_ids = np.split(indices, np.cumsum(group_counts)[:-1])
    n_groups = len(group_ids)
    test_start = np.where(group_names == start_group)[0][0]
    test_starts = np.arange(test_start, n_groups, step)
    for test_start in test_starts:
        test_end = test_start + group_window
        if soft or (not soft and test_end <= n_groups):
            test_end = np.clip(test_end, 0, n_groups)
            train_gi = np.arange(0, test_start)
            test_gi = np.arange(test_start, test_end)
            yield (np.concatenate(group_ids[train_gi[0]: train_gi[-1] + 1]),
                    np.concatenate(group_ids[test_gi[0]: test_gi[-1] + 1]))

class GroupTimeSeriesSplit(_BaseKFold):
    """
    Time Series cross-validator for a variable number of observations within the time 
    unit. In the kth split, it returns first k folds as train set and the (k+1)th fold 
    as test set. Indices can be grouped so that they enter the CV fold together.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    """
    def __init__(self, n_splits=5):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None, select_splits=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples and n_features is 
            the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into 
            train/test set.
            Most often just a time feature.

        Yields
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_splits = self.n_splits
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_names, group_counts = np.unique(groups, return_counts=True)
        group_ids = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(group_ids)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater than the number of groups: {1}.").format(n_folds, n_groups))
        test_size = (n_groups // n_folds)
        test_starts = np.arange(test_size + n_groups % n_folds,
                            n_groups, test_size)
        if not select_splits:
            select_splits = range(len(test_starts))
        for test_start in test_starts[select_splits]:
            train_gi = np.arange(0, test_start)
            test_gi = np.arange(test_start, test_start + test_size)
            yield (np.concatenate(group_ids[train_gi[0]: train_gi[-1] + 1]),
                    np.concatenate(group_ids[test_gi[0]: test_gi[-1] + 1]))

def predict_daily_revenue(model_adr, model_cancel, adr_x, cancel_x, total_nights, groups):
    pred_adr = model_adr.predict(adr_x) 
    pred_cancel = model_cancel.predict(cancel_x)
    #pred_cancel = model_cancel.predict_proba(cancel_x)[:, 1]
    result_df = pd.DataFrame({"arrival_date": groups, "label": pred_adr * (1 - pred_cancel) * total_nights})
    result_df = result_df.groupby("arrival_date", as_index=False).agg({"label": "sum"})
    result_df["label"] = result_df["label"].apply(lambda i: np.clip(i // 10000, 0, 9))
    return result_df
