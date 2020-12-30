import numpy as np
import pandas as pd
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

def weighted(base, weight_ratio):
    class WeightedEstimator(base):
        def fit(self, X, y, focus, weight_ratio=weight_ratio):
            return super(base, self).fit(X, y, sample_weight=np.where(focus, weight_ratio, 1))
    
    return WeightedEstimator

def _single_cv(params, x, y, groups, model, cv, scoring):
    errs = []
    model.set_params(**params)
    for train_i, (train_gi, _), valid_i, (valid_gi, valid_gn) in cv:
        train_x, train_y = x[train_i], y[train_i]
        valid_x, valid_y = x[valid_i], y[valid_i]
        if "WeightedEstimator" == type(model).__name__:
            focus_months = set([int(n.split('-')[1]) for n in valid_gn])
            focus = [int(groups[i].split('-')[1]) in focus_months for i in train_i]
            model.fit(train_x, train_y, focus)
        else:
            model = model.fit(train_x, train_y)

        err = get_scorer(scoring)(model, valid_x, valid_y)
        errs.append(err)
    
    return {"mean_score": np.mean(errs), "scores": errs, "params": params}

def single_search_cv(x, y, groups, model, params_grid, cv, scoring, n_iter=None, random_state=0, n_jobs=1):
    params_grid = shuffle(ParameterGrid(params_grid), random_state=random_state, n_samples=n_iter)
    cv_result = [i for i in cv]
    single_cv = partial(_single_cv, x=x, y=y, groups=groups, model=model, cv=cv_result, scoring=scoring)
    record = Parallel(n_jobs=n_jobs, verbose=50)(delayed(single_cv)(params) for params in params_grid)
    
    return record

class DailyRevenueEstimator(BaseEstimator):
    def __init__(self, adr_model, cancel_model):
        self.adr_model = adr_model
        self.cancel_model = cancel_model

    def fit(self, adr_x, adr_y, cancel_x, cancel_y):
        self.adr_model = self.adr_model.fit(adr_x, adr_y)
        self.cancel_model = self.cancel_model.fit(cancel_x, cancel_y)
        return self

    def predict(self, adr_x, cancel_x, total_nights, groups):
        result = predict_daily_revenue(self.adr_model, self.cancel_model, adr_x, cancel_x, total_nights, groups)
        return result

    def score(self, adr_x, cancel_x, y, total_nights, groups):
        result = self.predict(adr_x, cancel_x, total_nights, groups)
        err = np.mean(np.abs(result - y))
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
                    (train_gi, group_names[train_gi]),
                    np.concatenate(group_ids[test_gi[0]: test_gi[-1] + 1]),
                    (test_gi, group_names[test_gi]))


def group_sum(a, groups):
    assert len(a) == len(groups)
    a = pd.DataFrame({"arr": a, "group": groups})
    return a.groupby("group").sum()["arr"].tolist()

def predict_daily_revenue(model_adr, model_cancel, adr_x, cancel_x, total_nights, groups):
    pred_adr = model_adr.predict(adr_x) 
    pred_cancel = model_cancel.predict(cancel_x)
    #pred_cancel = model_cancel.predict_proba(cancel_x)[:, 1]
    result = group_sum(pred_adr * (1 - pred_cancel) * total_nights, groups)
    result = np.array([np.clip(i // 10000, 0, 9) for i in result])
    return result
