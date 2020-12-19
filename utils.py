import numpy as np
import pandas as pd
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _BaseKFold

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

    def split(self, X, y=None, groups=None, pass_splits=0):
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
        pass_splits : int
            Pass first n splits.

        Yields
        -------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        n_splits = self.n_splits
        if pass_splits >= n_splits:
            raise ValueError(
                ("Note that pass_splits = {0} must smaller than the number of splits: {1}.").format(pass_splits, n_splits))
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_folds = n_splits + 1
        indices = np.arange(n_samples)
        group_counts = np.unique(groups, return_counts=True)[1]
        groups = np.split(indices, np.cumsum(group_counts)[:-1])
        n_groups = _num_samples(groups)
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds ={0} greater than the number of groups: {1}.").format(n_folds, n_groups))
        test_size = (n_groups // n_folds)   # TODO
        test_starts = range(test_size + n_groups % n_folds,
                            n_groups, test_size)
        for test_start in test_starts[pass_splits:]:
            yield (np.concatenate(groups[:test_start]),
                    np.arange(0, test_start),
                    np.concatenate(groups[test_start:test_start + test_size]),
                    np.arange(test_start, test_start + test_size))

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
