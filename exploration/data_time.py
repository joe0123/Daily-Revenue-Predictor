import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datasets import raw_dataset

if __name__ == "__main__":
    dataset1 = raw_dataset("./data", durs={"train": ["2015-07", "2016-03"], "valid": ["2016-04", "2016-05"]})
    train_x, train_y, valid_x, valid_y, _ = dataset1.get_adr_data()
    x = np.concatenate((train_x, valid_x), axis=0)
    y = np.concatenate((train_y, valid_y), axis=0)
    dataset2 = raw_dataset("./data", durs={"train": ["2015-07", "2016-03"], "valid": ["2016-06", "2016-08"]})
    _, _, test_x, test_y, _ = dataset2.get_adr_data()
    
    for alpha in [1, 1e-1, 1e-2, 1e-3, 1e-4]:
        print(alpha)
        model = Lasso(alpha=alpha, max_iter=1e+5, random_state=1114).fit(train_x, train_y)
        print((1 - model.score(valid_x, valid_y)) * ((valid_y - valid_y.mean()) ** 2).mean())
        model = Lasso(alpha=alpha, max_iter=1e+5, random_state=1114).fit(x, y)
        print((1 - model.score(test_x, test_y)) * ((test_y - test_y.mean()) ** 2).mean())
    
    train_x, train_y, valid_x, valid_y, _ = dataset1.get_cancel_data()
    x = np.concatenate((train_x, valid_x), axis=0)
    y = np.concatenate((train_y, valid_y), axis=0)
    _, _, test_x, test_y, _ = dataset2.get_cancel_data()
    for c in [1e-2, 1e-1, 1, 10, 100]:
        print(c)
        model = LogisticRegression(C=c, max_iter=1e+5, random_state=1114).fit(train_x, train_y)
        print(model.score(train_x, train_y))
        print(model.score(valid_x, valid_y))
        model = LogisticRegression(C=c, max_iter=1e+5, random_state=1114).fit(x, y)
        print(model.score(test_x, test_y))
