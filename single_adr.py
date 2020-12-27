from sklearn.linear_model import Lasso, Ridge, HuberRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from datasets import *
from utils import *

# Initialization
dataset = Dataset("./data")
adr_x, adr_y, test_adr_x = dataset.get_adr_data()
groups = np.array(dataset.get_groups("train"))

models = [Lasso(max_iter=1e+8, tol=1e-6),
        Ridge(max_iter=1e+8, tol=1e-6),
        HuberRegressor(max_iter=1e+8, tol=1e-6),
        Pipeline([("feature_selection", SelectFromModel(Lasso(max_iter=1e+8, tol=1e-6))), \
                    ("regression", Ridge(max_iter=1e+8, tol=1e-6))]),
        Pipeline([("feature_selection", SelectFromModel(Lasso(max_iter=1e+8, tol=1e-6))), \
                    ("regression", HuberRegressor(max_iter=1e+8, tol=1e-6))])
]

params = [{"alpha": [1e-1, 1e-2, 1e-3]},
        {"alpha": [1e-1, 1e-2, 1e-3]},
        {"epsilon": [1.35, 1.5, 1.75], "alpha": [1e-1, 1e-2, 1e-3]},
        {"feature_selection__estimator__alpha": [1e-1, 1e-2, 1e-3], \
            "regression__alpha": [1e-1, 1e-2, 1e-3]},
        {"feature_selection__estimator__alpha": [1e-1, 1e-2, 1e-3], \
            "regression__epsilon": [1.35, 1.5, 1.75], \
            "regression__alpha": [1e-1, 1e-2, 1e-3]}
]

# Start grid search
for model, params_ in zip(models, params):
    print(model, flush=True)
    cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=[2], return_group_i=False)
    clf = GridSearchCV(model, params_, scoring="neg_mean_absolute_error", cv=cv).fit(adr_x, adr_y)
    print(-clf.best_score_, clf.best_params_, flush=True)
    print('\n')
