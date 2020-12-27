from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from datasets import *
from utils import *

# Initialization
dataset = Dataset("./data")
cancel_x, cancel_y, test_cancel_x = dataset.get_cancel_data()
groups = np.array(dataset.get_groups("train"))

model = Pipeline([("feature_selection", SelectFromModel(LogisticRegression(C=1e+3, penalty="l1", solver="liblinear", \
                                                                random_state=0, max_iter=1e+8))), \
                    ("classifier", LogisticRegression(C=1e+2, max_iter=1e+8))])

params = {"feature_selection__estimator__C": [1e+2, 1e+1, 1, 1e-1], \
            "classifier__C": [1e+3, 1e+2, 1e+1]}

cv = GroupTimeSeriesSplit(n_splits=5).split(adr_x, groups=groups, select_splits=[2], return_group_i=False)
clf = GridSearchCV(model, params, cv=cv).fit(adr_x, adr_y)
print(clf.best_score_)
