# Machine Learning Techniques final project

Instructor: 
1. Hsuan-Tien Lin (林軒田)

Members: 
1. Kai-Jo Ma (馬愷若)
2. Bo-Wei Huang (黃柏瑋)

## Task description
https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/project/

## Report
https://github.com/joe0123/HTML2020_final/blob/master/Report.pdf

## Requirement
* numpy == 1.20.0
* pandas == 1.2.1
* joblib == 1.0.0
* scikit-learn == 0.24.0
* xgboost == 1.3.1
* optuna == 2.3.0

## Usage
### Linear Model
Find optimal parameters automatically (by random search):
```
python lin_comb.py
```
Train and make inference with optimal parameters found by random search:
```
python lin_infer.py
```

### Random Forest
Find optimal parameters automatically (by random search):
```
python rf_adr.py
python rf_cancel.py
```
Train and make inference with optimal parameters found by random search:
```
python rf_infer.py
```

### XGBoost
Find optimal parameters manually. Note that you should open the file and freeze/release the parameters by yourself:
```
python xgb_adr.py
python xgb_cancel.py
```
Train and make inference with optimal parameters found manaully:
```
python xgb_infer.py
```
Find optimal parameters automatically (by Optuna):
```
python xgb_adr_opt.py
python xgb_cancel_opt.py
```
Train and make inference with optimal parameters found by Optuna:
```
python xgb_infer_opt.py
```

### Futher
In the appendix of our report, we did some experiments about feature selection for agency and country. To train and make inference with selected features, change *.adr_drop.txt* and *.cancel_drop.txt* to *adr_drop.txt* and *cancel_drop.txt*, respectively; then, run the commands mentioned above again.
