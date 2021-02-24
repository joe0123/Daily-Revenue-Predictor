# Revenue-Predictor

Machine Learning Techniques final project

Instructor: 
1. Hsuan-Tien Lin (林軒田)

Members: 
1. Kai-Jo Ma (馬愷若)
2. Bo-Wei Huang (黃柏瑋)

## Task description
https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/project/
> A hotel booking company tries to predict the daily revenue of the company from the data of reservation. In
particular, after a room reservation request is fulfilled (i.e. no cancellation), on the arrival date, the
revenue of the request is the rate of the room (called ADR) multiplied by the number of days that the
customer is going to stay in the room, and the daily revenue is the sum of all those fulfilled requests on
the same day. The goal of the prediction is to accurately infer the future daily revenue of the company,
where the daily revenue is quantized to 10 scales.
Now, having collected some data from the hotel booking service, the CEO of the company decides to
ask you, a rising star in the data science team, to help with the prediction task. You need to fight for
the most accurate prediction on the score board. Then, you need to submit a comprehensive report that
describes not only the recommended approaches, but also the reasoning behind your recommendations.

## Report
https://github.com/joe0123/HTML2020_final/blob/master/Report.pdf

## Requirement
* numpy == 1.20.0
* pandas == 1.2.1
* joblib == 1.0.0
* scikit-learn == 0.24.0
* xgboost == 1.3.1
* optuna == 2.3.0

## Usage for exploration/
Find the relationship between labels and revenues:
```
python task.py
```
Find the feature importance with respect to adr and is_canceled.
```
python ft_mi.py
```

## Usage for src/
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
In the appendix of our report, we did some experiments about feature selection for agency and country. To train and make inference with selected features, change *data/.adr_drop.txt* and *data/.cancel_drop.txt* to *data/adr_drop.txt* and *data/cancel_drop.txt*, respectively; then, run the commands mentioned above again.
