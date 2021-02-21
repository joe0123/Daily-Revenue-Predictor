import pandas as pd
import numpy as np
from pandas import Series

df_train = pd.read_csv("../data/train.csv", delimiter=',')
df_test = pd.read_csv("../data/test.csv", delimiter=',')
df_train_label = pd.read_csv("../data/train_label.csv", delimiter=',')
date_id = {}
dates = sorted(list(df_train_label['arrival_date']))
for i in range(len(dates)):
    date_id[dates[i]] = i
adr = [0] * len(dates)
month_dict = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
              'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df_adr = df_train.adr.tolist()
df_year = df_train.arrival_date_year.tolist()
df_month = df_train['arrival_date_month'].map(month_dict).tolist()
df_day = df_train.arrival_date_day_of_month.tolist()
df_weekends = df_train.stays_in_weekend_nights.to_numpy()
df_weekdays = df_train.stays_in_week_nights.to_numpy()
total_days = df_weekdays + df_weekends
df_is_cancelled = df_train.is_canceled.tolist()
for i in range(len(df_day)):
    date = '{}-{:02d}-{:02d}'.format(df_year[i], df_month[i], df_day[i])
    if df_is_cancelled[i] != 1:
        adr[date_id[date]] += df_adr[i] * total_days[i]

s1 = Series(adr)
s2 = Series(list(df_train_label['label']))
print('\n')
for i in range(10):
    print("Label = {}: min revenue = {}, max revenue = {}".format(i, s1[s2 == i].min(), s1[s2 == i].max()))
print("Correlation coefficient = {}".format(s1.corr(s2)))
print('\n')
