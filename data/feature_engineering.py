import numpy as np
import pandas as pd

def count_dow_nights(base, total_nights):
    result = np.ones(7) * (total_nights // 7)
    for i in range(total_nights % 7):
        result[(base + i) % 7] += 1
    
    return result


if __name__ == "__main__":
    for case in ["train", "test"]:
        df_ = pd.read_csv("./{}.csv".format(case))
        month_int = dict(zip(["January", "February", "March", "April", "May", "June", "July", "August", \
                    "September", "October", "November", "December"], range(1, 13)))
        df_["arrival_date_month_int"] = df_["arrival_date_month"].apply(month_int.get)
        full_date = df_["arrival_date_year"].astype("str").str.cat(  \
                    [df_["arrival_date_month_int"].apply(lambda i: "{:02d}".format(i)),
                    df_["arrival_date_day_of_month"].apply(lambda i: "{:02d}".format(i))], sep='-').tolist()

        df_["full_date"] = full_date
        df_["full_date"] = pd.to_datetime(df_["full_date"])
        df_["year_month"] = df_["arrival_date_year"].astype(str) + df_["arrival_date_month_int"].astype(str)
        df_["year_week"] = df_["arrival_date_year"].astype(str) + df_["arrival_date_week_number"].astype(str)
        df_["order_count"] = 1
        #df_.groupby("full_date", as_index=False)["order_count"].count()
        #df_['order_count'] = df_.groupby('full_date').transform('order_count')
        df_["day_order_count"] = df_.groupby('full_date')["order_count"].transform("count")
        df_["week_order_count"] = df_.groupby("year_week")["order_count"].transform("count")
        df_["month_order_count"] = df_.groupby("year_month")["order_count"].transform("count")
        df_["arrival_day_of_week"] = df_["full_date"].dt.dayofweek
        #df_["total_stay_nights"] = df_["stays_in_week_nights"] + df_["stays_in_weekend_nights"]
        #df_[["stays_in_{}_nights".format(s) for s in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]] = \
        #        df_.apply(lambda r: count_dow_nights(r["arrival_day_of_week"], r["total_stay_nights"]), \
        #                    result_type="expand", axis=1)

        df_ = df_.drop(["arrival_date_month_int", "full_date", "year_month", "year_week", "order_count"], axis=1)
        df_.to_csv("{}_new.csv".format(case), index=False)
        #print(df_)
