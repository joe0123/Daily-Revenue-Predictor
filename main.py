from sklearn.linear_model import LinearRegression, Lasso, Ridge

from datasets import raw_dataset

if __name__ == "__main__":
    dataset = raw_dataset("./data", durs={"train": ["2015-10", "2016-09"], "valid": ["2016-10", "2016-12"]})
    train_x, train_y, valid_x, valid_y, test_x = dataset.get_adr_data()
    print(train_x.shape, valid_x.shape)
    
    model = Lasso().fit(train_x, train_y)
    print((1 - model.score(train_x, train_y)) * ((train_y - train_y.mean()) ** 2).mean())
    print((1 - model.score(valid_x, valid_y)) * ((valid_y - valid_y.mean()) ** 2).mean())
