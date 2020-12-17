from datasets import raw_dataset

if __name__ == "__main__":
    dataset = raw_dataset("./data", durs={"train": ["2015-07", "2015-09"], "valid": ["2016-07", "2016-10"]})
    train_x, train_y, valid_x, valid_y, test_x = dataset.get_adr_data()
    #train_x, train_y, test_x = dataset.get_adr_data()
    print(train_x.shape, valid_x.shape)
