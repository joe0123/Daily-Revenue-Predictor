from datasets import raw_dataset

if __name__ == "__main__":
    dataset = raw_dataset("./data")
    train_x, train_y, test_x = dataset.get_adr_data()
