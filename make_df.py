import pandas as pd

def prepare_df(pos, neg, path):
    data_dict = {"labels": [ 'pos' for _ in pos ] + ['neg' for _ in neg],
                "tweets": [tweet for tweet in pos] + [tweet for tweet in neg]}
    df = pd.DataFrame(data_dict)
    df.to_csv(path, index=False)


def prepare_test_set():
    pos_test = open("positive_test.txt", "r").read().split("\n")
    neg_test = open("negative_test.txt", "r").read().split("\n")
    prepare_df(pos_test, neg_test, "test_df.csv")

def prepare_train_set():
    pos_train = open("positive_train.txt","r").read().split("\n")
    neg_train = open("negative_train.txt", "r").read().split("\n")
    prepare_df(pos_train, neg_train, "train_df.csv")

prepare_train_set()
prepare_test_set()