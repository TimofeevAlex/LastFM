import pandas as pd

def train_test_split(ratings, test_frac=0.2, seed=0):
    train = ratings.copy()     
    test = train.groupby('user_email').sample(frac=test_frac)
    train = train.drop(test.index)
    return train, test