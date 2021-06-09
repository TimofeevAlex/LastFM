# The function for train-test split which is chosen as
# a validation method since the task is computationally-intensive
import numpy as np
import pandas as pd
from tqdm import tqdm

def train_test_split(ratings, test_frac=0.1):
    '''
    Samples test_frac percents of interactions from each user for the test set
    Parameters:
        - ratings: pandas DataFrame, a set of interactions
        - test_frac: float, percentage of samples for the 
            test set. Should be in (0, 1)
    '''
    train = ratings.copy()     
    test = train.groupby('user_email').sample(frac=test_frac)
    train = train.drop(test.index)
    return train, test

            