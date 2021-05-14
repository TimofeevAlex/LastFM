import numpy as np
import pandas as pd
from tqdm import tqdm

def train_test_split(ratings, test_frac=0.2, seed=0):
    train = ratings.copy()     
    test = train.groupby('user_email').sample(frac=test_frac)
    train = train.drop(test.index)
    return train, test

def get_new_artists(train, artists):
    first = True
    for user, user_df in tqdm(train.groupby('user_email')):
        new_artists = list(set(artists) - set(user_df['artist_id']))
        ratings = pd.DataFrame({'user_email': [user]*len(new_artists), 'artist_id': len(new_artists), 
                        'pred_plays':[-1]*len(new_artists)}).astype(np.int32)
        if first:
            ratings.to_csv('lastfm-dataset-360K/new_artists.csv', mode='a', header=True, index=False)
        else:
            ratings.to_csv('lastfm-dataset-360K/new_artists.csv', mode='a', header=False, index=False)
            