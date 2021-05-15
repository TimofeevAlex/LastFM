import numpy as np
import pandas as pd
from tqdm import tqdm

def train_test_split(ratings, test_frac=0.2, seed=0):
    train = ratings.copy()     
    test = train.groupby('user_email').sample(frac=test_frac)
    train = train.drop(test.index)
    return train, test

# def get_new_artists(train, artists):
#     first = True
#     artists_set = set(artists)
#     for user, user_df in tqdm(train.groupby('user_email')):
#         new_artists = list(artists_set - set(user_df['artist_id']))
#         num_artists = int(0.75 * len(new_artists))
#         new_artists = np.random.choice(new_artists, num_artists, replace=False)
#         ratings = pd.DataFrame({'user_email': [user]*num_artists, 'artist_id': new_artists}).astype(np.int32)
#         if first:
#             ratings.to_csv('lastfm-dataset-360K/new_artists.csv', mode='a', header=True, index=False)
#             first=False
#         else:
#             ratings.to_csv('lastfm-dataset-360K/new_artists.csv', mode='a', header=False, index=False)
            