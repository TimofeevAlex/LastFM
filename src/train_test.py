from tqdm import tqdm
import pandas as pd

def train_test_split(ratings, test_frac=0.2, seed=0):
    train = ratings.copy()     
    test = train.groupby('user_email').sample(frac=test_frac)
    train = train.drop(test.index)
    return train, test

def get_new_artists(train, artists):
    ratings = pd.DataFrame(columns=['user_email', 'artist_id', 'pred_plays'])
    for user, user_df in tqdm(train.groupby('user_email')):
        new_artists = list(set(artists) - set(user_df['artist_id']))
        ratings = ratings.append({'user_email': [user]*len(new_artists), 'artist_id': len(new_artists), 
                        'pred_plays':[-1]*len(new_artists)}, ignore_index=True)
    ratings.to_csv('lastfm-dataset-360K/new_artists.csv', index=False)