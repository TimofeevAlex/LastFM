# Here are given functions to transform number of plays to ratings
# as well as to negative samples based on the given interactions set
import numpy as np
import pandas as pd

def to_ratings(user_df):
    '''
    Transform play frequencies to ratings for a given user.
    The example of usage:
    
    lastfm_360_behav.groupby('user_email').apply(to_ratings)
    
    '''
    user_df = user_df.sort_values('norm_plays', ascending=False) 
    cumsum = user_df['norm_plays'].cumsum() 
    user_df['norm_plays'] = 4 * (1 - cumsum)
    return user_df

def build_get_negative_ratings(all_artists, factor=1):
    '''
    Build a function that returns a set of unlistened to artists
    for each user.
    Parameters:
        - all_artists: list, the list of all artists in the dataset
        - factor: int, how many unlistened to artists should be added
    The example of usage:
    
    all_artists = set(ratings['artist_id'].unique())
    get_negative_ratings = build_get_negative_ratings(all_artists, factor=10)
    negative_ratings = ratings.groupby('user_email').progress_apply(get_negative_ratings)
   
   '''
    def get_negative_ratings(user_df):
        user_interactions = user_df['artist_id']
        num_interactions = user_df['artist_id'].shape[0]
        new_artists = list(all_artists - set(user_df['artist_id']))
        new_artists = np.random.choice(new_artists, num_interactions * factor, replace=False)
        return pd.DataFrame({'artist_id': new_artists, 'rating': [0] * new_artists})
    return get_negative_ratings