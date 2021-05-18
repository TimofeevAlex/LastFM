import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

def precision_recall_at_k(test, pred_ratings, k=10):
    '''
    - pred_ratings: dict
        Keys are users and values are arrays of shape (2, number of items)
    '''
    precisions = []
    recalls = []
    for (user, user_test) in test.groupby('user_email'):
        user_ratings = pred_ratings[user]
        ranked_artists = user_ratings[0, np.argsort(-user_ratings[1])].astype(np.int32)[:k]
        top_k = set(ranked_artists)
        test_artists = set(user_test.artist_id)
        precisions.append(len(top_k & test_artists) / float(k))
        recalls.append(len(top_k & test_artists) / len(test_artists))
    return precisions, recalls

def hit_rate_user(user, top_recs, test):
    '''
    - top_recs: list of top reccommendations for the given user
    '''
    num_hits = 0
    df_user = test.loc[test['user_email'] == user]
   # print('user = {}'.format(user))
    for artist in top_recs:
     # print('artist = {}'.format(artist))
      df_user_artist = df_user.loc[df_user['artist_id'] == artist]
      
      if((not df_user_artist.empty)):
        if (df_user_artist.iloc[0]['log_plays'] > 0):
         # print('second if')
          num_hits += 1

    return num_hits

def hit_rate_total(test, recs):
    '''
    - recs: dict
        Keys are users and values are lists of top reccommendations for the user
    '''
    users = test['user_email'].unique()
    total_hits = 0
    for user in users:
        total_hits += hit_rate_user(user, recs[user], test)
    
    return total_hits/test['user_email'].nunique()
