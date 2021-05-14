import pandas as pd
from sklearn.metrics import ndcg_score

def precision_recall_at_k(test, pred_ratings, k=10):
    '''
    - pred_ratings: DataFrame
        It must have columns 'user_email', 'artists_id', 'pred_plays' 
    '''
    precisions = []
    recalls = []
    for (user, user_test) in test.groupby('user_email'):
        user_ratings = pred_ratings[pred_ratings['user_email'] == user]
        user_ratings = user_ratings.sort_values('pred_plays', ascending=False)
        top_k = set(user_ratings.artist_id[:k])
        test_artists = set(user_df.artist_id)
        precisions.append(len(top_k & test_artists) / float(k))
        recalls.append(len(top_k & test_artists) / len(test_artists))
    return precisions, recalls