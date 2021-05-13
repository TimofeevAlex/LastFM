import pandas as pd
from sklearn.metrics import ndcg_score

def precision_recall_at_k(test, pred_ratings, k=10):
    precisions = []
    recalls = []
    for user, user_df in test.groupby('user_email'):
        user_ratings = pred_ratings[user_df.index]
        user_df['pred_ratings'] = user_ratings
        user_df = user_df.sort_values('pred_ratings', ascending=False)
        top_k = set(user_df.artist_id[:k])
        test_artists = set(user_df.artist_id)
        precisions.append(len(top_k & test_artists) / float(k))
        recalls.append(len(top_k & test_artists) / len(test_artists))
    return precisions, recalls