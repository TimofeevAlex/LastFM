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