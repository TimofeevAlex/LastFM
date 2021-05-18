import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from tqdm.auto import tqdm

def precision_recall_at_k(test, pred_ratings, k=10):
    '''
    - pred_ratings: dict
        Keys are users and values are arrays of shape (2, number of items)
    '''
    precisions = []
    recalls = []
    for (user, user_test) in tqdm(test.groupby('user_email')):
        user_ratings = pred_ratings[user]
        ranked_artists = user_ratings[0, np.argsort(-user_ratings[1])].astype(np.int32)[:k]
        top_k = set(ranked_artists)
        test_artists = set(user_test.artist_id)
        precisions.append(len(top_k & test_artists) / float(k))
        recalls.append(len(top_k & test_artists) / len(test_artists))
    return precisions, recalls


def hit_rate_total(test, recs):
    '''
    - recs: dict
        Keys are users and values are lists of top reccommendations for the user
    '''
    total_hits = 0
    for (user, user_test) in tqdm(test.groupby('user_email')):
        # Get all artists for a given user
        user_artists = user_test[user_test['log_plays'] > 0]['artist_id'].values
        # Compute the number of artists in both sets
        hit_rate_user = len(set(user_artists).intersection(set(recs[user])))
        
        total_hits += hit_rate_user
    
    return total_hits/test['user_email'].nunique()
    

def arhr_total(test, recs):
    '''
    - recs: dict
        Keys are users and values are lists of top reccommendations for the user
    '''
    total_hit_rank = 0
    for (user, user_test) in tqdm(test.groupby('user_email')):
        user_artists = user_test[user_test['log_plays'] > 0]['artist_id'].values
        # Compute the number of hits
        hits = set(user_artists).intersection(set(recs[user]))
        # Build map of ranks
        artist_to_rank = {a:(i+1) for (i, a) in enumerate(recs[user])}
        # Compute user hit rank
        total_hit_rank += sum([1/artist_to_rank[h] for h in hits])
    
    return total_hit_rank/test['user_email'].nunique()

def ndcg_at_k(test, pred_ratings, k=10):
    """
    - pred_ratings: dict
        Keys are users and values are arrays of shape (2, number of items)
    """
    ndcg_arr = []
    for (user, user_test) in tqdm(test.groupby('user_email')):
        user_ratings = pred_ratings[user]
        # Get top k artists and score
        ranked_artists = user_ratings[0, np.argsort(-user_ratings[1])][:k]
        ranked_scores = user_ratings[1, np.argsort(-user_ratings[1])][:k]
        
        # Get top k artists true score
        artists_df = pd.DataFrame(ranked_artists, columns=['artist_id'], dtype=np.int64)
        test_scores = artists_df.merge(user_test, on='artist_id', how='left', ).fillna(0)['log_plays'].values
        
        # Compute ndcg score
        user_score = ndcg_score([test_scores], [ranked_scores])
        
        ndcg_arr.append(user_score)
    return ndcg_arr
    