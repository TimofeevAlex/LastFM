# This file contains all methods used to build the user-user neighborhood model. 
import pandas as pd
import pickle
import numpy as np
from tqdm.auto import tqdm
import time

# ----- Preprocessing methods -----

def get_age_chunk_index(user_email:int, demo_df, chunk_size=4):
    """
    Used to divide the users in groups using age parameter. 
    Return an index corresponding to the user's age group.
    """
    age_columns = demo_df.columns[3:93]
    user_age_vals = demo_df.loc[user_email, age_columns]
    user_age = [i for i, v in user_age_vals.items() if v]
    
    if len(user_age) != 1:
        return -1
    
    min_age = int(float(user_age_vals.index[0]))
    age = int(float(user_age[0]))
    
    return (age - min_age) // chunk_size 


def compute_groups(train_df, demo_df):
    """
    Given the train data, split the users into smaller groups
    @return and list of user groups (list containing the user ids).
    """
    users = train_df['user_email'].unique()
    # Compute chunk index given user email
    user_to_chunk = {u:get_age_chunk_index(u, demo_df) for u in  tqdm(users)}
    
    # Get the chunk indices
    chunks = set(user_to_chunk.values())

    # Compute dict of chunks to users
    chunk_users = {}
    for c in chunks:
        chunk_users[c] = [u for u, v in user_to_chunk.items() if v == c]
        
    # Compute list of users group of less than 20k users (maximum for our correlation matrix).
    groups = []
    current_group = []
    
    chunks.remove(-1) # Remove the nan age group (added as an individual group)
    
    for i in chunks: # Build group of less than 20k users
        if len(current_group) > 10000 or (len(chunk_users[i]) > 10000): # If length above 10k, reset current group
            groups.append(current_group)
            current_group = []
        
        current_group.extend(chunk_users[i])
    groups.append(chunk_users[-1]) # Add the group of users without age 
    return groups

# ----- Model computation methods -----


def get_top_k_neighbors(corr_df, user, k=100):
    """
    Given the correlation matrix for a given user group, return the user's neighbors.
    @return 100 user's with highest similarity score: [(user_id, score), ...]
    """
    user_corr = corr_df.loc[user]
    top_k_indices = user_corr.values.argsort()[-(k+1):][::-1][1:]
    top_k_users = user_corr.index[top_k_indices]
    return list(user_corr[top_k_users].items())

def filter_artists(train_df, artist_threshold:int=100, verbose:bool=False):
    """
    Filter the total number of artists in the interaction matrix to speed computations.
    @return list of artist id that are above given threshold.
    """
    selected_artists = train_df['artist_id'].value_counts()
    selected_artists= selected_artists[selected_artists > artist_threshold].index.values
    if verbose: print(f"Number of selected artists: {len(selected_artists)}")
    return selected_artists


def compute_neighborhood_model(train_df, user_groups, artist_threshold:int=100, 
                         k_neighbors:int=100, binary_scores:bool=True, verbose=False):
    """
    Given train data and user groups, compute the neighbors model:
    1. Filter in artists given minimal threshold.
    2. Compute binary interactions.
    2. For each user group:
        * Pivot dataset into interactions matrix
        * Compute correlation matrix (using Pearson's correlation)
        * Select user's neighbors with score from correlation
    4. Merge groups to build user's neighbors dataset (model)
    """
    
    # In order to reduce the computation time, we make predictions from only a subset of artists. 
    # The subset contains artists listened by more than 100 users.
    selected_artists = filter_artists(train_df, artist_threshold, verbose)
    
    # Make sure there are no duplicates
    my_df = train_df.groupby(['user_email', 'artist_id'], as_index=False).sum()
    # Select row with selected artists
    my_df = my_df[my_df['artist_id'].isin(selected_artists)]
    
    # Select binary score or ratings
    score_col = 'binary' if binary_scores else 'rating'
    
    if score_col == 'binary':
        my_df['binary'] = my_df['rating'].apply(lambda x: 1 if x > 0 else 0)
        
    # Iterate over the user groups to reduce the correlation matrix dimention
    user_to_neighbors = {}
    groups_size = [len(user_groups[i]) for i in range(len(user_groups))]
    if verbose: print(f"User groups size: {groups_size}")
    for user_group in tqdm(user_groups):
        start = time.time()
        my_df_small = my_df[my_df['user_email'].isin(user_group)] # Get data for given user group
        # Build my with scores
        my_df_small = my_df_small.pivot(index='user_email', columns='artist_id', values=score_col).fillna(0)
        # Compute correlation matrix
        corr = np.corrcoef(my_df_small.values)
        corr_df = pd.DataFrame(data = corr, index=my_df_small.index, columns=my_df_small.index).fillna(0)
        end = time.time()
        if verbose: print(f"Correlation matrix computation: {end - start} seconds.")
        # Get top k neighbors
        for user in tqdm(corr_df.index):
            user_to_neighbors[user] = get_top_k_neighbors(corr_df, user)
    
    neighbors_df = pd.DataFrame({'user_email':user_to_neighbors.keys(), 
                                 'neighbors':user_to_neighbors.values()}).set_index('user_email')
    return neighbors_df
    
# ----- Compute predictions -----

def get_neighbors_rating(neighbors, n_df, artist_id):
    """
    Given a list of neighbors and an artist, get neighbor's rating in train data.
    @return a DataFrame with neighbors id and rating.
    """
    res_df = pd.DataFrame(neighbors, columns=['user_email'])
    a_df = n_df[n_df['artist_id'] == artist_id]
    res_df = res_df.merge(a_df, how='left', on='user_email')[['user_email', 'rating']].fillna(0).set_index('user_email')
    return res_df

def compute_var(n_rating, n_corr_df, avg_n_rating):
    """
    Compute the variance term in the prediction (this term depends on the neighbors.
    """
    var = 0
    corr_sum = 0
    for n in n_rating.index:
        corr_sum += n_corr_df.loc[n, 'corr']
        
        var += n_corr_df.loc[n, 'corr'] * (n_rating.loc[n, 'rating'] - avg_n_rating.loc[n, 'rating'])   
    var = var / corr_sum
    return var

def compute_user_predictions(train_df, user_email, selected_artists,  model, verbose=False):
    """
    Compute the predictions for a given user and a list of artists to predict. 
    1. Get the user's neighbors
    2. Get train rating for neighbors
    3. Compute the prediction for each artist
    @return a list of predictions
    """
    
    if user_email not in model.index: # If the user had no artists in the selected artists, return 0. 
        return [0]*selected_artists
    
    # Get neighbors data
    neighbors = model.loc[user_email, 'neighbors']
    neighbors_email = np.array([n[0] for n in neighbors])
    neighbors_corr = np.array([n[1] for n in neighbors])
    n_corr_df = pd.DataFrame(data={'user_email':neighbors_email, 'corr':neighbors_corr}).set_index('user_email')
    neighbors_behav = train_df[train_df['user_email'].isin(neighbors_email)]
    
    # Get average rating for neighbors and user 
    avg_n_rating = pd.DataFrame(neighbors_behav.groupby('user_email')['rating'].mean())
    avg_user_rating = train_df.loc[train_df['user_email'] == user_email, 'rating'].mean()
    
    # For each artist, compute the prediction
    results = []
    for a in tqdm(selected_artists, disable=(not verbose)): 
        neighbors_rating = get_neighbors_rating(neighbors_email, neighbors_behav, a)
        remaining_n_rating = neighbors_rating[neighbors_rating['rating'] > 0]
        
        if not len(remaining_n_rating) == 0:
            result = avg_user_rating + compute_var(remaining_n_rating, n_corr_df, avg_n_rating)
            if np.isnan(result): 
                result = 0
        else:
            result = 0 # No neighbors have listened to the artist, return 0. 
            
        results.append(result)
        
    return results



# Due to computation time of predictions, we save snapshots of the predictions while running. 
def save_dict(dict_: dict, directory: str, name: str):
    """
    Save dictionary to pickle file. 
    """
    with open(directory + name + '.pickle', 'wb') as handle:
        pickle.dump(dict_, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def load_dict(directory: str, name: str):
    """
    Load dictionary from pickle file.
    """
    with open(directory + name + '.pickle', 'rb') as handle:
        res = pickle.load(handle)
        return res
    return None

