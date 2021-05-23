
def to_ratings(user_df):
    user_df = user_df.sort_values('norm_plays', ascending=False) 
    cumsum = user_df['norm_plays'].cumsum() 
    user_df['norm_plays'] = 4 * (1 - cumsum)
    return user_df

def to_weights(user_df):
    weights = user_df['weight'].value_counts()
    user_df['weight'] = user_df['weight'].map(weights)
    return user_df

def build_get_negative_ratings(all_artists):
    def get_negative_ratings(user_df):
        user_interactions = user_df['artist_id']
        num_interactions = user_df['artist_id'].shape[0]
        new_artists = list(all_artists - set(user_df['artist_id']))
        new_artists = np.random.choice(new_artists, num_interactions, replace=False)
        return pd.DataFrame({'artist_id': new_artists, 'rating': [0] * new_artists})
    return get_negative_ratings