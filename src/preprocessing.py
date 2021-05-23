
def to_ratings(user_df):
    user_df = user_df.sort_values('norm_plays', ascending=False) 
    cumsum = user_df['norm_plays'].cumsum() 
    user_df['norm_plays'] = 4 * (1 - cumsum)
    return user_df

def to_weights(user_df):
    weights = user_df['weight'].value_counts()
    user_df['weight'] = user_df['weight'].map(weights)
    return user_df