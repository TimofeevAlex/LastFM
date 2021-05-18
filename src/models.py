import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2

def baseline_predict(data, k=100):
    """
    Recommends the most popular artists over a given set 
    """
    total_log_plays = data.groupby('artist_id').sum()
    ranked_artist_scores = total_log_plays.sort_values(['log_plays'], ascending=False)['log_plays']
    return ranked_artist_scores[:k]

def create_ncf_model(num_factors, num_user_features, num_artists, num_users, reg=0.01):
    # User IDs branch
    user_id = tf.keras.layers.Input(shape=[1], name='user_id')
    user_matrix_1 = tf.keras.layers.Embedding(num_users+1, num_factors, name='user_matrix_1', embeddings_regularizer=l2(reg))(user_id)
    user_matrix_2 = tf.keras.layers.Embedding(num_users+1, num_factors, name='user_matrix_2', embeddings_regularizer=l2(reg))(user_id)
    user_vector_proc_1 = tf.keras.layers.Flatten(name='user_id_vector_1')(user_matrix_1)
    user_vector_proc_2 = tf.keras.layers.Flatten(name='user_id_vector_2')(user_matrix_2)

    # User features  branch
    user_feats = tf.keras.layers.Input(shape=[num_user_features], name='user_features')
    features_vector_1 = tf.keras.layers.Dense(num_factors, name='user_features_vector_1', activation='relu', kernel_regularizer=l2(reg))(user_feats)
    features_vector_2 = tf.keras.layers.Dense(num_factors, name='user_features_vector_2', activation='relu', kernel_regularizer=l2(reg))(user_feats)
    
    # Item IDs branch
    artist_id = tf.keras.layers.Input(shape=[1], name='artist_id')
    artist_matrix_1 = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix_1', embeddings_regularizer=l2(reg))(artist_id)
    artist_matrix_2 = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix_2', embeddings_regularizer=l2(reg))(artist_id)
    artist_vector_proc_1 = tf.keras.layers.Flatten(name='artist_vector_1')(artist_matrix_1)
    artist_vector_proc_2 = tf.keras.layers.Flatten(name='artist_vector_2')(artist_matrix_2)
    
    # Concantenation and multiplication
    vectors_mult = tf.keras.layers.Multiply()([user_vector_proc_1, features_vector_1, artist_vector_proc_1])
    vectors_concat = tf.keras.layers.Concatenate()([user_vector_proc_2, features_vector_2, artist_vector_proc_2])
    vectors_concat_dropout = tf.keras.layers.Dropout(0.2)(vectors_concat)
    
    # Backbone 
    dense_1 = tf.keras.layers.Dense(2 * num_factors, name='fc1', activation='relu', kernel_regularizer=l2(reg))(vectors_concat)
    dense_1_bn = tf.keras.layers.BatchNormalization()(dense_1)
    dropout_1 = tf.keras.layers.Dropout(0.2, name='d1')(dense_1_bn)
    dense_2 = tf.keras.layers.Dense(num_factors, name='fc2', activation='relu', kernel_regularizer=l2(reg))(dropout_1)
    dense_2_bn = tf.keras.layers.BatchNormalization()(dense_2)
    dropout_2 = tf.keras.layers.Dropout(0.2, name='d2')(dense_2_bn)
    dense_3 = tf.keras.layers.Dense(num_factors // 2, name='fc3', activation='relu', kernel_regularizer=l2(reg))(dropout_2)
    dense_4 = tf.keras.layers.Dense(num_factors // 4, name='fc4', activation='relu', kernel_regularizer=l2(reg))(dense_3)

    # Merging a processed concatenated vector and a multiplication result 
    vectors_merged = tf.keras.layers.Concatenate()([vectors_mult, dense_4])
    output = tf.keras.layers.Dense(1, name='output', kernel_initializer="lecun_uniform", kernel_regularizer=l2(reg))(vectors_merged)
    
    # Model definition
    model = tf.keras.models.Model(inputs=[user_id, user_feats, artist_id], outputs=[output], name='deep_factor_model')
    return model

def create_shallow_model(num_factors, num_users, num_artists):
    # users
    user_email = tf.keras.layers.Input(shape = [1], name = 'user_email')
    user_matrix = tf.keras.layers.Embedding(num_users+1, num_factors, name = 'user_matrix')(user_email)
    user_vector = tf.keras.layers.Flatten(name = 'user_vector')(user_matrix)
    
    # artists
    artist_id = tf.keras.layers.Input(shape = [1], name = 'artist_id')
    artist_matrix = tf.keras.layers.Embedding(num_artists+1, num_factors, name = 'artist_matrix')(artist_id)
    artist_vector = tf.keras.layers.Flatten(name = 'artist_vector')(artist_matrix)
    
    # dot product
    vector_product = tf.keras.layers.dot([user_vector, artist_vector], axes = 1, normalize = False)
    
    # model
    model = tf.keras.models.Model(inputs = [user_email, artist_id], outputs = [vector_product], name = 'shallow_model')
    
    return model
