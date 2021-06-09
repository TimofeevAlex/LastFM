# This file contains baseline and factorization-based models
# The neighborhood model is implemented in separate file neighborhood_helpers.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2

def baseline_predict(data, k=100):
    """
    Recommends the most popular artists over a given set 
    """
    total_log_plays = data.groupby('artist_id').sum()
    ranked_artist_scores = total_log_plays.sort_values(['rating'], ascending=False)['rating']
    return ranked_artist_scores[:k]

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

    # activation function
    output = tf.keras.activations.sigmoid(vector_product)
    
    # model
    model = tf.keras.models.Model(inputs = [user_email, artist_id], outputs = [output], name = 'shallow_model')

    return model

def create_neumf_model(num_factors, num_artists, num_users, reg=0.01):
    '''
    Creates the NeuMF model which consists of two branches. 
    The first embedding of user IDs branch are multiplied 
    by the first embedding of the artist IDs branch.
    The second embeddings of all branches are concatenated and
    passed to the shallow NN which output is concatenated to
    the result of the multiplication and passed to the final 
    layer which produces the probability of an interaction.
    Parameters:
        - num_factors: int, number of latent factors, basically, defines 
        the size of embeddings
        - num_user_features: int, number of user features to be passed
        - num_artists: int, number of artists which can be passed
        - num_users: int, number of users which can be passed
        - reg: float, a coefficient for l2-regularization
    '''
    # User IDs branch
    user_id = tf.keras.layers.Input(shape=[1], name='user_id')
    user_matrix_1 = tf.keras.layers.Embedding(num_users+1, num_factors, name='user_matrix_1', embeddings_regularizer=l2(reg))(user_id)
    user_matrix_2 = tf.keras.layers.Embedding(num_users+1, num_factors, name='user_matrix_2', embeddings_regularizer=l2(reg))(user_id)
    user_vector_proc_1 = tf.keras.layers.Flatten(name='user_id_vector_1')(user_matrix_1)
    user_vector_proc_2 = tf.keras.layers.Flatten(name='user_id_vector_2')(user_matrix_2)
    
    # Item IDs branch
    artist_id = tf.keras.layers.Input(shape=[1], name='artist_id')
    artist_matrix_1 = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix_1', embeddings_regularizer=l2(reg))(artist_id)
    artist_matrix_2 = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix_2', embeddings_regularizer=l2(reg))(artist_id)
    artist_vector_proc_1 = tf.keras.layers.Flatten(name='artist_vector_1')(artist_matrix_1)
    artist_vector_proc_2 = tf.keras.layers.Flatten(name='artist_vector_2')(artist_matrix_2)
    
    # Concantenation and multiplication
    vectors_mult = tf.keras.layers.Multiply()([user_vector_proc_1, artist_vector_proc_1]) #
    vectors_concat = tf.keras.layers.Concatenate()([user_vector_proc_2, artist_vector_proc_2]) #
    vectors_concat_dropout = tf.keras.layers.Dropout(0.5)(vectors_concat)
    
    # Backbone 
    dense_1 = tf.keras.layers.Dense(2 * num_factors, name='fc1', activation='relu', kernel_regularizer=l2(reg))(vectors_concat_dropout)
    dropout_1 = tf.keras.layers.Dropout(0.5, name='d1')(dense_1)
    dense_2 = tf.keras.layers.Dense(num_factors, name='fc2', activation='relu', kernel_regularizer=l2(reg))(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.5, name='d2')(dense_2)
    dense_3 = tf.keras.layers.Dense(num_factors // 2, name='fc3', activation='relu', kernel_regularizer=l2(reg))(dropout_2)
    dense_4 = tf.keras.layers.Dense(num_factors // 4, name='fc4', activation='relu', kernel_regularizer=l2(reg))(dense_3)

    # Merging a processed concatenated vector and a multiplication result 
    vectors_merged = tf.keras.layers.Concatenate()([vectors_mult, dense_4])
    output = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False,
                                   name='output', kernel_initializer="lecun_uniform", kernel_regularizer=l2(reg))(vectors_merged)
    
    # Model definition
    model = tf.keras.models.Model(inputs=[user_id, artist_id], outputs=[output], name='deep_factor_model')#
    return model

def create_neumf_model_si(num_factors, num_user_features, num_artists, num_users, reg=0.01):
    '''
    Creates the NeuMF model which consists of three branches. 
    The first embedding of user branches are concatenated and 
    multiplied by the first embedding of the artist IDs branch.
    The second embeddings of all branches are concatenated and
    passed to the shallow NN which output is concatenated to
    the result of the multiplication and passed to the final 
    layer which produces the probability of an interaction.
    Parameters:
        - num_factors: int, number of latent factors, basically, defines 
        the size of embeddings
        - num_user_features: int, number of user features to be passed
        - num_artists: int, number of artists which can be passed
        - num_users: int, number of users which can be passed
        - reg: float, a coefficient for l2-regularization
    '''
    # User IDs branch
    user_id = tf.keras.layers.Input(shape=[1], name='user_id')
    user_matrix_1 = tf.keras.layers.Embedding(num_users+1, num_factors // 2, name='user_matrix_1', embeddings_regularizer=l2(reg))(user_id)
    user_matrix_2 = tf.keras.layers.Embedding(num_users+1, num_factors // 2, name='user_matrix_2', embeddings_regularizer=l2(reg))(user_id)
    user_vector_proc_1 = tf.keras.layers.Flatten(name='user_id_vector_1')(user_matrix_1)
    user_vector_proc_2 = tf.keras.layers.Flatten(name='user_id_vector_2')(user_matrix_2)

    # User features  branch (concatenation with user's ID?)
    user_feats = tf.keras.layers.Input(shape=[num_user_features], name='user_features')
    features_vector_1 = tf.keras.layers.Dense(num_factors // 2, name='user_features_vector_1', activation='relu', kernel_regularizer=l2(reg))(user_feats)
    features_vector_2 = tf.keras.layers.Dense(num_factors // 2, name='user_features_vector_2', activation='relu', kernel_regularizer=l2(reg))(user_feats)
    
    # Item IDs branch
    artist_id = tf.keras.layers.Input(shape=[1], name='artist_id')
    artist_matrix_1 = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix_1', embeddings_regularizer=l2(reg))(artist_id)
    artist_matrix_2 = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix_2', embeddings_regularizer=l2(reg))(artist_id)
    artist_vector_proc_1 = tf.keras.layers.Flatten(name='artist_vector_1')(artist_matrix_1)
    artist_vector_proc_2 = tf.keras.layers.Flatten(name='artist_vector_2')(artist_matrix_2)
    
    # Concantenation and multiplication
    user_concat = tf.keras.layers.Concatenate()([user_vector_proc_1, features_vector_1])
    vectors_mult = tf.keras.layers.Multiply()([user_concat, artist_vector_proc_1]) #
    vectors_concat = tf.keras.layers.Concatenate()([user_vector_proc_2, features_vector_2, artist_vector_proc_2]) #
    vectors_concat_dropout = tf.keras.layers.Dropout(0.5)(vectors_concat)
    
    # Backbone 
    dense_1 = tf.keras.layers.Dense(2 * num_factors, name='fc1', activation='relu', kernel_regularizer=l2(reg))(vectors_concat_dropout)
    dropout_1 = tf.keras.layers.Dropout(0.5, name='d1')(dense_1)
    dense_2 = tf.keras.layers.Dense(num_factors, name='fc2', activation='relu', kernel_regularizer=l2(reg))(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.5, name='d2')(dense_2)
    dense_3 = tf.keras.layers.Dense(num_factors // 2, name='fc3', activation='relu', kernel_regularizer=l2(reg))(dropout_2)
    dense_4 = tf.keras.layers.Dense(num_factors // 4, name='fc4', activation='relu', kernel_regularizer=l2(reg))(dense_3)

    # Merging a processed concatenated vector and a multiplication result 
    vectors_merged = tf.keras.layers.Concatenate()([vectors_mult, dense_4])
    output = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False,
                                   name='output', kernel_initializer="lecun_uniform", kernel_regularizer=l2(reg))(vectors_merged)
    
    # Model definition
    model = tf.keras.models.Model(inputs=[user_id, user_feats, artist_id], outputs=[output], name='deep_factor_model')#
    return model

def create_neumf_only_si(num_factors, num_user_features, num_artists, num_users, reg=0.01):
    '''
    Creates the NeuMF model which consists of two branches. 
    The first embedding of user features branch are multiplied 
    by the first embedding of the artist IDs branch.
    The second embeddings of all branches are concatenated and
    passed to the shallow NN which output is concatenated to
    the result of the multiplication and passed to the final 
    layer which produces the probability of an interaction.
    Parameters:
        - num_factors: int, number of latent factors, basically, defines 
        the size of embeddings
        - num_user_features: int, number of user features to be passed
        - num_artists: int, number of artists which can be passed
        - num_users: int, number of users which can be passed
        - reg: float, a coefficient for l2-regularization
    '''
    # User features  branch (concatenation with user's ID?)
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
    vectors_mult = tf.keras.layers.Multiply()([features_vector_1, artist_vector_proc_1]) #
    vectors_concat = tf.keras.layers.Concatenate()([features_vector_2, artist_vector_proc_2]) #
    vectors_concat_dropout = tf.keras.layers.Dropout(0.5)(vectors_concat)
    
    # Backbone 
    dense_1 = tf.keras.layers.Dense(2 * num_factors, name='fc1', activation='relu', kernel_regularizer=l2(reg))(vectors_concat_dropout)
    dropout_1 = tf.keras.layers.Dropout(0.5, name='d1')(dense_1)
    dense_2 = tf.keras.layers.Dense(num_factors, name='fc2', activation='relu', kernel_regularizer=l2(reg))(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.5, name='d2')(dense_2)
    dense_3 = tf.keras.layers.Dense(num_factors // 2, name='fc3', activation='relu', kernel_regularizer=l2(reg))(dropout_2)
    dense_4 = tf.keras.layers.Dense(num_factors // 4, name='fc4', activation='relu', kernel_regularizer=l2(reg))(dense_3)

    # Merging a processed concatenated vector and a multiplication result 
    vectors_merged = tf.keras.layers.Concatenate()([vectors_mult, dense_4])
    output = tf.keras.layers.Dense(1, activation='sigmoid', use_bias=False,
                                   name='output', kernel_initializer="lecun_uniform", kernel_regularizer=l2(reg))(vectors_merged)
    
    # Model definition
    model = tf.keras.models.Model(inputs=[user_feats, artist_id], outputs=[output], name='deep_factor_model')#
    return model
