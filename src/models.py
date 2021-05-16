import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.regularizers import l2

#probably a branch that takes 
def create_ncf_model(num_factors, num_user_features, num_artists, num_users):
    # User branch - IDs
    user_id = tf.keras.layers.Input(shape=[1], name='user_id')
    user_matrix = tf.keras.layers.Embedding(num_users+1, num_factors, name='user_matrix')(user_id)
    user_vector = tf.keras.layers.Flatten(name='user_id_vector')(user_matrix)
    user_vector_proc = tf.keras.layers.Dense(num_factors, name='user_id_vector_proc', activation='selu', kernel_regularizer=l2())(user_vector)
    # User branch - features
    user_feats = tf.keras.layers.Input(shape=[num_user_features], name='user_features')
    dense_1 = tf.keras.layers.Dense(3 * num_user_features // 4, name='fc1', activation='selu', kernel_regularizer=l2())(user_feats)
    dropout_1 = tf.keras.layers.Dropout(0.2, name='d1')(dense_1)
    dense_2 = tf.keras.layers.Dense(num_user_features // 2, name='fc2', activation='selu', kernel_regularizer=l2())(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.2, name='d2')(dense_2)
    user_vector = tf.keras.layers.Dense(num_factors, name='user_vector', activation='selu', kernel_regularizer=l2())(dropout_2)
    
    # Item branch
    artist_id = tf.keras.layers.Input(shape=[1], name='artist_id')
    artist_matrix = tf.keras.layers.Embedding(num_artists+1, num_factors, name='artist_matrix')(artist_id)
    artist_vector = tf.keras.layers.Flatten(name='artist_vector')(artist_matrix)
    artist_vector_proc = tf.keras.layers.Dense(num_factors, name='artist_vector_proc', activation='selu', kernel_regularizer=l2())(artist_vector)
    
    # Concantenation
    vectors_concat = tf.keras.layers.Concatenate()([user_vector_proc, user_vector, artist_vector_proc]) #add
    vectors_concat_dropout = tf.keras.layers.Dropout(0.2)(vectors_concat)
    
    # Backbone 
    dense_3 = tf.keras.layers.Dense(num_factors, name='fc3', activation='selu', kernel_regularizer=l2())(vectors_concat_dropout)
    dropout_3 = tf.keras.layers.Dropout(0.2, name='d3')(dense_3)
    dense_4 = tf.keras.layers.Dense(num_factors // 2, name='fc4', activation='selu', kernel_regularizer=l2())(dropout_3)
    dropout_4 = tf.keras.layers.Dropout(0.2, name='d4')(dense_4)
    dense_5 = tf.keras.layers.Dense(num_factors // 4, name='fc5', activation='selu', kernel_regularizer=l2())(dropout_4)
    dense_6_output = tf.keras.layers.Dense(1, activation='selu', name='output', kernel_regularizer=l2())(dense_5)
    
    # Model definition
    model = tf.keras.models.Model(inputs=[user_id, user_feats, artist_id], outputs=[dense_6_output], name='deep_factor_model')
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
