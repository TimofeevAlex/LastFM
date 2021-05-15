import numpy as np
# import pandas as pd
import tensorflow as tf
from tqdm import tqdm

def create_train_step(model, loss_fn, epoch_loss_avg, epoch_rmse):
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        epoch_loss_avg.update_state(loss_value)
        epoch_rmse.update_state(y, logits)
        return loss_value
    return train_step

def create_test_step(model, loss_fn, test_loss_avg, test_rmse):
    @tf.function
    def test_step(x, y):
        val_logits = model(x, training=False)
        loss_value = loss_fn(y, val_logits)
        test_loss_avg.update_state(loss_value)
        test_rmse.update_state(y, val_logits)
        return loss_value
    return test_step

def create_inference(model):
    @tf.function(experimental_relax_shapes=True)
    def inference(x):
        logits = model(x, training=False)
        return logits
    return inference

def get_ratings(pred_func, train, users_demo, artists):
    ratings = {}
    set_artists = set(artists)
    for user, user_df in tqdm(train.groupby('user_email')):
        new_artists = np.array(list(set_artists - set(user_df['artist_id'])))
        user_feats = users_demo.loc[[user]*new_artists.shape[0]]
        ratings[user] = np.stack([new_artists, pred_func([user_feats, new_artists]).numpy().squeeze()])
    return ratings