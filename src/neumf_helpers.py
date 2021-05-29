import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

def create_train_step(model, optimizer, loss_fn, epoch_loss_avg, epoch_rmse):
    @tf.function
    def train_step(x, y, weights):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits, sample_weight=weights)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        epoch_loss_avg.update_state(loss_value)
        epoch_rmse.update_state(y, logits)
    return train_step

def create_test_step(model, loss_fn, test_loss_avg, test_rmse):
    @tf.function
    def test_step(x, y):
        logits = model(x, training=False)
        loss_value = loss_fn(y, logits)
        test_loss_avg.update_state(loss_value)
        test_rmse.update_state(y, logits)
        return logits
    return test_step

def create_inference(model):
    @tf.function(experimental_relax_shapes=True)
    def inference(x):
        logits = model(x, training=False)
        return logits
    return inference

def train_one_epoch(train_step, train_dataset, lastfm_360_demo, epoch_loss_avg, epoch_rmse, threshold, si=True, only_si=False):
    epoch_loss_avg.reset_states()
    epoch_rmse.reset_states()
    for batch in train_dataset:
        user_id = batch[:, 0]
        if si:
            user_feats = lastfm_360_demo.loc[user_id]
        artist_id = batch[:, 1]
        y = batch[:, 2]
        weights = tf.expand_dims(1. + 0.25*y, -1)##weights from another method, another strategy of uncertainty, SGD/Adam
        if si:
            train_step([user_id, user_feats, artist_id], y >= threshold, weights) 
        elif only_si:
            train_step([user_feats, artist_id], y >= threshold, weights) 
        else:
            train_step([user_id, artist_id], y >= threshold, weights)
        
    return epoch_loss_avg.result().numpy(), epoch_rmse.result().numpy()

def validate_one_epoch(val_step, valid_dataset, lastfm_360_demo, val_loss_avg, val_rmse, threshold, si=True, only_si=False):
    val_loss_avg.reset_states()
    val_rmse.reset_states()
    for batch in valid_dataset:
        user_id = batch[:, 0]
        if si:
            user_feats = lastfm_360_demo.loc[user_id]
        artist_id = batch[:, 1]
        y = batch[:, 2]
        if si:
            val_step([user_id, user_feats, artist_id], y >= threshold) 
        elif only_si:
            val_step([user_feats, artist_id], y >= threshold) 
        else:
            val_step([user_id, artist_id], y >= threshold) 
    return val_loss_avg.result().numpy(), val_rmse.result().numpy()

def test_one_epoch(test_step, test_dataset, lastfm_360_demo, test_loss_avg, test_rmse, threshold, si=True, only_si=False):
    test_loss_avg.reset_states()
    test_rmse.reset_states()
    probs = np.array([])
    for batch in test_dataset:
        user_id = batch[:, 0]
        if si:
            user_feats = lastfm_360_demo.loc[user_id]
        artist_id = batch[:, 1]
        y = batch[:, 2]
        if si:
            probs_batch = test_step([user_id, user_feats, artist_id], y >= threshold)
        elif only_si:
            probs_batch = test_step([user_feats, artist_id], y >= threshold)
        else:
            probs_batch = test_step([user_id, artist_id], y >= threshold)
        probs = np.append(probs, tf.squeeze(probs_batch).numpy())
    return probs, test_loss_avg.result().numpy(), test_rmse.result().numpy()

def plot_metrics(train_loss_results, train_rmse_results, val_loss_results, val_rmse_results, epoch, log_frequency, timenow):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results, label='Train')
    axes[0].plot(np.arange(0, epoch+1, log_frequency), val_loss_results, label='Validation')

    axes[1].set_ylabel("RMSE", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_rmse_results, label='Train')
    axes[1].plot(np.arange(0, epoch+1, log_frequency), val_rmse_results, label='Validation')
    plt.legend()
    plt.savefig('plots/metrics_'+ timenow +'.png')
    plt.show()