import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

def create_train_step(model, optimizer, loss_fn, epoch_loss_avg, epoch_rmse):
    ''' 
    Creates a function that makes one step of training
    Parameters:
        - model: keras model to be optimized
        - optimizer: keras optimizer which is used to train the model
        - loss_fn: loss function
        - epoch_loss_avg: tf.keras.metrics.Mean object which is used
            to compute the mean of the loss at the end of epoch
        - epoch_rmse: tf.keras.metrics.RootMeanSquaredError object which is used
            to compute the RMSE of the model at the end of epoch   
    '''
    @tf.function
    def train_step(x, y, weights):
        '''
        Trains the model one step
        Parameters:
            - x: input batch
            - y: target variables to samples from the input batch.
                Should have the same size as x
            - weights: weights of particular samples which are used
                in loss computation. Should have the same size as x and y  
        '''
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits, sample_weight=weights)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        epoch_loss_avg.update_state(loss_value)
        epoch_rmse.update_state(y, logits)
    return train_step

def create_test_step(model, loss_fn, test_loss_avg, test_rmse):
    ''' 
    Creates a function that makes one step of test
    Parameters:
        - model: keras model to be used
        - loss_fn: loss function
        - test_loss_avg: tf.keras.metrics.Mean object which is used
            to compute the mean of the loss
        - test_rmse: tf.keras.metrics.RootMeanSquaredError object which is used
            to compute the RMSE of the model  
    '''
    @tf.function
    def test_step(x, y):
        '''
        Trains the model one step
        Parameters:
            - x: input batch
            - y: target variables to samples from the input batch.
                Should have the same size as x
        '''
        logits = model(x, training=False)
        loss_value = loss_fn(y, logits)
        test_loss_avg.update_state(loss_value)
        test_rmse.update_state(y, logits)
        return logits
    return test_step

def create_inference(model):
    '''
    Creates a wrap-function for the model inference
    Parameters:
        - model: keras model to be used
    '''
    @tf.function
    def inference(x):
        ''
        logits = model(x, training=False)
        return logits
    return inference

def train_one_epoch(train_step, train_dataset, lastfm_360_demo, epoch_loss_avg, epoch_rmse, threshold, si=True, only_si=False):
    '''
    Trains the model for one epoch
    Parameters:
        - train_step: tf compiled function, train model using one batch
        - train_dataset: keras generator, train dataset loader which prepares batches
        - lastfm_360_demo: pandas DataFrame, contain users demographic data
        - epoch_loss_avg: tf.keras.metrics.Mean object which is used
            to compute the mean of the loss at the end of epoch
        - epoch_rmse: tf.keras.metrics.RootMeanSquaredError object which is used
            to compute the RMSE of the model at the end of epoch 
        - threshold: float, to make users with small ratings as negative samples
        - si: bool, whether use user features branch or not. Can't be True if only_si is True
        - only_si: bool, whether use only user features branch. Can't be True if si is True
    '''
    epoch_loss_avg.reset_states()
    epoch_rmse.reset_states()
    for batch in train_dataset:
        user_id = batch[:, 0]
        if si or only_si:
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
    '''
    Validates the model for one epoch
    Parameters:
        - val_step: tf compiled function, validates model using one batch
        - valid_dataset: keras generator, validation dataset loader which prepares batches
        - lastfm_360_demo: pandas DataFrame, contain users demographic data
        - val_loss_avg: tf.keras.metrics.Mean object which is used
            to compute the mean of the loss
        - val_rmse: tf.keras.metrics.RootMeanSquaredError object which is used
            to compute the RMSE of the model
        - threshold: float, to make users with small ratings as negative samples
        - si: bool, whether use user features branch or not. Can't be True if only_si is True
        - only_si: bool, whether use only user features branch. Can't be True if si is True
    '''
    val_loss_avg.reset_states()
    val_rmse.reset_states()
    for batch in valid_dataset:
        user_id = batch[:, 0]
        if si or only_si:
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
    '''
    Tests the model for one epoch
    Parameters:
        - test_step: tf compiled function, tests model using one batch
        - test_dataset: keras generator, test dataset loader which prepares batches
        - lastfm_360_demo: pandas DataFrame, contain users demographic data
        - test_loss_avg: tf.keras.metrics.Mean object which is used
            to compute the mean of the loss
        - test_rmse: tf.keras.metrics.RootMeanSquaredError object which is used
            to compute the RMSE of the model
        - threshold: float, to make users with small ratings as negative samples
        - si: bool, whether use user features branch or not. Can't be True if only_si is True
        - only_si: bool, whether use only user features branch. Can't be True if si is True
    '''
    test_loss_avg.reset_states()
    test_rmse.reset_states()
    probs = np.array([])
    for batch in test_dataset:
        user_id = batch[:, 0]
        if si or only_si:
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
    '''
    Plots the training and validation curves of the loss and the RMSE
    Parameters:
        - train_loss_results: list, a value of the train loss for each epoch before
        - train_rmse_results: list, a value of the train RMSE for each epoch before
        - val_loss_results: list, a value of the validation loss for each epoch before
        - val_loss_results: list, a value of the validation RMSE for each epoch before
        - epoch: int, a current epoch
        - log_frequency: int, the frequency of logging validation metrics and saving model weights
        - timenow: timestamp, a timestamp when training started
    '''
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