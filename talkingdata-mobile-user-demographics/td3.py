# -*- coding: utf-8 -*-
"""
Approach 3 : Makes use of sparse features and a sequential model having one
fully-connected hidden layer.

@author: Nirmalya Ghosh
"""

import os
import time

import numpy as np
import pandas as pd
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

import utils as u
from td_config import cfg, logger


def batch_generator(X, y, batch_size, shuffle):
    # As of now Keras does not support sparse matrix.
    # So the Xtrain and Xtest need to be converted from csr_matrix to dense.
    # Using csr.toarray() not a good option (expensive).
    # A reply(https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices/129481#post129481)
    # on the forum suggest using Keras' train_on_batch feature, and convert
    # only the current input batch to dense.
    number_of_batches = X.shape[0]/batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_predict_generator(X, batch_size, shuffle):
    # Similar to the batch_generator used for train_on_batch feature,
    # we need a generate for predictions as well, hence batch_predict_generator.
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


if __name__ == "__main__":
    np.random.seed(730521)
    # Read the train and test data
    data_dir = "data"
    logger.info("Running script for Approach 3")
    train = pd.read_csv(os.path.join(data_dir, "gender_age_train.csv.gz"),
                        index_col = "device_id")
    test = pd.read_csv(os.path.join(data_dir, "gender_age_test.csv.gz"),
                       index_col = "device_id")
    # Encode the age groups
    y_enc = LabelEncoder().fit(train.group)
    y = y_enc.transform(train.group)
    # Create sparse features
    Xtrain, Xtest = u.prepare_sparse_features_dataset(train, test, data_dir)
    dummy_y = np_utils.to_categorical(y)

    # Create the Keras model and compile it
    model = Sequential()
    inp_dim = Xtrain.shape[1]
    model.add(Dense(10, input_dim=inp_dim, init="normal", activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, input_dim=inp_dim, init="normal", activation="tanh"))
    model.add(Dense(12, init="normal", activation="sigmoid"))

    sgd = SGD(lr=.01, decay=1e-9, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd,
                  metrics=["accuracy"])

    # Training
    logger.info("Training model, {}".format(model.to_json()))
    X_train, X_val, y_train, y_val = \
        train_test_split(Xtrain, dummy_y, test_size=0.02, random_state=42)

    num_epoch = 15
    batch_gen = batch_generator(X_train, y_train, 32, True)
    fit = model.fit_generator(generator=batch_gen,
                              nb_epoch=num_epoch,
                              samples_per_epoch=69984,
                              validation_data=(X_val.todense(), y_val),
                              verbose=2)

    # Evaluate the model
    scores_val = model.predict_generator(
        generator=batch_predict_generator(X_val, 32, False),
        val_samples=X_val.shape[0])
    scores = model.predict_generator(
        generator=batch_predict_generator(Xtest, 32, False),
        val_samples=Xtest.shape[0])
    logger.info("logloss val {}".format(log_loss(y_val, scores_val)))

    # Get the predicted_probabilities and prepare file for submission
    pred = pd.DataFrame(scores, index = test.index, columns=y_enc.classes_)
    pred = pd.DataFrame(pred, index = test.index, columns=y_enc.classes_)
    ts = time.strftime("%a_%d%b%Y_%H%M%S")
    name_prefix = "sparse_keras_v2_{}epoch_".format(num_epoch)
    file_path = os.path.join("submissions", "%s%s.csv" % (name_prefix, ts))
    pred.to_csv(file_path, index=True)
    u.gzip_file(file_path)
