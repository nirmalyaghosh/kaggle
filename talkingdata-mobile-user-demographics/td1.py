# -*- coding: utf-8 -*-
"""
Approach 1 : 

"""

import numpy as np
import pandas as pd
import time

from sklearn import cross_validation as cv
from sklearn import ensemble
from sklearn import metrics

import utils
from td_config import cfg, logger


def prepare_datasets(data_dir):
    deviceinfo = utils.prepare_device_related_datasets(data_dir)

    # Count number of events per hour for each device (ephpd)
    ephpd = utils.prepare_events_per_hour_per_device_dataset(data_dir)

    # Read the training & test datasets
    train = utils.read_gz(data_dir, "gender_age_train.csv.gz")
    test = utils.read_gz(data_dir, "gender_age_test.csv.gz")

    # Merge train and test with the events per hour per device dataset, ephpd
    train = pd.merge(train, ephpd, how="left")
    test = pd.merge(test, ephpd, how="left")

    # Merge train and test with a subset of columns of the device info dataset
    df2 = deviceinfo[["device_id", "phone_brand_id", "is_foreign_brand",
                      "device_model_id"]].copy()
    df2 = df2.drop_duplicates(subset=["device_id"], keep="last")
    train = pd.merge(train, df2, how="left", on="device_id")
    test = pd.merge(test, df2, how="left", on="device_id")

    # Prepare the train and test datasets
    hour_of_day_cols = ["h" + str(x) for x in np.arange(0, 24).tolist()]
    cols_to_drop = list(hour_of_day_cols)
    test.drop(cols_to_drop, axis=1, inplace=True)
    test.fillna(-1, inplace=True)
    cols_to_drop.extend(["gender", "age"])
    train.drop(cols_to_drop, axis=1, inplace=True)
    target = train.group.values
    train = train.drop(["group"], axis=1)
    train.fillna(-1, inplace=True)
    return train, test, target


def train_model(X, y, clf):
    clf = ensemble.RandomForestClassifier()
    model = utils.find_best_estimator(clf, X, y, section="approach1",
                                      grid_search_params_key="param_dist")
    logger.info(model)
    preds = model.predict_proba(X_eval)
    log_loss = metrics.log_loss(y_eval, preds)
    logger.info("Log loss : %.6f" % log_loss)
    return model


if __name__ == "__main__":
    s = "approach1"
    logger.info("Running script for Approach 1, %s", cfg[s]["description"])
    t0 = time.time()

    # Prepare the data
    train, test, target = prepare_datasets("data")
    X, X_eval, y, y_eval = cv.train_test_split(train, target, test_size=0.4)

    model = train_model(X, y, "rf")
    predicted_probabilities = model.predict_proba(test)
    logger.info("Time taken : %.2f seconds" % (time.time() - t0))

    # Prepare the DataFrame containing from the predicted_probabilities
    df = pd.DataFrame(predicted_probabilities)
    df["device_id"] = test["device_id"]
    df = df[["device_id"] + np.arange(0, 12).tolist()]
    new_names = dict(zip(np.arange(0, 12).tolist(), model.classes_.tolist()))
    df.rename(columns=new_names, inplace=True)

    utils.make_submission_file(model, df, "rf_")
