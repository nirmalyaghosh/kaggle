# -*- coding: utf-8 -*-
"""
Approach 1 : Makes use of basic features + the number of events per hour for
each device, then compares 2 methods to choose the one with lowest log loss.

First method involves comparing a few classifiers.
Second method uses stacked generalization, where the output of the Level 0
classifiers is fed to train the blended classifier.

Reference : http://www.chioka.in/stacking-blending-and-stacked-generalization/

@author: Nirmalya Ghosh
"""

import numpy as np
import pandas as pd
import time

from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

import utils
from td_config import cfg, logger


def prepare_datasets(data_dir):
    deviceinfo = utils.prepare_device_related_datasets(data_dir)

    # Count number of events per hour for each device (ephpd)
    ephpd = utils.prepare_events_per_hour_per_device_dataset(data_dir)

    # Events spread over 6 windows/splits through the day
    esd = utils.prepare_events_spread_dataset(data_dir)

    # Read the training & test datasets
    train = utils.read_gz(data_dir, "gender_age_train.csv.gz")
    test = utils.read_gz(data_dir, "gender_age_test.csv.gz")

    # Merge train and test with the events per hour per device dataset, ephpd
    train = pd.merge(train, ephpd, how="left")
    test = pd.merge(test, ephpd, how="left")
    # Merge train and test with the events spread dataset, esd
    train = pd.merge(train, esd, how="left")
    test = pd.merge(test, esd, how="left")

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


def run_stacked_generalization(clfs, X, y):
    # Shuffle
    idx = np.random.permutation(X.index)
    X = X.iloc[idx]
    y = y[idx]

    dev_cutoff = len(y) * 4/5
    X_dev = X[:dev_cutoff]  # (35829, 28)
    y_dev = y[:dev_cutoff]  # (35829L,)
    X_test = X[dev_cutoff:] # (8958, 28)
    y_test = y[dev_cutoff:] # (8958L,)

    skf = list(cv.StratifiedKFold(y_dev, cfg[s]["cv_nfold"]))

    # Pre-allocate the data
    blend_train = np.zeros((X_dev.shape[0], len(clfs))) # (35829L, 3L)
    blend_test = np.zeros((X_test.shape[0], len(clfs))) # (8958L, 3L)

    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(clfs):
        print("Training classifier {}".format(type(clf).__name__))
        blend_test_j = np.zeros((X_test.shape[0], len(skf)))

        # Number of testing data x Number of folds , we will take the mean of the predictions later
        for i, (train_index, cv_index) in enumerate(skf):
            print("Fold {}".format(i))
            X_train = X_dev.as_matrix()[train_index] # (17911L, 28L)
            Y_train = y_dev[train_index]             # (17911L,)
            X_valid = X_dev.as_matrix()[cv_index]    # (17918L, 28L)
            Y_valid = y_dev[cv_index]                # (17918L,)

            clf.fit(X_train, Y_train)
            blend_train[cv_index, j] = clf.predict_proba(X_valid)[:, 1]
            blend_test_j[:, i] = clf.predict_proba(X_test)[:,1]

        # Take the mean of the predictions of the cross validation set
        blend_test[:, j] = blend_test_j.mean(1)

    # Level 1 classifier, which does the blending
    # bclf = LogisticRegression(C=0.1, random_state=480, tol=0.005,
    #                           solver="newton-cg")
    bclf = LogisticRegression()
    bclf.fit(blend_train, y_dev)
    y_test_predict = bclf.predict_proba(blend_test)
    log_loss = metrics.log_loss(y_test, y_test_predict)
    return bclf, blend_test, log_loss


if __name__ == "__main__":
    s = "approach1"
    logger.info("Running script for Approach 1, %s", cfg[s]["description"])
    t0 = time.time()

    train, test, target = prepare_datasets("data")
    random_state = cfg["common"]["seed"]
    X_train, X_valid, y_train, y_valid = cv.train_test_split(
        train, target, test_size=0.4, random_state=random_state)
    X_submission = test.values[:, 1:]

    # Transforming the string output to numeric
    label_encoder = LabelEncoder()
    label_encoder.fit(target)
    num_classes = len(label_encoder.classes_)
    y = label_encoder.transform(target)

    # Level 0 classifiers
    clfs = [
        ExtraTreesClassifier(**utils.read_estimator_params(s, "et")),
        KNeighborsClassifier(),
        LogisticRegression(**utils.read_estimator_params(s, "lr")),
        RandomForestClassifier(**utils.read_estimator_params(s, "rf"))
    ]

    # First, run grid search (if enabled) to find the best estimator
    results_1 = []
    for clf in clfs:
        ts = time.time()
        clf_name = type(clf).__name__
        model = utils.find_best_estimator(clf, X_train, y_train, section=s)
        preds = model.predict_proba(X_valid)
        log_loss = metrics.log_loss(y_valid, preds)
        results_1.append((utils.get_key(clf_name), model, log_loss))
        logger.info("Trained {} in {:.2f} seconds, Log loss : {:.6f}"
            .format(clf_name, (time.time() - ts), log_loss))
    # Sort by log_loss
    results_1.sort(key=lambda tup: tup[2])
    logger.info(tabulate(zip([r[0] for r in results_1],
                             [r[2] for r in results_1]),
                         floatfmt=".4f", headers=("model", "log_loss")))
    clfs = [clf[1] for clf in results_1] # required for blending stage

    # Next, run stacked generalization (blending)
    logger.info("Start blending")
    results_2 = []
    for i in xrange(cfg[s]["n_blends"]):
        print("Iteration {}".format(i))
        bclf, b_t, log_loss = run_stacked_generalization(clfs, train, target)
        results_2.append((bclf, b_t, log_loss))
        logger.info("Iteration {}, Log loss : {:.4f}".format(i, log_loss))
    # Sort by log_loss
    results_2.sort(key=lambda tup: tup[2])

    # Prepare the DataFrame containing from the predicted_probabilities
    log_loss_1, log_loss_2 = results_1[0][2], results_2[0][2]
    model, predicted_probabilities = None, None
    if log_loss_1 < log_loss_2:
        logger.info("Method 1 has lower log loss {:.4f}".format(log_loss_1))
        model = results_1[0][1]
        predicted_probabilities = model.predict_proba(test)
    else:
        logger.info("Method 2 has lower log loss {:.4f}".format(log_loss_2))
        model = results_2[0][0]
        blend_test = results_2[0][1]
        predicted_probabilities = model.predict_proba(blend_test)

    df = pd.DataFrame(predicted_probabilities)
    df["device_id"] = test["device_id"]
    df = df[["device_id"] + np.arange(0, 12).tolist()]
    new_names = dict(zip(np.arange(0, 12).tolist(), model.classes_.tolist()))
    df.rename(columns=new_names, inplace=True)

    # Submission file
    prefix = utils.get_key(type(model).__name__)
    utils.make_submission_file(model, df, "{}_".format(prefix))
