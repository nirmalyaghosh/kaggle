# -*- coding: utf-8 -*-
"""
Approach 2 : Makes use of basic features + Bag-of-Apps,
then compares a few classifiers and selects the one with lowest log loss.
"""

import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from tabulate import tabulate

import utils
from td_config import cfg, logger


def prepare_datasets(data_dir):
    # Bag-of-Apps features based on
    # https://www.kaggle.com/xiaoml/talkingdata-mobile-user-demographics/
    # bag-of-app-id-python-2-27392/code

    # Read App Events
    app_events = utils.read_gz(data_dir, "app_events.csv.gz")
    app_events = app_events.groupby("event_id")["app_id"].apply(
        lambda x: " ".join(set("app_id:" + str(s) for s in x)))

    # Read Events
    events = pd.read_csv(os.path.join(data_dir, "events.csv.gz"),
                         dtype={"device_id": np.str})
    events["app_id"] = events["event_id"].map(app_events)
    events = events.dropna()
    del app_events
    events = events[["device_id", "app_id"]]

    events = events.groupby("device_id")["app_id"]\
        .apply(lambda x: " "
               .join(set(str(" ".join(str(s) for s in x)).split(" "))))
    events = events.reset_index(name="app_id")
    # expand to multiple rows
    events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))
                    for _, row in events.iterrows()]).reset_index()
    events.columns = ["app_id", "device_id"]

    # Read Phone Brand Device Model
    pbd = pd.read_csv(os.path.join(data_dir, "phone_brand_device_model.csv.gz"),
                      dtype={"device_id": np.str})
    pbd.drop_duplicates("device_id", keep="first", inplace=True)

    # Read Train and Test
    train = pd.read_csv(os.path.join(data_dir, "gender_age_train.csv.gz"),
                        dtype={"device_id": np.str})
    train.drop(["age", "gender"], axis=1, inplace=True)
    test = pd.read_csv(os.path.join(data_dir, "gender_age_test.csv.gz"),
                        dtype={"device_id": np.str})
    test["group"] = np.nan

    Y = train["group"]
    label_group = LabelEncoder()
    Y = label_group.fit_transform(Y)

    # Concat train and test,
    # before concatenating the features (phone_brand, device_model and app_id)
    df_all = pd.concat((train, test), axis=0, ignore_index=True)
    df_all = pd.merge(df_all, pbd, how="left", on="device_id")
    df_all["phone_brand"] = df_all["phone_brand"]\
        .apply(lambda x: "phone_brand:" + str(x))
    df_all["device_model"] = df_all["device_model"]\
        .apply(lambda x: "device_model:" + str(x))
    f1 = df_all[["device_id", "phone_brand"]]   # phone_brand
    f2 = df_all[["device_id", "device_model"]]  # device_model
    f3 = events[["device_id", "app_id"]]    # app_id
    del df_all
    # Rename the 2nd column
    f1.columns.values[1] = "feature"
    f2.columns.values[1] = "feature"
    f3.columns.values[1] = "feature"

    FLS = pd.concat((f1, f2, f3), axis=0, ignore_index=True)
    FLS = FLS.reset_index()

    # User-Item Feature
    device_ids = FLS["device_id"].unique()
    feature_cs = FLS["feature"].unique()

    data = np.ones(len(FLS))
    device_id_enc = LabelEncoder().fit(FLS["device_id"])
    row = device_id_enc.transform(FLS["device_id"])
    col = LabelEncoder().fit_transform(FLS["feature"])
    sparse_matrix = sparse.csr_matrix((data, (row, col)),
                                      shape=(len(device_ids), len(feature_cs)))
    sparse_matrix = sparse_matrix[:, sparse_matrix.getnnz(0) > 0]
    logger.info("sparse_matrix {}".format(sparse_matrix.shape))

    # Data Prep
    train_row = device_id_enc.transform(train["device_id"])
    train_sp = sparse_matrix[train_row, :]

    test_row = device_id_enc.transform(test["device_id"])
    test_sp = sparse_matrix[test_row, :]

    random_state = cfg["common"]["seed"]
    X_train, X_val, y_train, y_val = cv.train_test_split(
        train_sp, Y, train_size=.80, random_state=random_state)

    # Feature Selection
    selector = SelectPercentile(f_classif, percentile=23)
    selector.fit(X_train, y_train)
    X_train = selector.transform(X_train)
    X_val = selector.transform(X_val)
    train_sp = selector.transform(train_sp)
    test_sp = selector.transform(test_sp)
    logger.info("# Num of Features: {}".format(X_train.shape[1]))

    return X_train, X_val, y_train, y_val, test_sp


def train_model(X, y, X_, y_, clf):
    model = utils.find_best_estimator(clf, X, y, section="approach2")
    preds = model.predict_proba(X_)
    log_loss = metrics.log_loss(y_, preds)
    return model, log_loss


if __name__ == "__main__":
    s = "approach2"
    logger.info("Running script for Approach 2, %s", cfg[s]["description"])
    t0 = time.time()

    X_train, X_val, y_train, y_val, test_sp = prepare_datasets("data")
    logger.info("Data prep took {:.2f} seconds".format((time.time() - t0)))

    # Compare a few classifiers
    clfs = [
        (ExtraTreesClassifier(**utils.read_estimator_params(s, "et")), "et"),
        (LogisticRegression(**utils.read_estimator_params(s, "lr")), "lr"),
        (RandomForestClassifier(**utils.read_estimator_params(s, "rf")), "rf")
    ]
    results = []
    for clf in clfs:
        ts = time.time()
        model, log_loss = train_model(X_train, y_train, X_val, y_val, clf[0])
        results.append((clf[1], model, log_loss))
        logger.info("Trained {} in {:.2f} seconds, Log loss : {:.6f}"
            .format(type(clf[0]).__name__, (time.time() - ts), log_loss))
    # Sort by log_loss
    results.sort(key=lambda tup: tup[2])

    # Prepare the DataFrame containing from the predicted_probabilities
    model = results[0][1]
    predicted_probabilities = model.predict_proba(test_sp)
    df = pd.DataFrame(predicted_probabilities)
    subm = pd.read_csv(os.path.join("data", "sample_submission.csv.gz"),
                       dtype={"device_id": np.str})
    classes = subm.columns.values.tolist()[1:]

    df["device_id"] = subm["device_id"]
    df = df[["device_id"] + np.arange(0, 12).tolist()]
    new_names = dict(zip(np.arange(0, 12).tolist(), classes))
    df.rename(columns=new_names, inplace=True)

    # Submission file
    logger.info(tabulate(zip([r[0] for r in results], [r[2] for r in results]),
                         floatfmt=".4f", headers=("model", "log_loss")))
    utils.make_submission_file(model, df, "%s_" % results[0][0])
