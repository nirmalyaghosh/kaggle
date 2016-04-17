# -*- coding: utf-8 -*-
"""
Approach 5 : Use ExtraTreesClassifier.
Drop columns identified by other scripts.
Undo the feature scaling (based on a variant of  http://bit.ly/1RV4w0y)

@author: Nirmalya Ghosh
"""

import time

import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

import utils
from bnp_config import cfg, logger


if __name__ == '__main__':
    s = "approach5"
    logger.info("Running script for BNP Approach 5, %s", cfg[s]["description"])

    train = pd.read_csv("data/train.csv.gz")
    test = pd.read_csv("data/test.csv.gz")
    id_test = test.ID.values
    target = train.target.values

    drops = cfg[s]["columns_to_drop"]
    train = train.drop(drops, axis=1)
    train = train.drop(["target"], axis=1)
    test = test.drop(drops, axis=1)

    # Get the columns with numeric data
    # Credit : http://stackoverflow.com/a/28155580
    numeric_cols = list(
        train.select_dtypes(include=[np.number]).columns.values)

    # Undo the feature scaling
    df = pd.concat([train, test])
    for col in numeric_cols:
        train.loc[train[col].round(5) == 0, col] = 0
        test.loc[test[col].round(5) == 0, col] = 0
        denominator = utils.find_denominator(df, col)
        train[col] *= 1 / denominator
        test[col] *= 1 / denominator

    for (a, a_vals), (b, b_vals) in zip(train.iteritems(), test.iteritems()):
        if a_vals.dtype == "O":
            train[a], tmp_indexer = pd.factorize(train[a])
            test[b] = tmp_indexer.get_indexer(test[b])
        else:
            # For numeric columns, replace missing values with -999
            tmp_len = len(train[a_vals.isnull()])
            if tmp_len > 0:
                train.loc[a_vals.isnull(), a] = -999
            tmp_len = len(test[b_vals.isnull()])
            if tmp_len > 0:
                test.loc[b_vals.isnull(), b] = -999

    # Training
    t0 = time.time()
    clf = ExtraTreesClassifier()
    clf.set_params(**cfg[s]["estimator_params_etc"])
    X, X_eval, y, y_eval = cv.train_test_split(train, target, test_size=0.4)

    if cfg[s]["find_best"] == True:
        model = utils.find_best_estimator(clf, X, y, cfg, section=s,
                                          grid_search_params_key="gs_params_etc",
                                          scoring="log_loss", verbosity=2)
        logger.info(model)
    else:
        model = clf.fit(X, y)
        logger.info("%.2f seconds to train %s" % ((time.time() - t0), model))

    preds = model.predict_proba(X_eval)[:, 1]
    log_loss = metrics.log_loss(y_eval, preds)
    logger.info("Log loss : %.6f" % log_loss)

    logger.info("Making predictions..")
    y_pred = model.predict_proba(test)
    utils.make_submission_file(y_pred[:, 1], "etc_")
