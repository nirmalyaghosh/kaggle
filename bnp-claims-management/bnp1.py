# -*- coding: utf-8 -*-
"""
Approach 1 : Apply Random Forest with 13 of 19 categorical variables
One-hot encoded, with the rest discarded.

@author: Nirmalya Ghosh
"""

import pandas as pd
from sklearn import cross_validation as cv
from sklearn import ensemble
from sklearn import metrics

import utils
from bnp_config import cfg, logger


if __name__ == "__main__":
    section = "approach1"
    logger.info("Running script for BNP Approach 1")
    
    train = pd.read_csv("data/train.csv.gz")
    test = pd.read_csv("data/test.csv.gz")
    id_train = train.ID
    id_test = test.ID
    target = train.target
    train = train.drop(["target"], axis=1)

    # Combining the train and test to do some preprocessing
    df_all = pd.concat((train, test), axis=0, ignore_index=True) # 228714 x 132

    # Deal with missing values
    df_all_nmv = utils.BasicImputer().fit_transform(df_all)

    # Deal with categorical variables
    cat_cols_w_len = utils.count_column_unique_values(df_all_nmv)
    logger.info("Categorical columns (with # unique values): %s" %
                cat_cols_w_len)
    cat_cols = [t[0] for t in cat_cols_w_len]
    max_unique_values = cfg[section]["max_unique_values"]
    # Only interested in those with fewer unique values
    filtered = [t for t in cat_cols_w_len if t[1] <= max_unique_values]
    filtered_cols = [t[0] for t in filtered]
    remaining_cols = sorted(list(set(cat_cols) - set(filtered_cols)))
    logger.info("Columns with under %d unique values : %s" %
                 (max_unique_values, filtered_cols))
    df_all = utils.convert_categorical_features(df_all_nmv, filtered_cols, False)
    logger.info("%d columns after converting. Was %d before." %
                (df_all.shape[1], df_all_nmv.shape[1]))

    # Drop the remaining categorical columns (ones we did not convert)
    col_names = list(df_all.columns.values)
    logger.info("Categorical columns not converted : %s" % remaining_cols)
    df_all = df_all.drop(remaining_cols, axis=1)
    logger.info("%d columns after dropping remaining categorical columns." %
                df_all.shape[1])

    # Separating the train and test
    train = df_all[df_all["ID"].isin(id_train)]
    test = df_all[df_all["ID"].isin(id_test)]

    logger.info("Training model. Train dataset shape : %s" % str(train.shape))
    X, X_eval, y, y_eval = cv.train_test_split(train, target, test_size=0.4)
    preds = None
    clf = ensemble.RandomForestClassifier()
    clf.set_params(**cfg[section]["best_estimator"])
    if cfg[section]["find_best"] == True:
        model = utils.find_best_estimator(clf, X, y, cfg, section=section,
                                          grid_search_params_key="param_dist",
                                          scoring="f1", verbosity=2)
    else:
        model = clf.fit(X, y)
        preds = model.predict_proba(X_eval)[:, 1]
        log_loss = metrics.log_loss(y_eval, preds)
        logger.info("Trained model %s" % model)
        logger.info("Log loss : %.6f" % log_loss)

    logger.info("Making predictions..")
    predicted_probabilities = model.predict_proba(test)[:, 1]
    utils.make_submission_file(predicted_probabilities, "simple-randomforest")
