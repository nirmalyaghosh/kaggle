# -*- coding: utf-8 -*-
"""
Common code as well as utility classes/functions.

@author: Nirmalya Ghosh and others (where credit is due)
"""

import csv
import gzip
import json
import logging
import numpy as np
import pandas as pd
import operator
import random
import shutil
import time

from sklearn import grid_search
from sklearn.base import TransformerMixin

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)-15s : %(message)s", )


def convert_categorical_features(df, cat_features, random_delete_one=False):
    for f in cat_features:
        dummy_cols_df = pd.get_dummies(df[f], prefix=f)
        cols = list(dummy_cols_df.columns.values)
        num_ones = [int(dummy_cols_df[col].sum()) for col in cols]
        cols_w_len = zip(cols, num_ones)
        logging.info("Categorical feature '%s' has %d unique values "
                     "with distribution %s" % (f, len(cols), cols_w_len))
        if random_delete_one == True:
            # Deleting one of the dummy variables
            col = random.choice(list(dummy_cols_df.columns.values))
            logging.info("Deleting column %s" % col)
            dummy_cols_df.drop([col], axis=1, inplace=True)
            # Doing so helps avoid the Multicollinearity problem.
            # Tip Credit : http://stackoverflow.com/a/22130844
        df = df.drop([f], axis=1)
        df = pd.concat((df, dummy_cols_df), axis=1)
    
    return df


def count_column_unique_values(df, count_only_categorical=True):
    if count_only_categorical == True:
        col_names = list(df.columns.values)
        col_types = list(df.dtypes)
        cols = [col_names[x] for x in range(0, len(col_types))
                if col_types[x] == "object"]
    else:
        cols = list(df.columns.values)
    lengths = [len(df[col].unique()) for col in cols]
    return zip(cols, lengths)  # List of tuples of column values and counts


def find_best_estimator(base_estimator, X, y, cfg, section,
                        random_search=True, scoring="accuracy", verbosity=3):
    cv_nfold = cfg.getint(section, "cv_nfold")
    name = type(base_estimator).__name__
    n_iter = cfg.getint(section, "n_iters")
    n_jobs = cfg.getint(section, "n_jobs")
    param_dist = json.loads(cfg.get(section, "param_dist"))
    random_state = cfg.getint("common", "seed")
    logging.info("Finding the best %s based on %s score" % (name, scoring))
    if random_search == True:
        logging.info("Using random search to find the best %s" % name)
        search = grid_search.RandomizedSearchCV(estimator=base_estimator,
                                                param_distributions=param_dist,
                                                n_iter=n_iter,
                                                n_jobs=n_jobs,
                                                cv=cv_nfold,
                                                random_state=random_state,
                                                scoring=scoring,
                                                verbose=verbosity)
    else:
        logging.info("Using grid search to find the best %s" % name)
        search = grid_search.GridSearchCV(estimator=base_estimator,
                                          param_grid=param_dist,
                                          n_jobs=n_jobs,
                                          cv=cv_nfold,
                                          verbose=verbosity)

    logging.info(search)
    start = time.time()
    search.fit(X, y)
    logging.info("Took %.2f seconds to find the best %s." %
                 ((time.time() - start), name))
    report_grid_search_scores(search.grid_scores_, n_top=3)
    return search.best_estimator_


def make_submission_file(predicted_vals, name_prefix, create_gz=True):
    current_ts = time.strftime("%a_%d%b%Y_%H%M%S")
    submission_filepath = "submissions/%s%s.csv" % (name_prefix, current_ts)
    submission = pd.read_csv("data/sample_submission.csv")
    submission.PredictedProb = predicted_vals
    submission.to_csv(submission_filepath, index=False, quoting=csv.QUOTE_NONE)
    if create_gz == False:
        logging.info("See %s" % submission_filepath)
        return
    with open(submission_filepath, 'rb') as f_in, \
        gzip.open(submission_filepath + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    logging.info("See %s.gz" % submission_filepath)
    return submission


def read_estimator_params(cfg, section, key="best_estimator"):
    return json.loads(cfg.get(section, key))


def report_grid_search_scores(grid_scores, n_top=5):
    # Utility function to report best scores
    # Credit : http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    top_scores = sorted(grid_scores, key=operator.itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        logging.info("Model with rank: {0}".format(i + 1))
        logging.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                     score.mean_validation_score,
                     np.std(score.cv_validation_scores)))
        logging.info("Parameters: {0}".format(score.parameters))
        logging.info("")


class BasicImputer(TransformerMixin):
    """
    Given a Pandas dataframe, imputes missing values, including categorical.
    
    Columns of dtype object are imputed with the most frequent value
    in column.
    
    Columns of other types are imputed with mean of column.
    
    Credit : http://stackoverflow.com/a/25562948
    """
    
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('O') else X[c].mean()
                               for c in X],
                              index=X.columns)
        
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)