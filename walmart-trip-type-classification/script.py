# -*- coding: utf-8 -*-
"""
Script for training various models.
"""

from operator import itemgetter
from sklearn import grid_search, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
import csv
import gzip
import logging
import numpy as np
import pandas as pd
import shutil
import time


column_names = [ "TripType_3",  "TripType_4",  "TripType_5",  "TripType_6", 
                 "TripType_7",  "TripType_8",  "TripType_9",  "TripType_12",
                 "TripType_14", "TripType_15", "TripType_18", "TripType_19",
                 "TripType_20", "TripType_21", "TripType_22", "TripType_23",
                 "TripType_24", "TripType_25", "TripType_26", "TripType_27",
                 "TripType_28", "TripType_29", "TripType_30", "TripType_31",
                 "TripType_32", "TripType_33", "TripType_34", "TripType_35",
                 "TripType_36", "TripType_37", "TripType_38", "TripType_39",
                 "TripType_40", "TripType_41", "TripType_42", "TripType_43",
                 "TripType_44", "TripType_999"]
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s : %(message)s",)


def make_submission_file(predicted_probabilties, test, create_gz = True):
    current_ts = time.strftime("%a_%d%b%Y_%H%M%S")
    submission_filepath = "submissions/%s.csv" % current_ts
    df = pd.DataFrame(predicted_probabilties, columns = column_names)
    df.insert(0, "VisitNumber", test.VisitNumber)
    df.to_csv(submission_filepath, index=False, 
          quoting=csv.QUOTE_NONNUMERIC)
    if create_gz == False:
        logging.info("See %s" % submission_filepath)
        return
    with open(submission_filepath, 'rb') as f_in, \
         gzip.open(submission_filepath+'.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    logging.info("See %s.gz" % submission_filepath)
    
#==============================================================================
#     np.savetxt(submission_filepath, proba, delimiter=",", fmt='%0.5f', 
#                header="""VisitNumber", "TripType_3", "TripType_4", 
#                "TripType_5", "TripType_6", "TripType_7", "TripType_8", 
#                "TripType_9", "TripType_12", "TripType_14", "TripType_15", 
#                "TripType_18", "TripType_19", "TripType_20", "TripType_21",
#                "TripType_22", "TripType_23", "TripType_24", "TripType_25",
#                "TripType_26", "TripType_27", "TripType_28", "TripType_29",
#                "TripType_30", "TripType_31", "TripType_32", "TripType_33",
#                "TripType_34", "TripType_35", "TripType_36", "TripType_37",
#                "TripType_38", "TripType_39", "TripType_40", "TripType_41", 
#                "TripType_42", "TripType_43", "TripType_44", "TripType_999""")
#==============================================================================


def read_dataset(idx):
    logging.info("Reading dataset %d" % idx)
    train = pd.read_csv('data/train-dataset-%d.txt.gz' % idx, sep='\t', 
                        index_col=False)
    test = pd.read_csv('data/test-dataset-%d.txt.gz' % idx, sep='\t', 
                       index_col=False)
    return train, test


# Utility function to report best scores
# Credit : http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        logging.info("Model with rank: {0}".format(i + 1))
        logging.info("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                     score.mean_validation_score,
                     np.std(score.cv_validation_scores)))
        logging.info("Parameters: {0}".format(score.parameters))
        logging.info("")


def run_randomized_search_cv():
    train, test = read_dataset(1)
    test = test.drop(test.columns[0], axis=1)
    l = list(train.columns.values)
    
    logging.info("Preparing X & Y")
    X = train[l[:-1]]
    Y = train[l[-1]]
    
    # The classifier being used
    clf = RandomForestClassifier(n_estimators=150)
    
    # Specify parameters and distributions to sample from
    param_dist = {"max_depth": list(np.arange(81, 130, 3)),
                  #"max_features": sp_randint(1, 5356),
                  "min_samples_split": list(np.arange(60, 160, 5)),
                  "min_samples_leaf": list(np.arange(3, 8)),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}
    
    # Run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, n_jobs=8, cv=10,
                                       verbose=1, random_state=720)
    
    logging.info("Starting the RandomizedSearchCV")
    start = time.time()
    random_search.fit(X, Y)
    logging.info("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.grid_scores_)


def train_and_make_predictions(clf, X, Y, test, model_desc):
    logging.info("Training model %s" % model_desc)
    clf.fit(X, Y)
    logging.info("Making predictions")
    predicted_probabilties = clf.predict_proba(test)
    make_submission_file(predicted_probabilties, test, True)


def train_model_01():
    train, test = read_dataset(1)
    test = test.drop(test.columns[0], axis=1)
    l = list(train.columns.values)
    X = train[l[:-1]]
    Y = train[l[-1]]
    clf = CalibratedClassifierCV(svm.LinearSVC())
    train_and_make_predictions(clf, X, Y, test, "01 (LinearSVC)")


def train_model_02():
    train, test = read_dataset(1)
    test = test.drop(test.columns[0], axis=1)
    l = list(train.columns.values)
    X = train[l[:-1]]
    Y = train[l[-1]]
    gs_params = { 
        "kernel" : ["linear"], "C" : [1, 5, 10]
        # rbf TOO SLOW
    }
    svc = svm.SVC(decision_function_shape='ovo', random_state=720, 
                  probability=True)
    clf = grid_search.GridSearchCV(svc, gs_params, n_jobs=1, verbose=6)
    train_and_make_predictions(clf, X, Y, test, "02 - SVC (apply 'ovo')")


def train_model_03():
    train, test = read_dataset(1)
    test = test.drop(test.columns[0], axis=1)
    l = list(train.columns.values)
    X = train[l[:-1]]
    Y = train[l[-1]]
    clf = RandomForestClassifier(n_estimators=150, min_samples_split=145, 
                                 bootstrap=False, criterion="gini", 
                                 max_depth=102, min_samples_leaf=4, n_jobs=-1)
    train_and_make_predictions(clf, X, Y, test, "RF150")


if __name__ == '__main__':
    train_model_03()
    #run_randomized_search_cv()
