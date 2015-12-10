# -*- coding: utf-8 -*-
"""
Script for training various models.
"""

from operator import itemgetter
from sklearn import grid_search, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import csv
import gc
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


def prepare_data_for_training(dataset_id):
    train, test = read_dataset(dataset_id)
    if dataset_id==1:
        test = test.drop(test.columns[0], axis=1)
    l = list(train.columns.values)
    X = train[l[:-1]]
    Y = train[l[-1]]
    return X, Y, test


def read_dataset(dataset_id):
    logging.info("Reading dataset %d" % dataset_id)
    train = pd.read_csv('data/train-dataset-%d.txt.gz' % dataset_id, sep='\t', 
                        index_col=False)
    test = pd.read_csv('data/test-dataset-%d.txt.gz' % dataset_id, sep='\t', 
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


def run_randomized_search_cv(dataset_id, candidate_classifier="RandomForest", 
                             cv_nfold=10, num_parallel_jobs=-1):
    train, test = read_dataset(dataset_id)
    test = test.drop(test.columns[0], axis=1)
    l = list(train.columns.values)
    
    logging.info("Preparing X & Y")
    X = train[l[:-1]]
    Y = train[l[-1]]
    
    # The classifier being used
    (clf, param_dist) = None, {}
    if candidate_classifier == "decisiontree":
        clf = DecisionTreeClassifier()
        param_dist = {"max_depth": list(np.arange(108, 243, 3)),
                      "min_samples_split": list(np.arange(140, 305, 5)),
                      "min_samples_leaf": list(np.arange(3, 12)),
                      "criterion": ["gini", "entropy"]
                     }
    elif candidate_classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=600)
        param_dist = {"max_depth": list(np.arange(108, 153, 3)),
                      #"max_features": sp_randint(1, 5356),
                      "min_samples_split": list(np.arange(140, 185, 5)),
                      "min_samples_leaf": list(np.arange(3, 9)),
                      "bootstrap": [True, False],
                      "criterion": ["gini", "entropy"]
                     }
    elif candidate_classifier == "KNN":
        clf = KNeighborsClassifier()
        param_dist = {"metric": ["minkowski", "euclidean", "manhattan"],
                      "weights": ["uniform", "distance"],
                      "leaf_size": np.arange(5,105,5),
                      "n_neighbors": np.arange(5,105,5)
                     }
    # NOTE : Unable to complete random search for KNN
    # getting "WindowsError: [Error 5] Access is denied"
    elif candidate_classifier == "NB":
        clf = MultinomialNB()
        param_dist = {"alpha": np.arange(0,1.55,0.05)
                     }
    elif candidate_classifier == "SVM":
        clf = SVC(probability=True)
        param_dist = {"C": np.arange(1,10), 
                      "kernel": ["linear"], 
                      "tol": [0.001, 0.01, 0.015, 0.1]
                     }
    
    # Run randomized search
    gc.collect()
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search, 
                                       n_jobs=num_parallel_jobs, 
                                       cv=cv_nfold, verbose=3, 
                                       random_state=720)
    
    logging.info("Starting the RandomizedSearchCV")
    logging.info(random_search)
    start = time.time()
    random_search.fit(X, Y)
    logging.info("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time.time() - start), n_iter_search))
    report(random_search.grid_scores_)


def train_and_make_predictions(clf, X, Y, test, model_desc):
    logging.info("Training model %s" % model_desc)
    model = clf.fit(X, Y)
    try:
        logging.info(model)
    except:
        print "exception printing model"
    logging.info("Making predictions")
    predicted_probabilties = clf.predict_proba(test)
    make_submission_file(predicted_probabilties, test, True)


def train_model_01():
    X, Y, test = prepare_data_for_training(1)
    clf = CalibratedClassifierCV(svm.LinearSVC())
    train_and_make_predictions(clf, X, Y, test, "01 (LinearSVC)")


def train_model_02():
    X, Y, test = prepare_data_for_training(1)
    gs_params = { 
        "kernel" : ["linear"], "C" : [1, 5, 10]
        # rbf TOO SLOW
    }
    svc = svm.SVC(decision_function_shape='ovo', random_state=720, 
                  probability=True)
    clf = grid_search.GridSearchCV(svc, gs_params, n_jobs=1, verbose=6)
    train_and_make_predictions(clf, X, Y, test, "02 - SVC (apply 'ovo')")


def train_model_03(dataset_id):
    # Random Forest
    X, Y, test = prepare_data_for_training(dataset_id)
    clf = RandomForestClassifier(n_estimators=300, min_samples_split=150, 
                                 bootstrap=False, criterion="gini", 
                                 max_depth=117, min_samples_leaf=3, n_jobs=-1)
    train_and_make_predictions(clf, X, Y, test, 
                               "RandomForest %s" % clf.get_params())


def train_model_04():
    X, Y, test = prepare_data_for_training(1)
    clf = RandomForestClassifier(n_estimators=150, min_samples_split=160, 
                                 bootstrap=False, criterion="gini", 
                                 max_depth=87, min_samples_leaf=3, n_jobs=-1)
    train_and_make_predictions(clf, X, Y, test, "RF150")


if __name__ == '__main__':
    run_randomized_search_cv(2, "decisiontree", cv_nfold=10, 
                             num_parallel_jobs=7)
    # NOTE: Some classifiers give WindowsError when num_parallel_jobs > 1
    # Classifiers which are OK to set num_parallel_jobs > 1 (Windows version): 
    # - RandomForest
    # - SVM (sklearn.svm.SVC)
    
    #train_model_03(2)
    #run_randomized_search_cv_2(2)
