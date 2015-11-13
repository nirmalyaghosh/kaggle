# -*- coding: utf-8 -*-
"""
Script for training various models.
"""

from sklearn import grid_search, svm
from sklearn.calibration import CalibratedClassifierCV
import csv
import gzip
import logging
#import numpy as np
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
        "kernel" : ["linear", "rbf"], "C" : [1, 10]
    }
    svc = svm.SVC(decision_function_shape='ovo', random_state=720, 
                  probability = True)
    clf = grid_search.GridSearchCV(svc, gs_params, n_jobs=2, verbose=2)
    train_and_make_predictions(clf, X, Y, test, "02 - SVC (apply 'ovo')")


if __name__ == '__main__':
    train_model_02()
