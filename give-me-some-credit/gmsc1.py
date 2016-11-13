# -*- coding: utf-8 -*-
"""
Approach 1 : Get rid of outliers, predict/impute the missing values,
split into near-equal sized buckets a few continuous variables,
then under-sample the majority class to deal with class imbalance and
finally build a few simple models to predict delinquency.

@author: Nirmalya Ghosh
"""

import logging
import os

import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

# from sklearn.svm import SVC

import feats
import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.FileHandler("gmsc.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

colmap = {"NumberOfTime30-59DaysPastDueNotWorse": "NumLate3059",
          "NumberOfOpenCreditLinesAndLoans": "NumOCLL",
          "NumberOfTimes90DaysLate": "NumLate90",
          "NumberRealEstateLoansOrLines": "NumRELL",
          "NumberOfTime60-89DaysPastDueNotWorse": "NumLate6089",
          "NumberOfDependents": "NumDependents",
          "RevolvingUtilizationOfUnsecuredLines": "RUoUL"
          }

rs = 329521  # random_state


def _main():
    np.random.seed(rs)
    logger.info("Running script for Approach 1")
    tr_df = pd.read_csv(os.path.join("data", "cs-training.csv"), index_col=0)
    te_df = pd.read_csv(os.path.join("data", "cs-test.csv"), index_col=0)
    tr_df, te_df = _preprocess_data(tr_df, te_df)

    # Add features
    tr_df, te_df = feats.add_features_based_on_NumOCLL(tr_df, te_df)
    tr_df, te_df = feats.add_features_based_on_NumRELL(tr_df, te_df)
    tr_df, te_df = feats.add_features_based_on_RUoUL(tr_df, te_df)

    # Preparing dataset for training
    excluded_cols = ["age", "MonthlyIncome", "MonthlyIncome_Imputed",
                     "SeriousDlqin2yrs"]
    train_df = tr_df[tr_df.columns.difference(excluded_cols)]
    cols = train_df.columns.values.tolist()
    X, _ = utils.normalize_df(train_df)
    X = X.as_matrix()
    y = tr_df["SeriousDlqin2yrs"].values

    # Split
    sss = StratifiedShuffleSplit(n_splits=3, random_state=rs, test_size=0.3)
    for train_index, test_index in sss.split(X, y):
        X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], y[
            train_index], y[test_index]

    logger.info("X {}, train {}, valid {}" \
                .format(X.shape, X_train.shape, X_valid.shape))

    # Train
    logger.info("Features used for training : {}".format(cols))
    base_estimators = [
        ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=rs),
        LogisticRegressionCV(random_state=rs),
        RandomForestClassifier(bootstrap=True, criterion="gini",
                               max_depth=None, max_features=5,
                               n_estimators=150, n_jobs=-1, random_state=rs),
        # SVC(C=0.01, gamma=0.01, kernel="rbf", probability=True,
        #     random_state=rs)
    ]

    # Each classifier is trained on 5 stratified splits
    # and the one (amongst the 5) with best AUC score is selected
    best_auc = 0.0
    common_top_n_features = []
    for est in base_estimators:
        fitted_est = utils.train_estimator(est, X_train, y_train, 5)
        top_n_features = []
        top_n_features_df = utils.log_important_features(est, cols)
        if top_n_features_df.shape[0] > 0:
            top_n_features = top_n_features_df.head(15).feature.values.tolist()
        common_top_n_features.extend(top_n_features)
        common_top_n_features = list(set(common_top_n_features))
        logger.info("{} common_top_n_features : {}" \
                    .format(len(common_top_n_features), common_top_n_features))
        preds = fitted_est.predict(X_valid)
        score = roc_auc_score(y_valid, preds)
        logger.info("AUC : {:.5f}".format(score))
        if score > best_auc:
            best_auc = score
            best_est = fitted_est

    logger.info("Best estimator : {}".format(best_est))

    # Re-fitting the best estimator using the common top N features
    refit = False  # TODO read from config
    if refit == True:
        logger.info("Re-fitting best estimator {} using top N features ..." \
                    .format(best_est.__class__.__name__))
        X, _ = utils.normalize_df(train_df[common_top_n_features])
        X = X.as_matrix()
        y = tr_df["SeriousDlqin2yrs"].values
        sss = StratifiedShuffleSplit(n_splits=3, random_state=rs,
                                     test_size=0.3)
        for train_index, test_index in sss.split(X, y):
            X_train, X_valid, y_train, y_valid = X[train_index], X[test_index], \
                                                 y[train_index], y[test_index]
        fitted_best_est = utils.train_estimator(best_est, X_train, y_train, 5)
        preds = fitted_best_est.predict(X_valid)
        score = roc_auc_score(y_valid, preds)
        logger.info("AUC : {:.5f}".format(score))
        if score > best_auc:
            best_auc = score
            best_est = fitted_est

    # Getting the predictions
    logger.info("Get the predictions using {} ...".format(best_est))
    te_df_, _ = utils.normalize_df(te_df[cols])
    identifiers = te_df_.index.tolist()
    if refit == True:
        p = [x[1] for x in
             best_est.predict_proba(te_df_[common_top_n_features])]
    else:
        p = [x[1] for x in best_est.predict_proba(te_df_)]
    _prepare_submission_file(identifiers, p)


def _handle_class_imbalance(_df, is_train=True):
    df_name = "Train" if is_train == True else "Test"
    logger.info("Handling class imbalance in {} ...".format(df_name))

    # First, get the data in shape
    df2 = _df.copy()
    y = df2.SeriousDlqin2yrs.values
    df_tmp = _df[["MonthlyIncome"]]
    df2.drop(["MonthlyIncome", "SeriousDlqin2yrs"], axis=1, inplace=True)

    # Under-sampling the majority class in train data
    logger.info("Under-sampling the majority class in {} ...".format(df_name))
    imb_handler = RandomUnderSampler()

    X = df2.as_matrix()
    if is_train == True:
        X_resampled, y_resampled = imb_handler.fit_sample(X, y)
    else:
        X_resampled, y_resampled = imb_handler.fit_sample(X, [0] * len(y))

    df2x = pd.DataFrame(X_resampled, columns=df2.columns.values)
    df2y = pd.DataFrame(y_resampled, columns=["SeriousDlqin2yrs"])
    df2 = pd.merge(df2x, df2y, left_index=True, right_index=True)
    df2 = pd.merge(df2, df_tmp, left_index=True, right_index=True)
    del df2x, df2y

    _df = df2[_df.columns.values.tolist()]
    del df2

    return _df


def _predict_monthly_income(tr_df, te_df):
    logger.info("Preparing dataset to train model to predict MonthlyIncome")
    mask = np.logical_not(tr_df.MonthlyIncome.isnull())
    tr_tr = tr_df[mask]  # Train's training data (has MonthlyIncome)
    tr_te = tr_df[tr_df.MonthlyIncome.isnull()]  # Train's test data
    mask = np.logical_not(te_df.MonthlyIncome.isnull())
    te_tr = te_df[mask]
    te_te = te_df[te_df.MonthlyIncome.isnull()]
    logger.info("tr_tr, tr_te : {},{}".format(tr_tr.shape, tr_te.shape))
    logger.info("te_tr, te_te : {},{}".format(te_tr.shape, te_te.shape))

    # Prepare the dataset : Normalizing the dataset
    tr_tr, scaler_1 = utils.normalize_df(tr_tr)
    tr_te.drop(["MonthlyIncome"], axis=1, inplace=True)  # Temporarily
    tr_te, _ = utils.normalize_df(tr_te)
    tr_te["MonthlyIncome"] = None  # add it back in
    te_tr, scaler_2 = utils.normalize_df(te_tr)
    te_te.drop(["MonthlyIncome"], axis=1, inplace=True)  # Temporarily
    te_te, _ = utils.normalize_df(te_te)
    te_te["MonthlyIncome"] = None  # add it back in

    # Prepare the dataset : split
    cols = ["RUoUL", "age", "NumLate3059", "NumLate6089", "NumLate90",
            "DebtRatio", "NumOCLL", "NumRELL", "NumDependents"]
    X_train, Y_train = tr_tr[cols], tr_tr[["MonthlyIncome"]]
    Y_train = Y_train.MonthlyIncome.ravel()
    X_test, Y_test = tr_te[cols], tr_te[["MonthlyIncome"]]
    Y_test = Y_test.MonthlyIncome.ravel()
    logger.info("X_train : {}, X_test : {}, Y_train : {}, Y_test : {}" \
                .format(X_train.shape, X_test.shape, Y_train.shape,
                        Y_test.shape))

    # Train the model
    pickle_file = "monthly_income_predictor.pkl"
    if os.path.exists(pickle_file):
        est = joblib.load(pickle_file)
    else:
        logger.info("Training model to predict MonthlyIncome")
        est = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                    random_state=329521)
        scorer_name = "neg_median_absolute_error"
        scores = utils.get_cv_scores(est, X_train, Y_train, scorer_name)
        est.fit(X_train, Y_train)
        joblib.dump(est, "monthly_income_predictor.pkl")

    # Predict the MonthlyIncome
    est_name = est.__class__.__name__
    logger.info("{} rows in tr_te missing MonthlyIncome".format(len(X_test)))
    logger.info("{} rows in te_te missing MonthlyIncome".format(len(te_te)))
    logger.info("Using {} to predict MonthlyIncome".format(est_name))
    predictions_1 = est.predict(X_test)
    predictions_2 = est.predict(te_te[cols])

    # Set the MonthlyIncome for the rows where it was missing (X_test & te_te)
    X_train["MonthlyIncome"] = Y_train  # Adding it back to X_train
    X_test["MonthlyIncome"] = predictions_1
    te_te["MonthlyIncome"] = predictions_2

    # Un-scale the MonthlyIncome - we used a MinMaxScaler earlier
    # First, un-scale  MonthlyIncome values in X_test
    # X_train constructed from tr_tr, X_test constructed from tr_te
    logger.info("Un-scaling the MonthlyIncome values in X_train & X_test")
    X_train["MonthlyIncome"] = \
        utils.unscale_column_values(X_train["MonthlyIncome"], 8, scaler_1)
    X_test["MonthlyIncome"] = \
        utils.unscale_column_values(predictions_1, 8, scaler_1)
    # Next, un-scale  MonthlyIncome values in te_te
    logger.info("Un-scaling the MonthlyIncome values in te_tr & te_te")
    te_tr["MonthlyIncome"] = \
        utils.unscale_column_values(te_tr["MonthlyIncome"], 8, scaler_2)
    te_te["MonthlyIncome"] = \
        utils.unscale_column_values(predictions_2, 8, scaler_2)

    # Concat the DataFrames because we now have the missing MonthlyIncome,
    tmp_df_1 = pd.concat([X_train, X_test])
    tmp_df_2 = pd.concat([te_tr, te_te])

    # Next, merge the tmp_df with the existing train/test datasets
    df0 = pd.read_csv(os.path.join("data", "cs-training.csv"), index_col=0)
    df0 = df0[["SeriousDlqin2yrs", "NumberOfDependents"]]
    df0.rename(columns={"NumberOfDependents": "NumDependents"}, inplace=True)
    tr_df = pd.merge(tr_df, df0, left_index=True, right_index=True)
    df1 = pd.merge(tr_df, tmp_df_1, left_index=True, right_index=True,
                   suffixes=("", "_y"))
    # Repeat, for test,
    df0 = pd.read_csv(os.path.join("data", "cs-test.csv"), index_col=0)
    df0 = df0[["SeriousDlqin2yrs", "NumberOfDependents"]]
    df0.rename(columns={"NumberOfDependents": "NumDependents"}, inplace=True)
    te_df = pd.merge(te_df, df0, left_index=True, right_index=True)
    df2 = pd.merge(te_df, tmp_df_2, left_index=True, right_index=True,
                   suffixes=("", "_y"))

    # Next, retains the columns we need - and, in the order we need
    cols = ["SeriousDlqin2yrs", "RUoUL", "age", "NumLate3059",
            "DebtRatio", "MonthlyIncome", "MonthlyIncome_y", "NumOCLL",
            "NumLate90", "NumRELL", "NumLate6089", "NumDependents"]
    df1, df2 = df1[cols], df2[cols]
    df1.rename(columns={"MonthlyIncome_y": "MonthlyIncome_Imputed"},
               inplace=True)
    df2.rename(columns={"MonthlyIncome_y": "MonthlyIncome_Imputed"},
               inplace=True)

    tr_df, te_df = df1, df2
    tr_df.to_csv(os.path.join("data", "tr_with_income.csv"))
    te_df.to_csv(os.path.join("data", "te_with_income.csv"))

    logger.info("Done predicting the missing MonthlyIncome ...")
    tr_df["MonthlyIncome"] = tr_df["MonthlyIncome_Imputed"]
    te_df["MonthlyIncome"] = te_df["MonthlyIncome_Imputed"]
    return tr_df, te_df


def _prepare_submission_file(identifiers, probabilities):
    subm_df = pd.DataFrame(list(zip(identifiers, probabilities)),
                           columns=["Id", "Probability"])
    expected = np.arange(1, 101504).tolist()
    missing_indices = list(set(expected).difference(identifiers))
    if missing_indices and len(missing_indices) > 0:
        logger.info("No probabilities for {} IDs".format(len(missing_indices)))
        tmp_df = pd.read_csv(os.path.join("data", "sampleEntry.csv"))
        tmp_df = tmp_df[tmp_df["Id"].isin(missing_indices)]
        subm_df = pd.concat([tmp_df, subm_df])
        subm_df = subm_df.sort_values(by=["Id"], ascending=[1])

    subm_df.to_csv("submission.csv", float_format="%.9f", index=False)


def _preprocess_data(tr_df, te_df):
    logger.info("Preprocessing the data ...")
    tr_df.rename(columns=colmap, inplace=True)
    te_df.rename(columns=colmap, inplace=True)

    # Imputing missing values for the 'NumDependents' column
    excluded_cols = ["MonthlyIncome", "SeriousDlqin2yrs"]
    _tmp = tr_df[tr_df.columns.difference(excluded_cols)]
    _tmp = utils.handle_missing_values(_tmp, "median")
    tr_df["NumDependents"] = _tmp["NumDependents"]
    tr_df["NumDependents"].fillna(0, inplace=True)  # Because 1 gets left out
    _tmp = te_df[te_df.columns.difference(excluded_cols)]
    _tmp = utils.handle_missing_values(_tmp, "median")
    te_df["NumDependents"] = _tmp["NumDependents"]
    te_df["NumDependents"].fillna(0, inplace=True)  # Because 1 gets left out

    # Do a bit of pre-processing (Train) : Replace the outliers
    logger.info("Replacing outliers in train data ...")
    tr_df = utils.replace_outliers_in_df(tr_df, "MonthlyIncome")
    tr_df = utils.replace_outliers_in_df(tr_df, "DebtRatio")
    tr_df = utils.replace_outliers_in_df(tr_df, "NumOCLL")
    tr_df = utils.replace_outliers_in_df(tr_df, "NumRELL")
    # Do a bit of pre-processing (Test) : Replace the outliers
    logger.info("Replacing outliers in test data ...")
    te_df = utils.replace_outliers_in_df(te_df, "MonthlyIncome")
    te_df = utils.replace_outliers_in_df(te_df, "DebtRatio")
    te_df = utils.replace_outliers_in_df(te_df, "NumOCLL")
    te_df = utils.replace_outliers_in_df(te_df, "NumRELL")

    # Next, drop the rows with 3 NumLate columns with strange values (Train)
    strange_values = [96, 98]
    mask = np.logical_not(tr_df["NumLate3059"].isin(strange_values))
    tr_df = tr_df[mask]
    # Next, drop the rows with 3 NumLate columns with strange values (Test)
    mask = np.logical_not(te_df["NumLate3059"].isin(strange_values))
    te_df = te_df[mask]

    # Next,
    tr_df.drop(["SeriousDlqin2yrs"], axis=1, inplace=True)
    te_df.drop(["SeriousDlqin2yrs"], axis=1, inplace=True)

    # Reorder the columns
    cols = ["RUoUL", "age", "NumLate3059", "NumLate6089", "NumLate90",
            "DebtRatio", "NumOCLL", "NumRELL", "MonthlyIncome",
            "NumDependents"]
    tr_df = tr_df[cols]
    te_df = te_df[cols]

    # Split age into 10 near-equal-sized buckets and convert to dummy variables
    tr_df = utils.split_into_buckets(tr_df, "age", 10, True, False)
    te_df = utils.split_into_buckets(te_df, "age", 10, True, False)

    # Next, predict the MonthlyIncome (where missing)
    tr_df, te_df = _predict_monthly_income(tr_df, te_df)

    # Split MonthlyIncome into 10 near-equal-sized buckets
    # and convert to dummy variables
    tr_df = utils.split_into_buckets(tr_df, "MonthlyIncome_Imputed", 10,
                                     drop_original_col=False,
                                     add_jitter=True)
    te_df = utils.split_into_buckets(te_df, "MonthlyIncome_Imputed", 10,
                                     drop_original_col=False,
                                     add_jitter=True)

    utils.log_column_NA_counts(tr_df)
    utils.log_column_NA_counts(te_df)

    # # Next, deal with the class imbalance
    # tr_df = _handle_class_imbalance(tr_df, True)
    # te_df = _handle_class_imbalance(te_df, False)

    return tr_df, te_df


if __name__ == "__main__":
    _main()
