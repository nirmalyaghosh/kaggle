# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy import interp
from scipy.stats import mstats
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import cross_val_score


def describe_col(_df1, _df2, _col_name, index=["train", "test"]):
    describe_df = pd.concat([pd.DataFrame([_df1[_col_name].describe()]),
                             pd.DataFrame([_df2[_col_name].describe()])])
    describe_df.index = index
    return describe_df


def double_mad_based_outliers(points, thresh=3.5):
    # Double MAD: (1) Calculate the MAD from the median of all points less
    # than or equal to the median and (2) the MAD from the median of all
    # points greater than or equal to the median.
    # Credit : http://stackoverflow.com/a/29222992
    m = np.median(points)
    abs_dev = np.abs(points - m)
    left_mad = np.median(abs_dev[points <= m])
    right_mad = np.median(abs_dev[points >= m])
    y_mad = left_mad * np.ones(len(points))
    y_mad[points > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[points == m] = 0
    return modified_z_score > thresh


def generate_bucket_col_name(col_name):
    bkt_col_name = "{}_bucket".format(col_name).lower() \
        .replace(" ", "").replace(".0", "")
    return bkt_col_name


def get_bucket_column_summary(_df, col_name, num_buckets=10):
    # Stores the details into a DataFrame
    # - helps to get an idea of the min, max of each bucket
    bkt_col_name = generate_bucket_col_name(col_name)
    bucket_tuples = []
    for bin_id in range(num_buckets):
        desc = _df[_df[bkt_col_name] == bin_id][col_name].describe()
        t = (bin_id, desc["min"], desc["max"], desc["count"])
        bucket_tuples.append(t)
    bucket_df = pd.DataFrame(bucket_tuples, columns=[bkt_col_name, "min_value",
                                                     "max_value",
                                                     "num_records"])
    return bucket_df


def get_cv_scores(est, X, Y, scorer_name, nfold=5):
    est_name = est.__class__.__name__
    logging.info("Getting {}-fold CV score for {} model" \
                 .format(nfold, est_name))
    scores = cross_val_score(est, X, Y, cv=nfold, scoring=scorer_name)
    logging.info("{}-fold CV '{}' score for {} model : {:.4f}" \
                 .format(nfold, scorer_name, est_name, scores.mean()))
    return scores


def get_winsorized_version(x, cut_off_percentage=0.05):
    # Returns a Winsorized version of the input,
    # with specified cut off either end
    return mstats.winsorize(x, limits=[cut_off_percentage, cut_off_percentage])


def jitter(a_series, noise_reduction=1000000):
    # Credit: http://stackoverflow.com/a/37793940
    return (np.random.random(
        len(a_series)) * a_series.std() / noise_reduction) - (
               a_series.std() / (2 * noise_reduction))


def log_column_NA_counts(_df):
    for col_name in _df.columns.values.tolist():
        na_count = _df[col_name].isnull().sum()
        if na_count > 0:
            logging.info("'{}' column has {} NAs" \
                         .format(col_name, na_count))


def log_important_features(fitted_est, column_names, top_n=15):
    features = pd.DataFrame()
    features["feature"] = column_names
    features["importance"] = fitted_est.feature_importances_
    features = features.sort_values(by=["importance"], ascending=False)
    logging.info("Top {} Features : \n{}".format(top_n, features.head(top_n)))


def mad_based_outliers(points, thresh=3.5):
    # Credit : http://stackoverflow.com/a/22357811
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median) ** 2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh


def normalize_df(_df):
    scaler = preprocessing.MinMaxScaler()
    scaled = scaler.fit_transform(_df.values)
    normalized = pd.DataFrame(scaled, columns=_df.columns.values.tolist())
    normalized.index = _df.index
    _df = normalized
    return _df, scaler


def percentile_based_outliers(data, threshold=95):
    # Credit : http://stackoverflow.com/a/22357811
    diff = (100 - threshold) / 2.0
    minval, maxval = np.percentile(data, [diff, 100 - diff])
    return (data < minval) | (data > maxval)


def plot_outliers(x, indices, p, col_name):
    # Calculates and plots the outliers
    # Adapted from : http://stackoverflow.com/a/22357811
    logging.info("Plotting outliers for '{}' on {}% sample ({} records)" \
                 .format(col_name, p, len(x)))
    fig, axes = plt.subplots(figsize=(10, 8), nrows=4)

    # First, find the outliers using the 3 methods
    outliers = {}
    for ax, func in zip(axes, [percentile_based_outliers, mad_based_outliers,
                               double_mad_based_outliers]):
        sb.distplot(x, ax=ax, rug=True, hist=False, kde_kws={"label": "KDE"},
                    rug_kws={"color": "g"})
        raw = func(x)
        outlier_values = x[raw] if sum(raw) > 0 else np.array([])
        outliers[func.__name__] = raw
        logging.info("{} {}".format(len(outlier_values), func.__name__))
        ax.plot(outlier_values, np.zeros_like(outlier_values), "ro",
                clip_on=False)

    # Next, take majority vote amongst the outliers determined by the 3 above
    z = zip(x, indices.tolist(), *list(outliers.values()))
    majority = []  # outlier_values
    for v, i, a, b, c in z:
        if sum([a, b, c]) > 2:
            majority.append(v)
    logging.info("{} outliers based on majority of the 3 methods above" \
                 .format(len(majority)))
    outliers["majority_vote"] = majority
    ax = axes[3]
    sb.distplot(x, ax=ax, rug=True, hist=False, kde_kws={"label": "KDE"},
                rug_kws={"color": "g"})
    ax.plot(majority, np.zeros_like(majority), "ro", clip_on=False)

    kwargs = dict(y=0.95, x=0.05, ha="left", size=10, va="top")
    suffix = " for '{}'".format(col_name)
    axes[0].set_title("Percentile-based Outliers" + suffix, **kwargs)
    axes[1].set_title("MAD-based Outliers" + suffix, **kwargs)
    axes[2].set_title("Double-MAD-based Outliers" + suffix, **kwargs)
    axes[3].set_title("Majority Vote Amongst The 3 Above" + suffix, **kwargs)
    fig.suptitle("'{}' Of {}% sample ({} records) {} outliers" \
                 .format(col_name, p, len(x),
                         ("with" if len(majority) > 0 else ", no")), size=12)
    return outliers


def plot_outliers_for(col_name, _df, p):
    df_tmp = _df.sample(n=int(_df.shape[0] / 100) * p, random_state=329521)
    x = df_tmp[col_name].values
    indices = df_tmp[col_name].index
    outliers = plot_outliers(x, indices, p, col_name)
    return outliers


def replace_outliers_in_df(_df, col_name):
    _df1 = _df[pd.notnull(_df[col_name])]
    x = _df1[col_name].values
    indices = _df1.index.tolist()  # starts at 1, ends at 150000
    logging.info("Mean, Median {} before : {}, {}" \
                 .format(col_name, np.mean(x), np.median(x)))

    # Calculate Winsorized version of the list of values
    xnew = get_winsorized_version(x)
    # Update the DataFrame
    S = np.array(_df[col_name].values.tolist())
    indices = [i - 1 for i in indices]
    S[indices] = xnew
    _df[col_name] = S.tolist()
    # Check afterwards
    _df1 = _df[pd.notnull(_df[col_name])]
    x = _df1[col_name].values
    logging.info("Mean, Median {} after  : {}, {}" \
                 .format(col_name, np.mean(x), np.median(x)))
    return _df


def split_into_buckets(_df, col_name, num_buckets, convert_into_dummies=True,
                       drop_original_col=True, drop_bucket_col=True,
                       add_jitter=False):
    bkt_col_name = generate_bucket_col_name(col_name)

    if add_jitter == False:
        _df[bkt_col_name] = pd.qcut(_df[col_name], num_buckets, labels=False)
    else:
        s = _df[col_name]
        _df[bkt_col_name] = pd.qcut(s + jitter(s), num_buckets, labels=False)

    if convert_into_dummies == True:
        bkt_dummies = pd.get_dummies(_df[bkt_col_name], prefix=bkt_col_name)
        _df = pd.concat([_df, bkt_dummies], axis=1)
        if drop_bucket_col == True:
            _df.drop([bkt_col_name], axis=1, inplace=True)

    if drop_original_col == True:
        _df.drop([col_name], axis=1, inplace=True)

    return _df


def train_estimator(est, X, y, n_folds):
    # Adapted from : https://git.io/vXu0Z
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    logging.info("Training {}".format(est.__class__.__name__))
    for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        probas_ = est.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        logging.info("ROC fold {} (area = {:.2f})".format(i, roc_auc))

    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc_str = "Mean ROC (area = {:.2f})".format(mean_auc)
    logging.info(mean_auc_str)
    return est


def unscale_column_values(predictions, col_idx, scaler):
    unscaled_values = []
    for val in predictions:
        _scale = scaler.scale_[col_idx]
        unscaled_val = (val - scaler.min_[col_idx]) / _scale
        unscaled_values.append(int(unscaled_val))

    logging.info("{} values unscaled".format(len(unscaled_values)))
    return unscaled_values
