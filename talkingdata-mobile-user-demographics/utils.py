# -*- coding: utf-8 -*-
"""
Common code as well as utility classes/functions.

@author: Nirmalya Ghosh and others (where credit is due)
"""

import csv
import gzip
import operator
import os
import shutil
import time

import numpy as np
import pandas as pd
from sklearn import grid_search
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix, hstack

from td_config import cfg, logger

clf_keys = { "ExtraTreesClassifier": "et",
             "KNeighborsClassifier": "knn",
             "LogisticRegression": "lr",
             "RandomForestClassifier": "rf",
             "SVC": "svm" }


def find_best_estimator(base_estimator, X, y, section, verbosity=3):
    # grid_search_params_key : key under the indicated section of the
    # configuration YML file containing the grid search parameters
    if cfg[section]["find_best"] == False:
        base_estimator.fit(X, y)
        return base_estimator

    cv_nfold = cfg[section]["cv_nfold"]
    name = type(base_estimator).__name__
    grid_search_params_key = "param_dist_%s" % clf_keys[name]
    n_iter = cfg[section]["n_iters"]
    n_jobs = cfg[section]["n_jobs"]
    param_dist = cfg[section][grid_search_params_key]
    random_state = cfg["common"]["seed"]
    scoring = cfg["common"]["grid_search_scoring"]
    if cfg[section]["use_random_search"] == True:
        logger.info("Using random search to find best %s based on %s score" %\
                    (name, scoring))
        search = grid_search.RandomizedSearchCV(estimator=base_estimator,
                                                param_distributions=param_dist,
                                                n_iter=n_iter,
                                                n_jobs=n_jobs,
                                                cv=cv_nfold,
                                                random_state=random_state,
                                                scoring=scoring,
                                                verbose=verbosity)
    else:
        logger.info("Using grid search to find best %s based on %s score" %\
                    (name, scoring))
        search = grid_search.GridSearchCV(estimator=base_estimator,
                                          param_grid=param_dist,
                                          n_jobs=n_jobs,
                                          cv=cv_nfold,
                                          scoring=scoring,
                                          verbose=verbosity)

    start = time.time()
    search.fit(X, y)
    logger.info("Took %.2f seconds to find the best %s." %
                ((time.time() - start), name))
    report_grid_search_scores(search.grid_scores_, n_top=3)
    logger.info(search.best_estimator_)
    return search.best_estimator_


def get_key(clf_name):
    return clf_keys[clf_name] if clf_name in clf_keys else None


def make_submission_file(model, predicted_vals, name_prefix):
    ts = time.strftime("%a_%d%b%Y_%H%M%S")
    # First, save the model [See http://stackoverflow.com/a/11169797]
    if model is not None:
        file_name_prefix = "%s%s.model" % (name_prefix, ts)
        _ = joblib.dump(model, os.path.join("submissions", file_name_prefix),
                        compress=9)
    # Next, generate the submissions file
    file_path = os.path.join("submissions", "%s%s.csv" % (name_prefix, ts))
    predicted_vals.to_csv(file_path, index=False,
                          quoting=csv.QUOTE_NONE)
    with open(file_path, 'rb') as f_in, \
            gzip.open(file_path + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    logger.info("See %s.gz" % file_path)


def prepare_bag_of_apps_datasets(data_dir):
    # Based on : https://www.kaggle.com/xiaoml/talkingdata-mobile-user-demographics/low-ram-bag-of-apps-python/

    # First, check if the datasets have already been created
    boa_file_path_1 = os.path.join(data_dir, "bag_of_apps_train.h5")
    boa_file_path_2 = os.path.join(data_dir, "bag_of_apps_test.h5")
    if os.path.exists(boa_file_path_1) and os.path.exists(boa_file_path_2):
        logger.info("Reading Bag-of-Apps datasets from {} & {}"
                    .format(boa_file_path_1, boa_file_path_2))
        a = pd.read_hdf(boa_file_path_1, "a")
        b = pd.read_hdf(boa_file_path_2, "b")
        return a, b

    # Create the datasets
    logger.info("Preparing Bag-of-Apps datasets")
    app_labels = read_gz(data_dir, "app_labels.csv.gz")
    app_labels = app_labels.groupby("app_id")["label_id"]\
        .apply(lambda x: " ".join(str(s) for s in x))

    app_events = read_gz(data_dir, "app_events.csv.gz")
    app_events["app_labels"] = app_events["app_id"].map(app_labels)
    app_events = app_events.groupby("event_id")["app_labels"]\
        .apply(lambda x: " ".join(str(s) for s in x))
    del app_labels

    events = pd.read_csv(os.path.join(data_dir, "events.csv.gz"),
                         dtype={"device_id": np.str})
    events["app_labels"] = events["event_id"].map(app_events)
    events = events.groupby("device_id")["app_labels"]\
        .apply(lambda x: " ".join(str(s) for s in x))
    del app_events

    pbd = pd.read_csv(os.path.join(data_dir, "phone_brand_device_model.csv.gz"),
                      dtype={"device_id": np.str})
    pbd.drop_duplicates("device_id", keep="first", inplace=True)

    _train = read_gz(data_dir, "gender_age_train.csv.gz")
    _train["app_labels"] = _train["device_id"].map(events)
    _train = pd.merge(_train, pbd, how="left", on="device_id", left_index=True)
    _test = read_gz(data_dir, "gender_age_test.csv.gz")
    _test["app_labels"] = _test["device_id"].map(events)
    _test = pd.merge(_test, pbd, how="left", on="device_id", left_index=True)
    del pbd
    del events

    df_all = pd.concat((_train, _test), axis=0, ignore_index=True)
    split_len = len(_train)
    vec = CountVectorizer(min_df=1, binary=1)
    df_all = df_all[["phone_brand", "device_model", "app_labels"]]\
        .astype(np.str).apply(lambda x: " ".join(s for s in x), axis=1)\
        .fillna("Missing")
    df_tfv = vec.fit_transform(df_all) # 186716 x 2045 sparse matrix
    _train = df_tfv[:split_len, :] # 74645 x 2045 sparse matrix
    _test = df_tfv[split_len:, :] # 112071 x 2045 sparse matrix

    # Converting the sparse matrix into a DataFrame
    a = pd.SparseDataFrame([ pd.SparseSeries(_train[i].toarray().ravel())
                             for i in np.arange(_train.shape[0]) ])
    b = pd.SparseDataFrame([ pd.SparseSeries(_test[i].toarray().ravel())
                             for i in np.arange(_test.shape[0]) ])
    # Rename the columns
    app_labels_cols = ["a" + str(x) for x in np.arange(0, a.shape[1]).tolist()]
    d = dict(zip(np.arange(0, a.shape[1]).tolist(), app_labels_cols))
    a.rename(columns=d, inplace=True)
    b.rename(columns=d, inplace=True)
    # Write to file
    a.to_sparse(kind='block')\
        .to_hdf(boa_file_path_1, "a", mode="w", complib="blosc", complevel=9)
    b.to_sparse(kind='block')\
        .to_hdf(boa_file_path_2, "b", mode="w", complib="blosc", complevel=9)
    del _train
    del _test

    # TO USE, DO
    # train = pd.merge(train, a, left_index=True , right_index=True)

    return a, b # bag-of-apps datasets


def prepare_device_related_datasets(data_dir):
    logger.info("Preparing device related datasets")
    deviceinfo = read_gz(data_dir, "phone_brand_device_model.csv.gz")
    # Get rid of duplicate device ids
    deviceinfo = deviceinfo.drop_duplicates("device_id", keep="first")

    # Extract the phone brand names - translate Chinese to English
    file_path = os.path.join(data_dir, "phone_brands_map.txt")
    if os.path.exists(file_path) == False:
        phone_brands = pd.unique(deviceinfo.phone_brand.ravel()).tolist()
        phone_brands_map = dict(zip(phone_brands, [None] * len(phone_brands)))
        cols = ["phone_brand", "phone_brand_translated"]
        phone_brands = pd.DataFrame(phone_brands_map.items(), columns=cols)
        phone_brands["is_foreign_brand"] = False  # Needs to be hand coded
        phone_brands.to_csv(file_path, encoding="utf-8-sig", index=False,
                            sep="\t")
    else:
        phone_brands = pd.read_csv(file_path, encoding="utf-8-sig",
                                   index_col=False, sep="\t")

    # Convert the index into a column and rename it brand ID
    phone_brands.reset_index(level=0, inplace=True)
    phone_brands.rename(columns={"index": "phone_brand_id"}, inplace=True)

    # Some device_model (such as S6, T5, T9, X5, X6, etc.)
    # associated with more than one phone_brand.
    # So concatenate phone_brand and device_model and then encode it
    m_d = deviceinfo.phone_brand.str.cat(deviceinfo.device_model)
    le = preprocessing.LabelEncoder().fit(m_d)
    deviceinfo["device_model_id"] = le.transform(m_d)

    # Merge device info with phone brands
    deviceinfo = pd.merge(deviceinfo, phone_brands)
    return deviceinfo


def prepare_events_per_hour_per_device_dataset(data_dir):
    logger.info("Preparing events per hour per device dataset")
    events = read_gz(data_dir, "events.csv.gz")
    events.timestamp = pd.to_datetime(events.timestamp)
    events["time_hour"] = events.timestamp.apply(lambda x: x.hour)

    # Count number of events per hour for each device (ephpd)
    ephpd = pd.crosstab(events["device_id"], events["time_hour"])

    # Rename columns showing number of events per hour
    hour_of_day_cols = ["h" + str(x) for x in np.arange(0, 24).tolist()]
    d = dict(zip(np.arange(0, 24).tolist(), hour_of_day_cols))
    ephpd.rename(columns=d, inplace=True)
    ephpd.reset_index(level=0, inplace=True)
    # Normalize the rows in ephpd by their sums
    ephpd_normalized = ephpd[hour_of_day_cols] \
        .div(ephpd[hour_of_day_cols].sum(axis=1), axis=0)
    ephpd_normalized.head()
    ephpd = pd.merge(ephpd, ephpd_normalized,
                     right_index=True, left_index=True, suffixes=('', '_n'))
    return ephpd


def prepare_events_spread_dataset(data_dir):
    # Calculates spread of events over 6 splits:
    # Morning (5AM-10AM), Mid-day (10AM-2PM), Afternoon (2PM-5PM),
    # Evening (5PM-10PM), Night (10PM-1AM), NightOwl (1AM-5AM)
    logger.info("Preparing the events spread dataset")
    events = read_gz(data_dir, "events.csv.gz",
                     cols_to_read=["event_id", "device_id", "timestamp"])
    events.timestamp = pd.to_datetime(events.timestamp)
    events["time_hour"] = events.timestamp.apply(lambda x: x.hour)
    s = events.groupby(["device_id","time_hour"]).size()
    df = s.unstack(level=-1)

    # Create hour-of-day columns showing number of events per hour
    hour_of_day_cols = ["h" + str(x) for x in np.arange(0, 24).tolist()]
    d = dict(zip(np.arange(0, 24).tolist(), hour_of_day_cols))
    df.rename(columns=d, inplace=True)
    df = df.fillna(0)

    # Split into 6 splits
    #df["usage_morning"] = df_tmp[["h" + str(x) for x in np.arange(5, 10).tolist()]].sum(axis=1)
    df["h_morning"] = df[["h" + str(x) for x in [5, 6, 7, 8, 9]]].sum(axis=1)
    df["h_midday"] = df[["h" + str(x) for x in [10, 11, 12, 13]]].sum(axis=1)
    df["h_afternoon"] = df[["h" + str(x) for x in [14, 15, 16]]].sum(axis=1)
    df["h_evening"] = df[["h" + str(x) for x in [17, 18, 19, 20, 21]]].sum(axis=1)
    df["h_night"] = df[["h" + str(x) for x in [22, 23, 0]]].sum(axis=1)
    df["h_nightowl"] = df[["h" + str(x) for x in [1, 2, 3, 4]]].sum(axis=1)

    # Drop the hour-of-day columns
    df.drop(hour_of_day_cols, axis=1, inplace=True)

    # Normalize rows
    df = df.div(df.sum(axis=1), axis=0).round(2)

    df.reset_index(level=0, inplace=True)

    return df


def prepare_sparse_features_dataset(data_dir):
    # Credit : Data preparation strategy influenced by the script
    # written by dune_dweller,
    # https://www.kaggle.com/dvasyukova/talkingdata-mobile-user-demographics/a-linear-model-on-apps-and-labels

    # First, read the datasets
    gatrain = pd.read_csv(os.path.join(data_dir, "gender_age_train.csv.gz"),
                          index_col = "device_id")
    gatest = pd.read_csv(os.path.join(data_dir, "gender_age_test.csv.gz"),
                         index_col = "device_id")
    phone = pd.read_csv(os.path.join(data_dir, "phone_brand_device_model.csv"))
    # Get rid of duplicate device ids in phone
    phone = phone.drop_duplicates("device_id", keep="first")\
        .set_index("device_id")
    events = pd.read_csv(os.path.join(data_dir, "events.csv.gz"),
                         parse_dates=["timestamp"], index_col="event_id")
    appevents = pd.read_csv(os.path.join(data_dir, "app_events.csv.gz"),
                            usecols=["event_id", "app_id", "is_active"],
                            dtype={"is_active":bool})
    applabels = pd.read_csv(os.path.join(data_dir, "app_labels.csv.gz"))

    gatrain["trainrow"] = np.arange(gatrain.shape[0])
    gatest["testrow"] = np.arange(gatest.shape[0])

    # Next, create the sparse features
    #Phone brand
    brandencoder = LabelEncoder().fit(phone.phone_brand)
    phone["brand"] = brandencoder.transform(phone["phone_brand"])
    gatrain["brand"] = phone["brand"]
    gatest["brand"] = phone["brand"]
    Xtr_brand = csr_matrix((np.ones(gatrain.shape[0]),
                           (gatrain.trainrow, gatrain.brand)))
    Xte_brand = csr_matrix((np.ones(gatest.shape[0]),
                           (gatest.testrow, gatest.brand)))
    logger.info("Brand features: train shape {}, test shape {}"
                .format(Xtr_brand.shape, Xte_brand.shape))

    # Device model
    m = phone.phone_brand.str.cat(phone.device_model)
    modelencoder = LabelEncoder().fit(m)
    phone["model"] = modelencoder.transform(m)
    gatrain["model"] = phone["model"]
    gatest["model"] = phone["model"]
    Xtr_model = csr_matrix((np.ones(gatrain.shape[0]),
                           (gatrain.trainrow, gatrain.model)))
    Xte_model = csr_matrix((np.ones(gatest.shape[0]),
                           (gatest.testrow, gatest.model)))
    logger.info("Model features: train shape {}, test shape {}"
                .format(Xtr_model.shape, Xte_model.shape))

    # Installed apps features
    appencoder = LabelEncoder().fit(appevents.app_id)
    appevents["app"] = appencoder.transform(appevents.app_id)
    napps = len(appencoder.classes_)
    logger.info(napps, "apps")
    deviceapps = (appevents.merge(events[["device_id"]], how="left",
                                  left_on="event_id",right_index=True)
                           .groupby(["device_id","app"])["app"].agg(["size"])
                           .merge(gatrain[["trainrow"]], how="left",
                                  left_index=True, right_index=True)
                           .merge(gatest[["testrow"]], how="left",
                                  left_index=True, right_index=True)
                           .reset_index())

    d = deviceapps.dropna(subset=['trainrow'])
    Xtr_app = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.app)),
                          shape=(gatrain.shape[0],napps))
    d = deviceapps.dropna(subset=['testrow'])
    Xte_app = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.app)),
                          shape=(gatest.shape[0],napps))
    logger.info("Apps data: train shape {}, test shape {}"
                .format(Xtr_app.shape, Xte_app.shape))

    # App labels features
    applabels = applabels.loc[applabels.app_id.isin(appevents.app_id.unique())]
    applabels["app"] = appencoder.transform(applabels.app_id)
    labelencoder = LabelEncoder().fit(applabels.label_id)
    applabels["label"] = labelencoder.transform(applabels.label_id)
    nlabels = len(labelencoder.classes_)

    devicelabels = (deviceapps[["device_id","app"]]
                    .merge(applabels[["app","label"]])
                    .groupby(["device_id","label"])["app"].agg(["size"])
                    .merge(gatrain[["trainrow"]], how='left', left_index=True,
                           right_index=True)
                    .merge(gatest[["testrow"]], how='left', left_index=True,
                           right_index=True)
                    .reset_index())

    d = devicelabels.dropna(subset=["trainrow"])
    Xtr_label = csr_matrix((np.ones(d.shape[0]), (d.trainrow, d.label)),
                          shape=(gatrain.shape[0],nlabels))
    d = devicelabels.dropna(subset=["testrow"])
    Xte_label = csr_matrix((np.ones(d.shape[0]), (d.testrow, d.label)),
                          shape=(gatest.shape[0],nlabels))
    logger.info("Labels data: train shape {}, test shape {}"
                .format(Xtr_label.shape, Xte_label.shape))

    # Concatenate all features
    Xtrain = hstack((Xtr_brand, Xtr_model, Xtr_app, Xtr_label), format="csr")
    Xtest =  hstack((Xte_brand, Xte_model, Xte_app, Xte_label), format="csr")
    # Xtrain and Xtest are of type, scipy.sparse.csr.csr_matrix
    logger.info("All features: train shape {}, test shape {}"
                .format(Xtrain.shape, Xtest.shape))

    return Xtrain, Xtest


def read_estimator_params(section, estimator_suffix):
    return cfg[section]["params_" + estimator_suffix]


def read_gz(data_dir, file_name, cols_to_read=None):
    file_path = os.path.join(data_dir, file_name)
    if cols_to_read:
        return pd.read_csv(file_path, index_col=False, encoding="utf-8-sig",
                           usecols=cols_to_read)
    else:
        return pd.read_csv(file_path, index_col=False, encoding="utf-8-sig")


def report_grid_search_scores(grid_scores, n_top=5):
    # Utility function to report best scores
    # Credit : http://scikit-learn.org/stable/auto_examples/model_selection/randomized_search.html
    top_scores = sorted(grid_scores, key=operator.itemgetter(1),
                        reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        rank, mvs = (i + 1), score.mean_validation_score
        logger.info("Model rank {0}, mean validation score {1:.3f}, "\
                    "parameters : {2}".format(rank, mvs, score.parameters))
