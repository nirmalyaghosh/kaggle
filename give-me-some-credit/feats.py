# -*- coding: utf-8 -*-
"""
Common code for adding features.

@author: Nirmalya Ghosh
"""

import logging

import pandas as pd

import utils


def add_features_based_on_NumOCLL(_tr_df, _te_df):
    # Generates the features based on NumberOfOpenCreditLinesAndLoans
    num_buckets = 8
    _tr_df = utils.split_into_buckets(_tr_df, "NumOCLL", num_buckets,
                                      drop_original_col=True,
                                      drop_bucket_col=True)
    _te_df = utils.split_into_buckets(_te_df, "NumOCLL", num_buckets,
                                      drop_original_col=True,
                                      drop_bucket_col=True)
    return _tr_df, _te_df


def add_features_based_on_NumRELL(_tr_df, _te_df):
    # Generates the features based on NumberRealEstateLoansOrLines
    num_buckets = 4
    _tr_df = utils.split_into_buckets(_tr_df, "NumRELL", num_buckets,
                                      drop_original_col=True,
                                      drop_bucket_col=True,
                                      add_jitter=True)
    _te_df = utils.split_into_buckets(_te_df, "NumRELL", num_buckets,
                                      drop_original_col=True,
                                      drop_bucket_col=True,
                                      add_jitter=True)
    return _tr_df, _te_df


def add_features_based_on_RUoUL(_tr_df, _te_df):
    # Generates the 13 features based on RevolvingUtilizationOfUnsecuredLines
    logging.info("Adding 13 new features based on RUoUL ...")

    def get_ncc(row):
        if row["RUoUL"] <= 1:
            return 0
        if row["RUoUL"] > 1 and row["RUoUL"] <= 2:
            return 1
        if row["RUoUL"] > 2 and row["RUoUL"] <= 3:
            return 2
        if row["RUoUL"] > 3:
            return 3

    _tr_df["ncc"] = _tr_df.apply(lambda row: get_ncc(row), axis=1)
    _te_df["ncc"] = _te_df.apply(lambda row: get_ncc(row), axis=1)

    # Remove duplicates if present from previous run
    tr_df = _tr_df.loc[:, ~_tr_df.columns.duplicated()]
    te_df = _te_df.loc[:, ~_te_df.columns.duplicated()]

    # Construct dummy variables based on the number of credit cards cancelled
    tr_df = pd.concat([_tr_df, pd.get_dummies(_tr_df["ncc"], prefix="ncc")],
                      axis=1)
    te_df = pd.concat([_te_df, pd.get_dummies(_te_df["ncc"], prefix="ncc")],
                      axis=1)

    # Split 'RUoUL' (for "normal" range, 0-1) into 9 nearly equal-sized buckets
    num_buckets = 9
    tr_data = tr_df[(tr_df.RUoUL >= 0) & (tr_df.RUoUL <= 1)]
    tr_data = utils.split_into_buckets(tr_data, "RUoUL", num_buckets,
                                       convert_into_dummies=True,
                                       drop_original_col=False,
                                       drop_bucket_col=False)
    te_data = te_df[(te_df.RUoUL >= 0) & (te_df.RUoUL <= 1)]
    te_data = utils.split_into_buckets(te_data, "RUoUL", num_buckets,
                                       convert_into_dummies=True,
                                       drop_original_col=False,
                                       drop_bucket_col=False)

    # Set for train
    cols = ["ruoul_bucket_" + str(i) for i in range(0, 9)]
    for col in cols:
        tr_df[col] = tr_data[col]
        tr_df[col].fillna(0, inplace=True)
    # Set for test
    for col in cols:
        te_df[col] = te_data[col]
        te_df[col].fillna(0, inplace=True)

    # Rename the columns
    colmap2 = {"ncc_0": "has_no_cards_cancelled",
               "ncc_1": "has_1_card_cancelled",
               "ncc_2": "has_2_cards_cancelled",
               "ncc_3": "has_3ormore_cards_cancelled"
               }
    tr_df.rename(columns=colmap2, inplace=True)
    te_df.rename(columns=colmap2, inplace=True)

    tr_df.drop(["ncc", "RUoUL"], axis=1, inplace=True)
    te_df.drop(["ncc", "RUoUL"], axis=1, inplace=True)

    return tr_df, te_df
