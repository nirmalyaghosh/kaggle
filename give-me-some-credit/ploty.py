# -*- coding: utf-8 -*-
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import auc, roc_curve

_almost_black = "#262626"


def plot_resampled_plots(counts, X_vis, y, X_res_vis, X_resampled, y_resampled,
                         resampling_technique_title):
    # Adapted from
    # http://contrib.scikit-learn.org/imbalanced-learn/auto_examples/

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    labels_before = ["Class #0 (OK, {} points)".format(counts[0]),
                     "Class #1 (Distress, {} points)".format(counts[1])
                     ]
    ax1.scatter(X_vis[y == 0, 0], X_vis[y == 0, 1], label=labels_before[0],
                alpha=0.5,
                edgecolor=_almost_black, facecolor="blue", linewidth=0.15)
    ax1.scatter(X_vis[y == 1, 0], X_vis[y == 1, 1], label=labels_before[1],
                alpha=0.5,
                edgecolor=_almost_black, facecolor="magenta", linewidth=0.15)
    ax1.set_title("Original dataset")
    ax1.set_ylim([-1.0, 1.2])
    ax1.legend(loc="lower center", ncol=1)

    resampled_counts = [np.count_nonzero(y_resampled == 0),
                        np.count_nonzero(y_resampled == 1)
                        ]
    labels_after = ["Class #0 (OK, {} points)".format(resampled_counts[0]),
                    "Class #1 (Distress, {} points)".format(
                        resampled_counts[1])
                    ]
    ax2.scatter(X_res_vis[y_resampled == 0, 0], X_res_vis[y_resampled == 0, 1],
                label=labels_after[0], alpha=.5, edgecolor=_almost_black,
                facecolor="blue", linewidth=0.15)
    ax2.scatter(X_res_vis[y_resampled == 1, 0], X_res_vis[y_resampled == 1, 1],
                label=labels_after[1], alpha=.5, edgecolor=_almost_black,
                facecolor="magenta", linewidth=0.15)
    ax2.set_title(resampling_technique_title)
    ax2.set_ylim([-1.0, 1.2])
    ax2.legend(loc="lower center", ncol=1)

    plt.tight_layout()
    plt.show()


def plot_ROC_curve(est, X, y, pos_label=1, n_folds=5, title_suffix=""):
    # Adapted from : https://git.io/vXu0Z
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    logging.info("Training {} {}".format(est.__class__.__name__, title_suffix))
    for i, (train, test) in enumerate(StratifiedKFold(y, n_folds=n_folds)):
        probas_ = est.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area under the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1], pos_label=1)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1,
                 label="ROC fold {} (area = {:.2f})".format(i, roc_auc))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label="Random")
    mean_tpr /= n_folds
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    mean_auc_str = "Mean ROC (area = {:.2f})".format(mean_auc)
    logging.info(mean_auc_str)
    plt.plot(mean_fpr, mean_tpr, "k--", label=mean_auc_str, lw=2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    desc = "{} {}".format(est.__class__.__name__, title_suffix)
    plt.title("ROC curves for a {}".format(desc))
    plt.legend(loc="lower right")
    plt.show()
    return (desc, mean_auc)


def plot_top_n_important_features(features, n=10, figsize=(15,8)):
    # Plot the top N features
    features.sort_values(by=["importance"], ascending=True).tail(n)\
    .plot(x="feature", y="importance", kind="barh", figsize=figsize,
          title="Top {} Most Important Features".format(n))
