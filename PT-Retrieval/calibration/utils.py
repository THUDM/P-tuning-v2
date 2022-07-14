
import sys
from pathlib import Path
BASE_DIR = Path.resolve(Path(__file__)).parent.parent
sys.path.append(str(BASE_DIR))
print(BASE_DIR)
    
import logging
import numpy as np
from scipy.special import softmax

from beir import LoggingHandler

from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics._base import _check_pos_label_consistency
from sklearn.calibration import CalibrationDisplay, calibration_curve

import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

def calibration_curve_with_ece(
    y_true,
    y_prob,
    *,
    pos_label=None,
    n_bins=5,
    strategy="uniform",
):
    y_true = column_or_1d(y_true)
    y_prob = column_or_1d(y_prob)
    check_consistent_length(y_true, y_prob)
    pos_label = _check_pos_label_consistency(pos_label, y_true)

    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1].")

    labels = np.unique(y_true)
    if len(labels) > 2:
        raise ValueError(
            f"Only binary classification is supported. Provided labels {labels}."
        )
    y_true = y_true == pos_label

    if strategy == "quantile":  # Determine bin edges by distribution of data
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.percentile(y_prob, quantiles * 100)
    elif strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        raise ValueError(
            "Invalid entry to 'strategy' input. Strategy "
            "must be either 'quantile' or 'uniform'."
        )

    binids = np.searchsorted(bins[1:-1], y_prob)

    bin_sums = np.bincount(binids, weights=y_prob, minlength=len(bins))
    bin_true = np.bincount(binids, weights=y_true, minlength=len(bins))
    bin_total = np.bincount(binids, minlength=len(bins))

    nonzero = bin_total != 0
    prob_true = bin_true[nonzero] / bin_total[nonzero]
    prob_pred = bin_sums[nonzero] / bin_total[nonzero]
    ece = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

    return prob_true, prob_pred, ece

def calculate_ece(qrels, results, args):
    questions = qrels.keys()
    y_true = []
    y_prob = []
    for qid in questions:
        rels = [ k for k, v in qrels[qid].items() if v ]
        sorted_result = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        prob = []
        for pid, score in sorted_result[:args.n_ece_docs]:
            prob.append(score)
            y_true.append(1 if pid in rels else 0)
        y_prob.extend(softmax(prob))

    _, _, ece = calibration_curve_with_ece(y_true, y_prob, n_bins=10)
    return ece

def calculate_erce(qrels, results, args):
    questions = qrels.keys()
    y_true = []
    y_prob = []
    for qid in questions:
        rels = set([k for k, v in qrels[qid].items() if v])
        sorted_result = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)
        pos_scores = []
        neg_scores = []
        # VERSION 1
        for pid, score in sorted_result[:args.n_erce_docs]:
            if pid in rels:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
        if len(pos_scores) == 0:
            for pid, score in sorted_result:
                if pid in rels:
                    pos_scores.append(score)
                    break
        # VERSION 2
        # for pid, score in sorted_result:
        #     if pid in rels:
        #         pos_scores.append(score)
        #         break
        # if len(pos_scores) == 0: pos_scores.append(0.0)
        # for pid, score in sorted_result[:args.n_erce_docs]:
        #     if pid not in rels:
        #         neg_scores.append(score)
        
        for pscore in pos_scores:
            for nscore in neg_scores:
                pos_score, neg_score = softmax([pscore, nscore])
                diff = pos_score - neg_score
                y_prob.append(abs(diff))
                y_true.append(int(diff > 0))

    _, _, erce = calibration_curve_with_ece(y_true, y_prob, n_bins=10)
    return erce

def plot_calibration_curve(qrels, results, args):
    questions = qrels.keys()
    y_true = []
    y_prob = []

    for qid in questions:
        rels = set([k for k, v in qrels[qid].items() if v])
        result = set(results[qid].keys())
        pos_docs = list(result.intersection(rels))
        if len(pos_docs) == 0:
            continue
        neg_docs = list(result - rels)

        pos_pid = np.random.choice(pos_docs, 1)[0]
        neg_pid = np.random.choice(neg_docs, args.n_non_rel, replace=False)
        pos_score = results[qid][pos_pid]
        neg_scores = [ results[qid][i] for i in neg_pid ]
        scores = softmax([pos_score] + neg_scores)

        y_true.extend([1] + [0] * args.n_non_rel)
        y_prob.extend(scores)


    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)

    disp = CalibrationDisplay(prob_true, prob_pred, y_prob)
    disp.plot()

    method = "pt" if args.prefix else "ft"
    filename = f"{args.dataset}-{args.n_non_rel}-{method}"
    plt.savefig(f'results/{filename}.png')
