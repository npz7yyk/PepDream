# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import numpy as np

_accelerated = False
try:
    from pepdream.qvalue_acc import calculate_qvalue_acc
    _accelerated = True
    print('Using accelerated q-value calculation.')
except ImportError:
    pass


def accumulate(iterable, func):
    """An iterator that accumulate the results of a function.

    >>> list(accumulate([1,2,3,4,5], operator.add))
    [1, 3, 6, 10, 15]

    >>> list(accumulate([1,2,3,4,5], operator.mul))
    [1, 2, 6, 24, 120]

    Args:
        iterable (iterable): iterable object
        func (function): function to accumulate

    Yields:
        iterator: accumulated results

    @remark: This function is copied from proteoTorch and modified.
            (https://github.com/proteoTorch/proteoTorch)
            Licensed under the Open Software License version 3.0
    """
    itr = iter(iterable)
    total = next(itr)
    yield total
    for element in itr:
        total = func(total, element)
        yield total


def calculate_qvalue(scores, labels, pi=1.0):
    """Calculates q-values from sorted lists of scores and labels.

    Args:
        scores (list[float]): scores of PSMs
        labels (list[int]): labels of PSMs
        pi (float): pi used for q-values, defaults to 1.0

    Returns:
        list[float]: q-values of PSMs

    @remark: This function is copied from proteoTorch and modified.
            (https://github.com/proteoTorch/proteoTorch)
            Licensed under the Open Software License version 3.0
    """

    # try to use accelerated version
    if _accelerated:
        return calculate_qvalue_acc(scores, labels, pi)

    qvals = []

    h_w_le_z = []
    h_z_le_z = []

    if pi < 1.0:
        cnt_z = 0
        cnt_w = 0
        queue = 0
        for idx in range(len(scores) - 1, -1, -1):
            if (labels[idx] == 1):
                cnt_w += 1
            else:
                cnt_z += 1
                queue += 1
            if idx == 0 or scores[idx] != scores[idx - 1]:
                for _ in range(queue):
                    h_w_le_z.append(float(cnt_w))
                    h_z_le_z.append(float(cnt_z))
                queue = 0

    estPx_lt_zj = 0.0
    E_f1_mod_run_tot = 0.0
    fdr = 0.0
    n_z_ge_w = 1
    n_w_ge_w = 0

    decoyQueue = 0
    targetQueue = 0
    for idx in range(len(scores)):
        if labels[idx] == 1:
            n_w_ge_w += 1
            targetQueue += 1
        else:
            n_z_ge_w += 1
            decoyQueue += 1

        if idx == len(scores) - 1 or scores[idx] != scores[idx + 1]:
            if pi < 1.0 and decoyQueue > 0:
                j = len(h_w_le_z) - (n_z_ge_w - 1)
                cnt_w = float(h_w_le_z[j])
                cnt_z = float(h_z_le_z[j])
                estPx_lt_zj = (cnt_w - pi * cnt_z) / ((1.0 - pi) * cnt_z)

                if estPx_lt_zj > 1.:
                    estPx_lt_zj = 1.
                if estPx_lt_zj < 0.:
                    estPx_lt_zj = 0.
                E_f1_mod_run_tot += float(decoyQueue) * estPx_lt_zj * (1.0 - pi)

            targetQueue += decoyQueue

            fdr = (n_z_ge_w * pi + E_f1_mod_run_tot) / float(max(1, n_w_ge_w))
            for _ in range(targetQueue):
                qvals.append(min(fdr, 1.))
            decoyQueue = 0
            targetQueue = 0

    # Convert the FDRs into q-values.
    return list(accumulate(qvals[::-1], min))[::-1]


def qvalue(scores, labels, pi=1.0):
    """Calculates q-values from a list of scores and labels.

    Args:
        scores (list[float]): scores of PSMs
        labels (list[int]): labels of PSMs
        pi (float): pi used for q-values, defaults to 1.0

    Returns:
        list[float]: q-values of PSMs
    """

    # index after sorting by score
    order = np.argsort(-scores)
    scores = np.array(scores)[order]
    labels = np.array(labels)[order]

    qvals = calculate_qvalue(scores, labels, pi)

    # recover original order
    remap = np.argsort(order)
    qvals = np.array(qvals)[remap]

    return qvals


def target_identified(scores, labels, q):
    """Returns the identified targets in boolean.

    Args:
        scores (np.ndarray): PSM scores
        Y (np.ndarray): PSM labels
        q (float): q-value threshold

    Returns:
        np.ndarray: identified targets in boolean
    """
    return (qvalue(scores, labels) <= q) & (labels > 0)


def target_accumulate(scores, labels, thres=0.01):
    """Returns the accumulated number of identified targets.
    """
    # index after sorting by score
    order = np.argsort(-scores)
    scores = np.array(scores)[order]
    labels = np.array(labels)[order]

    qvalues = calculate_qvalue(scores, labels)

    labels = labels[np.array(qvalues) <= thres].astype(int)
    return np.cumsum(labels)


def qvalue_auc(thres=0.1):
    """Returns a function that calculates AUC at a given q-value threshold.
    """
    def func(predictions, labels):
        pos_accumulate = target_accumulate(predictions, labels, thres)
        # normalize by the number of negatives
        auc = np.trapz(pos_accumulate) / (len(labels) - np.sum(labels))
        return auc
    return func
