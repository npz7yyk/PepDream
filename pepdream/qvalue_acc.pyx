# cython: language_level=3
# distutils: language=c++

# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free


cdef calculate_qvalue(
    double* scores,
    unsigned int* labels,
    unsigned int size,
    double pi=1.0
):
    """
    @remark: This function is copied from proteoTorch and modified.
            (https://github.com/proteoTorch/proteoTorch)
            Licensed under the Open Software License version 3.0
    """
    cdef vector[double] qvals
    qvals.reserve(size)
    cdef unsigned int idx

    cdef vector[int] h_w_le_z
    cdef vector[int] h_z_le_z
    cdef unsigned int countTotal = 0
    cdef unsigned int n_z_ge_w = 0
    cdef unsigned int n_w_ge_w = 0
    cdef unsigned int queue = 0

    if pi < 1.0:
        for idx in range(size - 1, -1, -1):
            if (labels[idx] == 1):
                n_w_ge_w += 1
            else:
                n_z_ge_w += 1
                queue += 1
            if idx == 0 or scores[idx] != scores[idx - 1]:
                for _ in range(queue):
                    h_w_le_z.push_back(n_w_ge_w)
                    h_z_le_z.push_back(n_z_ge_w)
                    countTotal += 1
                queue = 0

    cdef double estPx_lt_zj = 0.
    cdef double E_f1_mod_run_tot = 0.0
    cdef double fdr = 0.0
    cdef double cnt_z = 0
    cdef double cnt_w = 0
    cdef int j = 0
    n_z_ge_w = 1
    n_w_ge_w = 0

    cdef unsigned int decoyQueue = 0
    cdef unsigned int targetQueue = 0
    for idx in range(size):
        if labels[idx] == 1:
            n_w_ge_w += 1
            targetQueue += 1
        else:
            n_z_ge_w += 1
            decoyQueue += 1

        if idx == size - 1 or scores[idx] != scores[idx + 1]:
            if pi < 1.0 and decoyQueue > 0:
                j = countTotal - (n_z_ge_w - 1)
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
                qvals.push_back(min(fdr, 1.))
            decoyQueue = 0
            targetQueue = 0

    cdef double last = qvals[size - 1]
    for idx in range(size - 2, -1, -1):
        if qvals[idx] > last:
            qvals[idx] = last
        else:
            last = qvals[idx]
    return list(qvals)


def calculate_qvalue_acc(
    scores, labels, pi=1.0
):
    cdef unsigned int size = len(scores)
    cdef double* cscores = <double*> malloc(size * sizeof(double))
    cdef unsigned int* clabels = <unsigned int*> malloc(size * sizeof(int))

    for i in range(size):
        cscores[i] = scores[i]
        clabels[i] = labels[i]

    rst = calculate_qvalue(cscores, clabels, size, pi)

    free(cscores)
    free(clabels)

    return rst
