# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import numpy as np
from sklearn import preprocessing
from . import constants 


def peptide_parser(p):
    if p[0] == "(":
        raise ValueError("sequence starts with '('")
    n = len(p)
    i = 0
    while i < n:
        if i < n - 3 and p[i + 1] == "(":
            j = p[i + 2:].index(")")
            offset = i + j + 3
            yield p[i:offset]
            i = offset
        else:
            yield p[i]
            i += 1


def peptide_encoder(seq: str, length=constants.MAX_SEQUENCE):
    seq = seq.replace("_", "")
    seq = seq.replace(".", "")
    rst = np.zeros(length, dtype='int')
    for i, s in enumerate(peptide_parser(seq)):
        rst[i] = constants.ALPHABET[s]
    return rst


def intensity_normalizer(observe: np.ndarray, predict: np.ndarray):
    """ Normalize ion intensities for observe and predict.
    """

    # WATCH OUT!!! we may need pred = pred * (real >= 0)
    observe = observe * (observe >= 0)
    predict = predict * (predict >= 0)

    observe = preprocessing.normalize(observe)
    predict = preprocessing.normalize(predict)

    return observe, predict
