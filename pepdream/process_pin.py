# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import warnings
import pandas as pd
import numpy as np
from sklearn import preprocessing
from .qvalue_utils import qvalue


def load_feature_matrix(filename):
    """ Load all PSMs and features from a percolator input (PIN) file

        For n input features and m total file fields, the file format is:
        header field 1: SpecId, or other PSM id
        header field 2: Label, denoting whether the PSM is a target or decoy
        header field 3: ScanNr, the scan number. Note this string must be exactly stated
        header field 4 (optional): ExpMass, PSM experimental mass. Not used as a feature
        header field 4 + 1 : Input feature 1
        header field 4 + 2 : Input feature 2
        ...
        header field 4 + n : Input feature n
        header field 4 + n + 1 : Peptide, the peptide string
        header field 4 + n + 2 : Protein id 1
        header field 4 + n + 3 : Protein id 2
        ...
        header field m : Protein id m - n - 4
    """
    df = pd.read_csv(filename, sep='\t')
    headers: list[str] = df.keys()

    # Check ScanNr and ExpMass
    if "ScanNr" not in headers:
        raise ValueError("No ScanNr field, exiting")

    if "ExpMass" not in headers:
        raise ValueError("No ExpMass field, exiting")

    # Check SpecId
    const_keys = []
    id_key = headers[0]
    if id_key.lower() == "specid":
        const_keys.append(id_key)

    # Check label
    label_key = headers[1]
    if label_key.lower() == "label":
        const_keys.append(label_key)

    # Exclude calcmass and expmass as features
    const_keys += ["ScanNr", "CalcMass", "ExpMass"]

    # Find peptide and protein ID fields
    pep_inform_key = [headers[0]]
    is_const = False
    for key in headers:
        if key.lower() == "peptide":
            is_const = True
        if is_const:
            assert "peptide" in key.lower() or "protein" in key.lower(), \
                f"Unexpected field {key} after peptide field"
            const_keys.append(key)
            pep_inform_key.append(key)

    # List of (SpecId, peptide, protein)
    pep_inform = [df[key] for key in pep_inform_key]

    # Features
    const_keys = set(const_keys)
    feature_names = []
    for h in headers:
        if h not in const_keys:
            feature_names.append(h)
    features = df[feature_names].to_numpy(dtype=np.float32)

    # Labels
    labels = df[label_key].to_numpy(dtype=np.int32) > 0

    pos = np.count_nonzero(labels)
    neg = np.count_nonzero(~labels)
    print(f"Loaded {pos} target and {neg} decoy PSMS with {features.shape[1]} features.")

    # Scale usually raises warnings, ignore them.
    warnings.filterwarnings("ignore", category=UserWarning)
    features = preprocessing.scale(features)
    warnings.resetwarnings()

    # Extract scan number
    scan_number = df["ScanNr"].to_numpy(dtype=np.int32)

    return pep_inform, feature_names, features, labels, scan_number


def search_init_dir(X: np.ndarray, Y: np.ndarray, q=0.01):
    """Search for initial feature which seperates most identifications.

    Args:
        X (np.ndarray): feature matrix
        Y (np.ndarray): labels
        q (float): q used for confidenct PSMs selection. defaults to 0.01

    Returns:
        max_feat (int): index of best feature
    """
    max_num = 0
    max_feat = -1
    max_scores = None

    def num_identified(scores: np.ndarray, labels: np.ndarray):
        return sum((qvalue(scores, Y) <= q) & labels)

    for i in range(X.shape[1]):
        scores = X[:, i]

        # Check scores at both directions.
        count = num_identified(scores, Y)
        if count > max_num:
            max_num = count
            max_feat = i
            max_scores = scores

        count = num_identified(-scores, Y)
        if count > max_num:
            max_num = count
            max_feat = i
            max_scores = -scores

    print(f"Found initial feature No.{max_feat} with {max_num} identifications.")
    return max_feat, max_scores


def normalize_spectrum(spectrum: np.ndarray):
    """Normalize spectrum to unit length

    Args:
        spectrum (np.ndarray): spectrum to normalize

    Returns:
        np.ndarray: normalized spectrum
    """
    spectrum = spectrum * (spectrum > 0)
    return preprocessing.normalize(spectrum)
