# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import os
import re
import h5py
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from typing import List, Dict

from .constants import *


# some of the following functions are copied from Prosit
# link: https://github.com/kusterlab/prosit
# license: Apache License 2.0
# they are marked with "copied from Prosit"


# copied from Prosit and modified
def sequence_filter(sequence: str):
    def include(tar: str, substrs: List[str]):
        for substr in substrs:
            if substr in tar:
                return True
        return False

    pattern = re.compile(r".+(M\(ox\))*.*")
    if sequence.startswith("_(ac)"):
        return False
    if sequence.startswith("(ac)"):
        return False
    if include(sequence, ["U", "X", "O"]):
        return False
    if not re.match(pattern, sequence):
        return False
    return True


def psm_filter(row: pd.Series):
    if row["Fragmentation"] != "HCD":
        return False
    if row["Charge"] > DEFAULT_MAX_CHARGE:
        return False
    if not MIN_SEQUENCE <= row["Length"] <= MAX_SEQUENCE:
        return False
    if row["Length"] > MAX_SEQUENCE:
        return False
    if not sequence_filter(row["Modified sequence"]):
        return False

    return True


def collect_raw(raw_dir: str):
    """Collect all the csv files in raw_dir."""

    # get all the corresponding csv file
    csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]

    raws = {}
    for csv in tqdm(csv_files):
        csv_dir = os.path.join(raw_dir, csv)
        csv_name = csv.split(".")[0]
        df = pd.read_csv(csv_dir)
        raws[csv_name] = df

    return raws


# copied from Prosit
def peptide_parser(p: str):
    p = p.replace("_", "")
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


# copied from Prosit and modified
def peptide_encoder(p: List[str]):
    rst = np.zeros((MAX_SEQUENCE, ), dtype="int")
    for i, s in enumerate(p):
        rst[i] = ALPHABET[s]
    return rst


# copied from Prosit and modified
def biaccumulated_sum(amino_acids: List[str]):
    """Calculate the accumulated sum of a peptide in both directions."""
    masses = [AMINO_ACID[a] for a in amino_acids]
    forward = np.cumsum(masses)
    backward = np.cumsum(list(reversed(masses)))
    return forward, backward


# copied from Prosit
def get_mz(mass, ion_offset, charge):
    """Calculate m/z."""
    return (mass + ion_offset + charge * PROTON) / charge


# copied from Prosit
def get_mzs(cumsum, ion_type, z):
    """Calculate m/z for a cumsum list."""
    return [get_mz(s, ION_OFFSET[ion_type], z) for s in cumsum[:-1]]


# copied from Prosit
def ion_theoretical_mz(forward, backward, charge):
    """Get dict of ion - theoretical m/z."""
    ion_mz_all = {}
    ion_types = ION_TYPES
    for ion_type in ion_types:
        # try to find the target ion type
        if ion_type in FORWARD:
            cummass = forward
        elif ion_type in BACKWARD:
            cummass = backward
        else:
            # invalid ion type, mark it
            raise ValueError(f"unkown ion_type: {ion_type}")

        mzs = get_mzs(cummass, ion_type, charge)
        ion_mz = {(ion_type, i): mz for i, mz in enumerate(mzs)}
        ion_mz_all.update(ion_mz)
    return ion_mz_all


# copied from Prosit
def get_tolerance(theoretical, mass_analyzer):
    if mass_analyzer in TOLERANCE:
        tolerance, unit = TOLERANCE[mass_analyzer]
        if unit == "ppm":
            return theoretical * float(tolerance) / 10 ** 6
        elif unit == "da":
            return float(tolerance)
        else:
            raise ValueError(f"unit {unit} not implemented")
    else:
        raise ValueError(f"no tolerance implemented for {mass_analyzer}")


# copied from Prosit
def in_tolerance(theoretical, observed, mass_analyzer):
    mz_tolerance = get_tolerance(theoretical, mass_analyzer)
    lower = observed - mz_tolerance
    upper = observed + mz_tolerance
    return theoretical >= lower and theoretical <= upper


# copied from Prosit
def search_tolerant_mz(mzs_observed, mz_theoretical, mass_analyzer):
    """Find tolerant mz observed"""
    head, tail = 0, len(mzs_observed) - 1
    mzs_observed = sorted(mzs_observed)
    while head <= tail:
        mid = (head + tail) // 2
        if in_tolerance(mz_theoretical, mzs_observed[mid], mass_analyzer):
            return mid
        elif mzs_observed[mid] < mz_theoretical:
            head = mid + 1
        elif mz_theoretical < mzs_observed[mid]:
            tail = mid - 1
    return None


def match_single_psm(
    masses_raw: str,
    intensities_raw: str,
    amino_acids: List[str],
    analyzer: str,
    charge: int
):
    """Give matches and relative information of a single psm."""

    def default_floats(floats: str):
        try:
            return [float(m) for m in floats.split(" ")]
        except BaseException:
            return []

    mzs_raw = default_floats(masses_raw)
    intensities_raw = default_floats(intensities_raw)

    forward_sum, backward_sum = biaccumulated_sum(amino_acids)
    max_charge = min(charge, MAX_FRAG_CHARGE)

    uniform_shape = (MAX_ION, len(ION_TYPES), MAX_FRAG_CHARGE)
    intensities = np.zeros(uniform_shape)
    for charge_index in range(max_charge):
        charge = charge_index + 1
        ion_mzs = ion_theoretical_mz(forward_sum, backward_sum, charge)
        for (ion_type, location), mz_theoretical in ion_mzs.items():
            index = search_tolerant_mz(mzs_raw, mz_theoretical, analyzer)
            if index is not None:
                ion_type = ION_TYPE_ALPHABET[ion_type]
                intensities[location, ion_type, charge_index] = intensities_raw[index]

    def mask_outofrange(ndarray: np.ndarray, mask=-1.0):
        """Mask out of range values."""
        length = len(amino_acids)
        ndarray[length - 1:, :, :] = mask
        ndarray[:, :, max_charge:] = mask
        return ndarray

    # mask out of range values
    intensities = mask_outofrange(intensities)

    # flatten the result
    intensities = intensities.reshape(-1)

    return intensities


ANDROMEDA_FEATURES = [
    "SpecId", "Label", "ScanNr", "ExpMass", "Mass",
    "deltaM_ppm", "deltaM_da", "absDeltaM_ppm", "absDeltaM_da",
    "missedCleavages", "sequence_length", "andromeda", "delta_score",
    "Charge2", "Charge3", "Peptide", "Protein"
]


def andromeda_feature(msms: pd.DataFrame, *args):
    """Generate features from tensors and msms file.

    Args:
        msms (pd.DataFrame): dataframe of msms file

    Returns:
        features (torch.Tensor): a tensor containing andromeda features
    """
    features = {}
    features["SpecId"] = msms["id"]
    features["ScanNr"] = msms["Scan number"]
    features["Label"] = msms["Reverse"].isnull() * 2 - 1
    features["ExpMass"] = 1000
    features["Mass"] = msms["Mass"]
    features["Peptide"] = ["_." + m + "._" for m in msms["Sequence"]]
    features["Protein"] = msms["Sequence"]
    features["Charge2"] = (msms["Charge"] == 2).to_numpy(dtype=int)
    features["Charge3"] = (msms["Charge"] == 3).to_numpy(dtype=int)
    features["missedCleavages"] = msms["Missed cleavages"]
    features["sequence_length"] = msms["Length"]
    features["deltaM_ppm"] = [
        0. if pd.isna(i) else i
        for i in msms["Mass Error [ppm]"]
    ]
    features["absDeltaM_ppm"] = [
        0. if pd.isna(i) else abs(i)
        for i in msms["Mass Error [ppm]"]
    ]
    features["deltaM_da"] = [
        p / 1e6 * m for m, p
        in zip(features["Mass"], features["deltaM_ppm"])
    ]
    features["absDeltaM_da"] = [
        abs(p / 1e6 * m) for m, p
        in zip(features["Mass"], features["deltaM_ppm"])
    ]
    features["andromeda"] = msms["Score"]
    features["delta_score"] = msms["Delta score"]
    features["KR"] = [
        sum(map(lambda x: 1 if x in "KR" else 0, s))
        for s in msms["Sequence"]
    ]

    return pd.DataFrame(features, columns=ANDROMEDA_FEATURES)


def match_ions(
    msms_file: str,
    csv_dir: str
):
    """Match observed ions with theoretical ions.

    Args:
        msms_file (str): file path of msms.txt
        csv_dir (str): directory of csv files

    Returns:
        df (pd.DataFrame): andromeda features
        psm (dict): dict of psm data
    """
    print(f"Reading msms data from {msms_file} ...")
    df = pd.read_csv(msms_file, sep="\t")

    # filter rows using the following columns
    filter_need = ["Fragmentation", "Charge", "Length", "Score",
                   "Modified sequence", "All modified sequences"]
    filter_need = df[filter_need]
    print("Filtering msms data ...")
    msms_df = df[filter_need.apply(psm_filter, axis=1)]
    msms_df = msms_df.sort_values(by="id").reset_index(drop=True)
    print(f"Msms data loaded, {len(msms_df)} PSMs selected.")

    # get the raw data
    print(f"Reading csv files in {csv_dir} ...")
    raw_dfs = collect_raw(csv_dir)
    print(f"Raw data loaded from {len(raw_dfs)} files.")

    psm = {
        "sequences": [],
        "lengths": [],
        "charges": [],
        "observe_intensities": []
    }

    print(f"Matching ions, {len(msms_df)} PSMs to match ...")

    selected_rows = []
    for index, msms in tqdm(msms_df.iterrows()):
        # get the corresponding raw data
        raw = raw_dfs[msms["Raw file"]]
        raw = raw.loc[raw["scan_number"] == msms["Scan number"]]
        if len(raw):
            # suppose scan_number is an unique key
            raw = raw.to_dict(orient="records")[0]
        else:
            # no corresponding scan number in csv file
            continue

        # prepare data for matching
        sequence = msms["Modified sequence"].strip("_")
        amino_acid = peptide_parser(sequence)
        amino_acid = list(amino_acid)
        intensity = match_single_psm(
            analyzer=raw["scan_type"].split(" ")[0],
            masses_raw=raw["masses"],
            intensities_raw=raw["intensities"],
            amino_acids=amino_acid,
            charge=msms["Charge"]
        )
        amino_acid = peptide_encoder(amino_acid)

        selected_rows.append(index)
        psm["sequences"].append(amino_acid)
        psm["lengths"].append(msms["Length"])
        psm["charges"].append(msms["Charge"])
        psm["observe_intensities"].append(intensity)
    # end of for loop
    print(f"Matching ions done, {len(selected_rows)} PSMs matched.")

    # select the corresponding rows
    msms_df = msms_df.iloc[selected_rows]

    # concatenate the result
    psm["sequences"] = np.vstack(psm["sequences"])
    psm["lengths"] = np.array(psm["lengths"])
    psm["charges"] = np.array(psm["charges"])
    psm["observe_intensities"] = np.vstack(psm["observe_intensities"])

    return andromeda_feature(msms_df), psm


def process_msms(
    msms_file: str,
    csv_dir: str,
    save_dir: str
):
    """Process msms data.

    Args:
        msms_file (str): file path of msms.txt
        csv_dir (str): directory of csv files
        save_dir (str): directory to save the result

    Returns:
        df (pd.DataFrame): andromeda features
        psm (dict): dict of psm data
    """
    df, psm = match_ions(msms_file, csv_dir)
    andromeda = os.path.join(save_dir, "andromeda_features.tab")
    df.to_csv(andromeda, index=False, sep="\t")
    save_processed_raw(psm, save_dir)
    return df, psm


def save_processed_raw(
    psm: Dict[str, np.ndarray],
    save_dir: str
):
    """Save the result to a hdf5 file.

    Args:
        psm (dict): dict of psm data
        save_dir (str): directory to save the result
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filepath = os.path.join(save_dir, "msms_raw.hdf5")

    with h5py.File(filepath, "w") as file:
        for key, value in psm.items():
            file[key] = value


def load_processed_raw(save_dir: str):
    """Load processed raw data from hdf5 file."""

    file_path = os.path.join(save_dir, "msms_raw.hdf5")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    with h5py.File(file_path) as file:
        rst = {}
        for key in file.keys():
            rst[key] = file[key][:]
        return rst
