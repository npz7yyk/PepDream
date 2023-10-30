# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict

from . import prosit
from . import pdeep2
from . import constants


# training device
device = torch.device("cpu")


def set_device(_device: torch.device):
    global device
    device = _device
# end device


DEFALUT_COLLISION_ENERGY = 33.0


def collison_energy(size: int, value: float = DEFALUT_COLLISION_ENERGY):
    """Generate normalized collision energy."""
    assert 0 <= value <= 100, "Collision energy should be in [0, 100]."
    return np.ones((size, 1), dtype=float) * value / 100.0


def prepare_data(data: Dict[str, np.ndarray], tensorize: bool = True):
    """Prepare data for training."""

    def onehot(array: np.ndarray, num_classes: int):
        return np.eye(num_classes)[array]

    # process sequences
    sequences = data["sequences"]

    # process collision energy
    size = len(sequences)
    normalized_collision_energy = collison_energy(size)

    # process charges
    charges = onehot(data["charges"] - 1, constants.DEFAULT_MAX_CHARGE)

    # process lengths
    lengths: np.ndarray = data["lengths"]

    rst = {}
    rst["sequences"] = sequences
    rst["normalized_collision_energy"] = normalized_collision_energy
    rst["charges"] = charges
    rst["lengths"] = lengths.reshape(-1, 1)

    if tensorize:
        for key, value in rst.items():
            rst[key] = torch.from_numpy(value).to(device)

    return rst


BATCH_SIZE = 8192


def inference(model: nn.Module, data: Dict[str, torch.Tensor]):
    """Run model inference on data."""

    model.eval().to(device)
    rst = []
    size = len(data["sequences"])
    batch_size = BATCH_SIZE
    with torch.no_grad():
        for batch in tqdm(range(0, size, batch_size)):
            batch_data = {
                key: value[batch:batch + batch_size]
                for key, value in data.items()
            }
            rst.append(model(batch_data).cpu().numpy())

    return np.concatenate(rst)


def save_predicted_spectrum(
    result: np.ndarray,
    save_dir: str,
    model: str
):
    """Save predicted spectrum."""

    file_path = os.path.join(save_dir, "msms_raw.hdf5")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    link = "predict_intensities_" + model
    with h5py.File(file_path, "a") as file:
        if link in file:
            del file[link]
        file["predict_intensities_" + model] = result


def pretrained_model(model: str):
    # spectrum prediction model
    # model input is a subset of (given in Dict[str, torch.Tensor]):
    # 1. amino acid sequence: "sequences", (batch_size, sequence_length)
    # 2. normalized collision energy: "normalized_collision_energy" (batch_size, )
    # 3. onehot charges: "charges", (batch_size, max_charge)
    # 4. sequence length: "lengths", (batch_size, )
    # other spectrum prediction models can also be supported like this
    if model == "prosit":
        SPECTRUM_MODEL = prosit.pretrained_prosit()
    elif model == "pdeep2":
        SPECTRUM_MODEL = pdeep2.pretrained_pdeep2()
    else:
        raise ValueError(f"Model {model} is not implemented yet.")
    return SPECTRUM_MODEL


def predict_spectrum(
    raw: Dict[str, np.ndarray],
    save_dir: str,
    model: str = "prosit",
):
    """Predict spectrum for given hdf5 file and save the result if required.

    Args:
        raw (Dict[str, np.ndarray]): raw data for PSM
        save_dir (str): result will be saved in save_dir/*.hdf5(s)
        model (str): model name, default: "prosit", should be one of ["prosit", "pdeep2"]

    Returns:
        np.ndarray: predicted spectrum
    """
    SPECTRUM_MODEL = pretrained_model(model)

    print(f"Predicting spectrum using {model} model...")
    spectrum = inference(SPECTRUM_MODEL, prepare_data(raw))
    print("Finished predicting spectrum.")
    save_predicted_spectrum(spectrum, save_dir, model)

    return spectrum
