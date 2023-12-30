# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from os.path import join
from collections import defaultdict
from torch.backends import cudnn
from argparse import Namespace
from typing import Callable, Optional, Union, Iterable, Tuple, List, Dict

from .pepdream import update_dropout_rate, PepDream, ConcatedModel
from .qvalue_utils import qvalue, target_identified, qvalue_auc


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


# hyperparameters
device = torch.device("cpu")

q = 0.01
deepq = 0.07
evalq = 0.10

k = 3
total_itrs = 8

batch_size = 5000

lr_init = 0.001
weight_decay = 2e-5
dropout_rate = 0.5

ensemble_snaps = 10
ensemble_epoch = 5
ensemble_count = 16

cache_dir = ""
output_dir = ""
output_interval = 1
spectrum_model_type = "prosit"


def set_hyperparams(hyperparams: Namespace):
    """Update hyperparameters."""
    global_vars = globals()
    params = global_vars.keys()
    for k, v in vars(hyperparams).items():
        if k in params:
            global_vars[k] = v
# end of hyperparameters


# log management
_ident = 0
_ident_str = "    "
_flush = True


def add_indent():
    """Increase indent level."""
    global _ident
    _ident += 1


def sub_indent():
    """Decrease indent level."""
    global _ident
    _ident -= 1


def printlog(msg: str):
    """Print log message with indent."""
    print(_ident_str * _ident + msg)
    if _flush:
        sys.stdout.flush()
# end of log management


# data split keys
train_keys = None
test_keys = None


def split_data(scan_numbers: List[int]):
    """Split data into k folds."""
    global train_keys, test_keys
    test_keys = [list() for _ in range(k)]

    unique_scan_number = list(set(scan_numbers))
    random.shuffle(unique_scan_number)

    num2index = defaultdict(list)
    for i, num in enumerate(scan_numbers):
        num2index[num].append(i)

    for scan_number in unique_scan_number:
        min(test_keys, key=len).extend(num2index[scan_number])

    train_keys = []
    for i in range(k):
        train_keys.append(
            np.concatenate(test_keys[:i] + test_keys[i + 1:])
        )


def load_keys(
    _test_keys: Iterable[np.ndarray]
):
    """Load data split keys."""
    global train_keys, test_keys
    test_keys = _test_keys
    _train_keys = []
    for i in range(k):
        _train_keys.append(
            np.concatenate(test_keys[:i] + test_keys[i + 1:])
        )
    train_keys = _train_keys
# end of data split keys


class EnsembleWrapper(object):
    def __init__(
        self,
        model: nn.Module,
        weights_list: List[Tuple[int, Dict[str, torch.Tensor]]]
    ):
        """Builds an ensemble of models from a list of weights.

        Args:
            model (nn.Module): model class
            weights_list (List[Tuple[int, Dict[str, torch.Tensor]]]): list of weights

        Returns:
            EnsembleWrapper: ensemble of models
        """
        self.model = model
        self.model_count = sum(t for t, _ in weights_list)
        self.weights_list = weights_list

    def __call__(self, *args):
        """Runs the ensemble on a batch of data.

        Args:
            x (torch.Tensor): input data

        Returns:
            torch.Tensor: ensemble predictions
        """
        rst = None
        model = self.model.to(device).eval()
        for times, weight in self.weights_list:
            set_model_param(self.model, weight)
            if rst is None:
                rst = times * model(*args)
            else:
                rst += times * model(*args)

        return rst / self.model_count

    def main_weights(self):
        """Returns the weights that has the most votes."""
        max_time = 0
        rst = None
        for t, w in reversed(self.weights_list):
            rst = w if t > max_time else rst
            max_time = max(max_time, t)
        return rst

    def eval(self):
        # Already done in __init__
        return self

    def train(self):
        # EnsembleWrapper is not trainable
        return self


class ArrayLoader:
    """A fast implementation of DataLoader for numpy arrays.

    Args:
        batch_size (int): batch size
        arrays (Iterable[Union[np.ndarray, Dict[str, np.ndarray]]]): input data
        labels (Optional[np.ndarray], optional): labels, default=None
        shuffle (Optional[bool], optional): whether to shuffle, default=False

    Returns:
        Iterable: Iterable for the input data

    @remarks:
        the reason why this class is fast is that:
        1. arrays are concatenated, which means less index operations
        2. arrays are send to GPU together (less "to(device)" calls)

    @attention:
        we assume that:
        1. all arrays are 2D
        2. all arrays are of the same length
    """
    def __init__(
        self,
        batch_size: int,
        arrays: Iterable[Union[np.ndarray, Dict[str, np.ndarray]]],
        labels: Optional[np.ndarray] = None,
        shuffle: Optional[bool] = False
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle

        _arrays = []
        names = []
        dicts = []
        for x in arrays:
            if isinstance(x, dict):
                _arrays.extend(x.values())
                names.extend(x.keys())
                dicts += [True] + [False] * (len(x) - 1)
            else:
                _arrays.append(x)
                names.append(None)
                dicts.append(False)
        arrays = _arrays
        self.names = names
        self.dicts = dicts

        # collect the second dimension of all arrays
        self.array_tails = list(np.cumsum([x.shape[1] for x in arrays]))
        self.array_heads = [0] + self.array_tails[:-1]

        # collect the types of all arrays for reconstruction
        self.array_types = ['float' in str(x.dtype) for x in arrays]
        self.arrays = np.concatenate(arrays, axis=1, dtype=np.float32)
        self.labels = labels

    def __iter__(self):
        size = len(self.arrays)

        # local variables for faster access
        heads = self.array_heads
        tails = self.array_tails
        types = self.array_types
        names = self.names
        dicts = self.dicts
        batch_size = self.batch_size

        # shuffle indices
        reindex = np.arange(size)
        reindex = np.random.permutation(reindex) if self.shuffle else reindex

        # reindex arrays and labels
        arrays = self.arrays[reindex]
        labels = self.labels[reindex] if self.labels is not None else None

        for ptr in range(0, size, batch_size):
            # get the end index
            end = ptr + batch_size
            # only two to(device) calls
            x = torch.from_numpy(arrays[ptr:end]).to(device)
            if labels is not None:
                y = torch.from_numpy(labels[ptr:end])
                y = y.to(dtype=torch.int32, device=device)
            # reconstruct arrays
            x = [
                x[:, h:t] if d else x[:, h:t].to(torch.int32)
                for h, t, d in zip(heads, tails, types)
            ]
            # reconstruct dicts
            rst = []
            for array, name, new_dict in zip(x, names, dicts):
                if name is None:
                    rst.append(array)
                else:
                    if new_dict:
                        rst.append({name: array})
                    else:
                        rst[-1][name] = array
            yield (*rst, y) if labels is not None else rst


def train_single_fold(
    model,
    train_features,
    train_labels,
    validation_Features,
    validation_Labels
):
    printlog(f"New {model} model on device {device}")

    model_params = train_ensemble_model(
        model=model,
        features=train_features,
        labels=train_labels
    )

    model, prediction = make_ensemble(
        model=model,
        param_list=model_params,
        validation_features=validation_Features,
        validation_labels=validation_Labels
    )

    return model, prediction


def indexer(x: Union[np.ndarray, Dict[str, np.ndarray]], keys: np.ndarray):
    return {k: x[k][keys] for k in x} if isinstance(x, dict) else x[keys]


def run_iteration(
    model_provider: Callable[[Dict[str, torch.Tensor]], nn.Module],
    features: Iterable[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    feature_model_weights: Optional[Dict[str, torch.Tensor]] = None
):
    new_scores = []
    models = []

    for fold, keys in enumerate(train_keys):
        # Find training set using q-value analysis
        index_train = target_identified(scores[fold], labels[keys], deepq)
        index_train = index_train | (labels[keys] == 0)

        validation_features = [indexer(f, keys) for f in features]
        validation_labels = labels[keys]
        train_features = [indexer(f, index_train) for f in validation_features]
        train_labels = validation_labels[index_train]

        # Train a new model or load a pretrained model
        if feature_model_weights:
            model = model_provider(feature_model_weights[fold])
        else:
            model = model_provider()
        printlog(f"Start training {model} for fold {fold + 1}.")
        msg = f"Training on {len(train_labels)} PSMs, "
        msg += f"validating on {len(validation_labels)} PSMs."
        printlog(msg)

        add_indent()
        model, new_score = train_single_fold(
            model.to(device),
            train_features, train_labels,
            validation_features, validation_labels
        )
        sub_indent()

        models.append(model)
        new_scores.append(new_score)

        count = np.sum(target_identified(new_score, validation_labels, deepq))
        printlog(
            f"Fold {fold + 1} finished. At q <= {deepq}, "
            f"{count} targets identified in self-validation."
        )

    return new_scores, models


def run_prediction(
    trained_models: Iterable[EnsembleWrapper],
    features: Iterable[Union[np.ndarray, Dict[str, np.ndarray]]]
):
    """Execute pre-trained models on their respective unseen folds."""
    scores = np.zeros(sum(len(key) for key in test_keys))
    for model, key in zip(trained_models, test_keys):
        test_features = [indexer(f, key) for f in features]
        scores[key] = inference(model, test_features)

    return scores


def write_output(filename, labels, scores, qvalues, psm_info):
    df = pd.DataFrame({
        'id': psm_info[0],
        "label": labels,
        'score': scores,
        'q-value': qvalues,
        'peptide': psm_info[1],
        'protein': psm_info[2]
    })
    df.to_csv(filename, sep='\t', index=False)


def run_algorithm(
    features: Iterable[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    psm_info: Iterable[Iterable[str]]
):
    # rebuild scores
    scores = [scores[keys] for keys in train_keys]

    for itr in range(total_itrs):
        printlog(f"Iteration {itr + 1} started.")
        add_indent()

        # run a single iteration
        scores, trained_models = run_iteration(
            model_provider=lambda: PepDream(),
            features=features,
            labels=labels,
            scores=scores
        )

        sub_indent()
        printlog(f"Iteration {itr + 1} finished.")

        # dump identified targets
        if (itr + 1) % output_interval == 0 or itr == total_itrs - 1:
            pred_scores = run_prediction(trained_models, features)
            qvalues = qvalue(pred_scores, labels)

            write_output(
                filename=join(output_dir, f"output_iteration{itr + 1}.txt"),
                labels=labels,
                scores=pred_scores,
                qvalues=qvalues,
                psm_info=psm_info
            )

            target = np.sum((qvalues <= q) & (labels == 1))
            printlog(f"At q <= {q}, {target} targets identified. Results dumped.")

    return scores, [model.main_weights() for model in trained_models]


def run_finetune(
    feature_model_weights: Dict[str, torch.Tensor],
    features: Iterable[np.ndarray],
    labels: np.ndarray,
    scores: np.ndarray,
    psm_info: Iterable[Iterable[str]]
):
    """Finetune a pretrained model.

    Args:
        pretrained_model (nn.Module): pretrained spectrum prediction model
        feature_model_weights (Dict[str, torch.Tensor]): weights of feature model
        features (Iterable[np.ndarray]): features
        labels (np.ndarray): labels
        scores (np.ndarray): scores
        psm_info (Iterable[Iterable[str]]): PSM information
    """
    print(f"Finetuning {spectrum_model_type} model.")
    update_dropout_rate(dropout_rate)

    from .predict_spectrum import pretrained_model
    scores, trained_models = run_iteration(
        model_provider=lambda w: ConcatedModel(
            spectrum_model=pretrained_model(spectrum_model_type),
            feature_model_weights=w
        ),
        features=features,
        labels=labels, scores=scores,
        feature_model_weights=feature_model_weights
    )
    pred_scores = run_prediction(trained_models, features)
    qvalues = qvalue(pred_scores, labels)

    write_output(
        filename=join(output_dir, "output_finetune.txt"),
        labels=labels,
        scores=pred_scores,
        qvalues=qvalues,
        psm_info=psm_info
    )

    target = np.sum((qvalues <= q) & (labels == 1))
    printlog(f"At q <= {q}, {target} targets identified. Results dumped.")

    # save spectrum prediction model weights for evaluation
    weights_lists = [model.weights_list for model in trained_models]
    for fold, weights_list in enumerate(weights_lists):
        for times, weights in weights_list:
            weights = ConcatedModel.spectrum_model_weights(weights)
            path = "finetuned_" + spectrum_model_type
            path += f"/fold{fold + 1}_times{times}.pth"
            torch.save(weights, join(output_dir, path))


def make_ensemble(
    model: nn.Module,
    param_list: List[Dict[str, torch.Tensor]],
    validation_features: Iterable[np.ndarray],
    validation_labels: np.ndarray
):
    """Builds an ensemble of models from a list of weights.

    Args:
        model (nn.Module): model to ensemble
        param_list (List[Dict[str, torch.Tensor]]): list of weights
        validation_features (Iterable[np.ndarray]): validation features
        validation_labels (np.ndarray): validation labels
        criterion (Callable[[np.ndarray, np.ndarray], float]): used for model selection

    Returns:
        EnsembleWrapper: ensemble wrapper
        np.ndarray: predictions
    """
    preds = []
    for param in param_list:
        set_model_param(model, param)
        preds.append(inference(model, validation_features))

    # criterion is the metric used for model selection
    criterion = qvalue_auc(evalq)

    param_times, prediction = select_ensemble_greedy(
        prediction_list=preds,
        labels=validation_labels,
        criterion=criterion
    )
    rst = EnsembleWrapper(
        model=model,
        weights_list=[(t, p) for t, p in zip(param_times, param_list) if t > 0]
    )
    return rst, prediction


def select_ensemble_greedy(
    prediction_list: List[np.ndarray],
    labels: np.ndarray,
    criterion: Callable[[np.ndarray, np.ndarray], float],
):
    """Selects a subset of models from a list of predictions.

    Args:
        prediction_list (List[np.ndarray]): list of predictions
        labels (np.ndarray): labels
        criterion (Callable[[np.ndarray, np.ndarray], float]): metric to use

    Returns:
        np.ndarray: number of times each model is selected
        np.ndarray: ensemble prediction
    """
    param_times = np.zeros(len(prediction_list), dtype=int)
    ensemble_predictions = np.zeros_like(prediction_list[0])
    for _ in range(ensemble_count):
        best_to_add = -1
        best_score = 0
        for i, p in enumerate(prediction_list):
            # no need to be divided by param_times[i] + 1
            ensemble_pred = (ensemble_predictions + p)
            ensemble_score = criterion(ensemble_pred, labels)
            if ensemble_score > best_score:
                best_score = ensemble_score
                best_to_add = i
        if best_to_add == -1:
            # no model improves the ensemble
            break
        param_times[best_to_add] += 1
        ensemble_predictions += prediction_list[best_to_add]
    return param_times, ensemble_predictions


def get_model_param(model: nn.Module):
    model_param = {}
    for k, v in model.state_dict().items():
        model_param[k] = v.cpu()
    return model_param


def set_model_param(
    model: nn.Module,
    model_param: Dict[str, torch.Tensor]
):
    state_dict = {}
    for k, v in model_param.items():
        state_dict[k] = v.to(device)
    model.load_state_dict(state_dict)


def inference(
    model: Union[nn.Module, EnsembleWrapper],
    data: Union[Iterable[np.ndarray], ArrayLoader]
):
    """Runs the model on a batch of data.

    Args:
        model (Union[nn.Module, EnsembleWrapper]): model to run
        data (Union[Iterable[np.ndarray], DataLoader]): input data

    Returns:
        np.ndarray: model predictions
    """
    if isinstance(data, ArrayLoader):
        loader = data
    else:
        loader = ArrayLoader(batch_size * 2, data)
    rst = []
    with torch.no_grad():
        model.eval()
        for batch in loader:
            rst.append(model(*batch))
        model.train()
    return torch.cat(rst).cpu().numpy()


def train_ensemble_model(
    model: Union[PepDream, ConcatedModel],
    features: Iterable[np.ndarray],
    labels: np.ndarray
):
    optimizer = model.build_optimizer(lr_init, weight_decay)

    pos = np.sum(labels == 1)
    total = len(labels)
    neg = total - pos
    # Attention!!! When placing the weights like this,
    # it actually makes the bigger class speaks louder.
    # But it seems to work better.
    class_ratios = [pos / total, neg / total]
    class_weights = [0.5 / ratio for ratio in class_ratios]

    # loss_func = LabelSmoothingLoss(class_weights).to(device)
    loss_func = model.build_loss(class_weights).to(device)

    loader = ArrayLoader(
        batch_size=batch_size,
        arrays=features,
        labels=labels,
        shuffle=True
    )

    model_params = []
    start = time.time()
    for ensemble in range(ensemble_snaps):
        for _ in range(ensemble_epoch):
            losses = []
            for batch in loader:
                optimizer.zero_grad()
                outputs = model(*batch[:-1])
                loss: torch.Tensor = loss_func(outputs, batch[-1])
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        msg = f"Voter {ensemble + 1:2d}/{ensemble_snaps} training "
        msg += f"completed with average loss {np.mean(losses):6.4f}"
        printlog(msg)
        # end of one epoch

        model_params.append(get_model_param(model))
    # end of one voter

    # print training speed
    delta = time.time() - start
    speed = delta / ensemble_snaps / ensemble_epoch
    printlog(f'Training @ {speed:5.3f} second/epoch')

    return model_params


def save_footprint(
    scores: Iterable[np.ndarray],
    weights: Iterable[Dict[str, torch.Tensor]]
):
    """Save training folds and scores for finetuning."""
    import h5py
    with h5py.File(join(cache_dir, "training_footprint.hdf5"), "w") as file:
        file["scores"] = np.concatenate(scores)
        file["splits"] = np.array([len(key) for key in test_keys])
        file["test_keys"] = np.concatenate(test_keys)
    for fold, weight in enumerate(weights):
        path = f"pepdream_fold{fold + 1}.pth"
        torch.save(weight, join(cache_dir, path))


def load_footprint():
    """Load training folds and scores for finetuning."""
    import h5py
    with h5py.File(join(cache_dir, "training_footprint.hdf5"), "r") as file:
        scores = file["scores"][:]
        splits = file["splits"][:]
        test_keys = file["test_keys"][:]

    # reconstruct keys
    splits = np.cumsum(splits)[:-1]
    test_keys = np.split(test_keys, splits)
    load_keys(test_keys)

    # reconstruct scores
    splits = np.cumsum([len(key) for key in train_keys])[:-1]
    scores = np.split(scores, splits)

    # load weights
    weights = []
    for fold in range(k):
        path = f"pepdream_fold{fold + 1}.pth"
        weights.append(torch.load(join(cache_dir, path)))

    return scores, weights
