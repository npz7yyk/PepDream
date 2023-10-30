# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from argparse import Namespace

from .constants import ALPHABET, MAX_ION, MAX_FRAG_CHARGE, ION_TYPES


# bioinformatics parameters
amino_acid = len(ALPHABET) + 1
sequence_length = MAX_ION
fragment_type = MAX_FRAG_CHARGE * len(ION_TYPES)
# end of bioinformatics parameters

# model parameters
embedding_dim = 36

gru2cnn_channels = 6
gru_num_layers = 2

cnn_base_channels = 3
cnn_inter_channels = 64
cnn_out_channels = 64
cnn_kernel1_width = 3
cnn_kernel2_width = 4

dnn_dropout_rate = 0.0

mlp_output_dim1 = 512
mlp_output_dim2 = 256
mlp_output_dim3 = 64

num_features = 11
# end of model parameters

# loss parameters
pos_smoothing = 0.99
neg_smoothing = 0.99
false_positive_loss_factor = 4.0
gamma = 0.5
# end of loss parameters


def set_hyperparams(hyperparams: Namespace):
    """Update hyperparameters."""
    global_vars = globals()
    params = global_vars.keys()
    for k, v in vars(hyperparams).items():
        if k in params:
            global_vars[k] = v


def set_num_features(features: int):
    """Update number of features."""
    global num_features
    num_features = features


def update_dropout_rate(dropout_rate: float):
    """Update dropout rate."""
    global dnn_dropout_rate
    dnn_dropout_rate = dropout_rate


class PepDream(nn.Module):
    def __init__(self):
        super(PepDream, self).__init__()

        self.acid_embedding = nn.Embedding(
            num_embeddings=amino_acid,
            embedding_dim=embedding_dim
        )
        assert gru2cnn_channels * fragment_type % 2 == 0, \
            "gru2cnn_channels * fragment_type should be even"
        gru_hidden_size = int(gru2cnn_channels * fragment_type / 2)
        self.gru_hidden_size = gru_hidden_size
        self.biGRU = nn.GRU(
            input_size=embedding_dim,
            hidden_size=gru_hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=gru_num_layers,
        )

        assert cnn_kernel1_width + cnn_kernel2_width == fragment_type + 1, \
            "sum of kernel widths should be equal to fragment_type + 1"
        self.CNN_1 = nn.Conv2d(
            in_channels=gru2cnn_channels + cnn_base_channels,
            out_channels=cnn_inter_channels,
            kernel_size=(cnn_kernel1_width, cnn_kernel1_width)
        )
        self.CNN_2 = nn.Conv2d(
            in_channels=cnn_inter_channels,
            out_channels=cnn_out_channels,
            kernel_size=(cnn_kernel2_width, cnn_kernel2_width)
        )

        self.dropout = nn.Dropout(dnn_dropout_rate)

        channel_dim = (sequence_length - cnn_kernel1_width - cnn_kernel2_width + 2)
        self.MLP_1 = nn.Linear(channel_dim * cnn_out_channels, mlp_output_dim1)
        self.MLP_2 = nn.Linear(mlp_output_dim1, mlp_output_dim2)
        self.MLP_3 = nn.Linear(mlp_output_dim2 + num_features, mlp_output_dim3)
        self.MLP_4 = nn.Linear(mlp_output_dim3, 2)

    def __str__(self):
        return "PepDream"

    def sequence_embedding(self, sequence: torch.Tensor):
        acid = self.acid_embedding(sequence)
        acid, _ = self.biGRU(acid)
        gru_hidden_size = self.gru_hidden_size
        acid = torch.concat(
            [acid[:, 1:, :gru_hidden_size],
             acid[:, :-1, gru_hidden_size:]],
            dim=2
        )

        batch_size = acid.shape[0]
        # acid.shape = (batch_size, sequence_length, 2 * gru_hidden_size)
        acid = acid.transpose(1, 2)
        # acid.shape = (batch_size, 2 * gru_hidden_size, sequence_length)
        acid = acid.reshape(batch_size, -1, fragment_type, sequence_length)
        # acid.shape = (batch_size, gru2cnn_channels, fragment_type, sequence_length)
        acid = acid.transpose(2, 3)
        # acid.shape = (batch_size, gru2cnn_channels, sequence_length, fragment_type)
        return acid

    def __call__(self, feature, sequence, observe, predict):
        # embedding sequence
        acid = self.sequence_embedding(sequence)

        multiply = observe * predict
        ions = torch.cat([observe, predict, multiply], dim=1)

        batch_size = acid.shape[0]
        # ions.shape = (batch_size, cnn_channels, sequence_length * fragment_type)
        ions = ions.reshape(batch_size, -1, sequence_length, fragment_type)
        # ions.shape = (batch_size, cnn_base_channels, sequence_length, fragment_type)
        ions = torch.cat([ions, acid], dim=1)

        ions = torch.relu(self.CNN_1(ions))
        ions = torch.relu(self.CNN_2(ions))
        # ions.shape = (batch_size, cnn_out_channels, channel_dim)
        ions = ions.reshape(ions.shape[0], -1)
        # ions.shape = (batch_size, cnn_out_channels * channel_dim)

        x = torch.relu(self.MLP_1(ions))
        x = self.dropout(x)

        x = torch.relu(self.MLP_2(x))
        x = self.dropout(x)

        x = torch.relu(self.MLP_3(torch.concat([x, feature], dim=1)))
        x = self.dropout(x)

        x = self.MLP_4(x)
        return x if self.training else F.softmax(x, dim=1)[:, 1]

    def build_optimizer(self, learning_rate: float, weight_decay: float):
        return torch.optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def build_loss(self, class_weights: List[float]):
        return LabelSmoothingLoss(class_weights)


class LabelSmoothingLoss(nn.Module):
    def __init__(self, class_weights: List[float]):
        """KL-divergence with label smoothing.

        Args:
            device (torch.device): device to use for computation
            class_weights (list): list of floats to weight classes

        Returns:
            LabelSmoothingLoss: loss function
        """
        super(LabelSmoothingLoss, self).__init__()

        # soft_distribs[i, j]:
        # probability of predicting class j when the true class is i
        class_confidence = [neg_smoothing, pos_smoothing]
        num_classes = len(class_confidence)
        soft_distribs = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.float32
        )
        for i in range(num_classes):
            rest = (1 - class_confidence[i]) / (num_classes - 1)
            soft_distribs[i, :] = rest
            soft_distribs[i, i] = class_confidence[i]
        self.register_buffer("soft_distribs", soft_distribs)

        # class_weights[i]: weight for class i
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, output, labels):
        """Compute KL-divergence loss with label smoothing.

        Args:
            output (torch.Tensor, shape=(batch_size, num_classes)): output of model
            labels (torch.Tensor, shape=(batch_size,)): labels of data
            spectrums (torch.Tensor, shape=(batch_size,)): spectrum loss of data

        Returns:
            torch.Tensor: loss
        """
        batch = len(labels)

        # compute KL-divergence loss
        soft_labels = self.soft_distribs[labels]
        softmax_preds = F.log_softmax(output, dim=1)
        kl_loss = F.kl_div(
            softmax_preds,
            soft_labels,
            reduction='none'
        ).mean(dim=1)

        # compute weights
        fp_factor = false_positive_loss_factor
        weights = self.class_weights[labels]
        negative = labels == 0
        idx = negative & (softmax_preds[:, 1] >= -0.693147180559)   # -ln2
        fpsum = torch.sum(idx)
        # normalize weights to maintain overall loss
        rest_factor = (batch - fpsum * fp_factor) / (batch - fpsum)
        weights[idx == 1] *= fp_factor
        weights[idx == 0] *= rest_factor

        loss = weights * kl_loss
        return torch.mean(loss)


class SpectrumNormalizer(nn.Module):
    def __init__(self):
        super(SpectrumNormalizer, self).__init__()

    def forward(self, spectrum: torch.Tensor):
        # mask negative values
        spectrum = torch.relu(spectrum)
        # L2 normalization
        spectrum = spectrum / torch.norm(spectrum, dim=1, keepdim=True)

        return spectrum


class FuseOptimizer:
    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()


class SpectrumAngle(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, observe, predict):
        dot_product = torch.sum(observe * predict, dim=1)
        return 1 - (2 / torch.pi) * torch.arccos(dot_product)


class FinetuneLoss(nn.Module):
    def __init__(self, class_weights: List[float]):
        super(FinetuneLoss, self).__init__()
        self.spectrum_angle = SpectrumAngle()
        self.label_loss = LabelSmoothingLoss(class_weights)
        self.mse = nn.MSELoss()

    def forward(self, output, labels: torch.Tensor):
        observe, predict, rescore = output
        spectrum_angle = self.spectrum_angle(observe, predict)
        spectrum_loss = self.mse(spectrum_angle, labels.float())
        spectrum_loss = torch.mean(spectrum_loss) * gamma
        return self.label_loss(rescore, labels) + spectrum_loss


class ConcatedModel(nn.Module):
    def __init__(
        self,
        spectrum_model: nn.Module,
        feature_model_weights: Dict[str, torch.Tensor]
    ):
        super(ConcatedModel, self).__init__()
        self.spectrum_model = spectrum_model
        self.nomalizer = SpectrumNormalizer()
        self.feature_model = PepDream()
        self.feature_model.load_state_dict(feature_model_weights)

    def __str__(self):
        return f"ConcatedModel ({self.spectrum_model} + PepDream)"

    @classmethod
    def spectrum_model_weights(self, weights: Dict[str, torch.Tensor]):
        return {
            key.removeprefix("spectrum_model."): value
            for key, value in weights.items()
            if key.startswith("spectrum_model.")
        }

    def forward(self, *args):
        predict = self.spectrum_model(args[0])
        predict = self.nomalizer(predict)
        rescore = self.feature_model(*args[1:], predict)
        if self.training:
            return (args[-1], predict, rescore)
        else:
            return rescore

    def build_optimizer(self, learning_rate: float, weight_decay: float):
        spectrum_optimizer = torch.optim.AdamW(
            self.spectrum_model.parameters(),
            lr=learning_rate
        )
        feature_optimizer = self.feature_model.build_optimizer(
            learning_rate,
            weight_decay
        )
        return FuseOptimizer([spectrum_optimizer, feature_optimizer])

    def build_loss(self, class_weights: List[float]):
        return FinetuneLoss(class_weights)
