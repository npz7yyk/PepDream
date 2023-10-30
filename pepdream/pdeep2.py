# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import torch
import torch.nn as nn
import torch.nn.functional as F

# PDeep spectrum prediction model
# copied from Pept3 and modified
# link: https://github.com/gusye1234/pept3
# license: MIT License


class PDeep2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.layer_size = 256
        self.peptide_dim = kwargs.pop('peptide_dim', 22)
        self.instrument_size = 8
        self.input_size = self.peptide_dim * 4 + 2 + 1 + 3
        self.ions_dim = kwargs.pop('ions_dim', 6)
        self.instrument_ce_scope = "instrument_nce"
        self.rnn_dropout = 0.2
        self.output_dropout = 0.2
        self.init_layers()

    def __str__(self):
        return "PDeep2"

    def init_layers(self):
        self.lstm_layer1 = nn.LSTM(
            self.input_size,
            self.layer_size,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_layer2 = nn.LSTM(
            self.layer_size * 2 + 1 + 3,
            self.layer_size,
            batch_first=True,
            bidirectional=True
        )
        self.lstm_output_layer = nn.LSTM(
            self.layer_size * 2 + 1 + 3,
            self.ions_dim,
            bidirectional=True,
            batch_first=True
        )
        self.linear_inst_proj = nn.Linear(
            self.instrument_size + 1, 3,
            bias=False
        )
        self.dropout = nn.Dropout(p=self.output_dropout)

    def pdeep2_long_feature(self, data):
        peptides = F.one_hot(
            data["sequences"].to(torch.int64),
            num_classes=self.peptide_dim
        )
        peptides_length = data["lengths"].squeeze(-1)
        batch, length, dim = peptides.shape
        assert dim == self.peptide_dim
        long_feature = peptides.new_zeros((batch, length - 1, dim * 4 + 2))
        long_feature[:, :, :dim] = peptides[:, :-1, :]
        long_feature[:, :, dim:2 * dim] = peptides[:, 1:, :]
        for i in range(length - 1):
            long_feature[:, i, 2 * dim:3 * dim] = \
                torch.sum(peptides[:, :i, :], dim=1) \
                if i != 0 \
                else peptides.new_zeros((batch, dim))
            long_feature[:, i, 3 * dim:4 * dim] = \
                torch.sum(peptides[:, i + 2:, :], dim=1) \
                if i == (length - 2) \
                else peptides.new_zeros((batch, dim))
            long_feature[:, i, 4 * dim] = 1 if (i == 0) else 0
            long_feature[:, i, 4 * dim + 1] = (peptides_length - 2) == i
        return long_feature

    def add_leng_dim(self, x, length):
        x = x.unsqueeze(dim=1)
        shape_repeat = [1] * len(x.shape)
        shape_repeat[1] = length
        return x.repeat(*shape_repeat)

    def forward(self, data):
        peptides = self.pdeep2_long_feature(data)  # n-1 input

        nce = data["normalized_collision_energy"].float()
        charge = data["charges"].float()
        charge = torch.argmax(charge, dim=1).unsqueeze(-1)

        B = peptides.shape[0]
        peptides_length = peptides.shape[1]
        inst_feat = charge.new_zeros((B, self.instrument_size))
        # ['QE', 'Velos', 'Elite', 'Fusion', 'Lumos', 'unknown']
        inst_feat[: 5] = 1
        charge = self.add_leng_dim(charge, peptides_length)
        nce = self.add_leng_dim(nce, peptides_length)
        inst_feat = self.add_leng_dim(inst_feat, peptides_length)

        proj_inst = self.linear_inst_proj(torch.cat([inst_feat, nce], dim=2))
        x = torch.cat([peptides, charge, proj_inst], dim=2)

        x, _ = self.lstm_layer1(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        x, _ = self.lstm_layer2(x)
        x = self.dropout(x)
        x = torch.cat([x, charge, proj_inst], dim=2)
        output, _ = self.lstm_output_layer(x)
        output = (output[:, :, :self.ions_dim] + output[:, :, self.ions_dim:])

        return output.reshape(B, -1)


# pretrained model
STATE_DICT = ""


def config_state_dict(state_dict_path):
    """Config PDeep2 state dict path.

    Args:
        state_dict_path (str): state dict path.
    """
    global STATE_DICT
    STATE_DICT = state_dict_path
# end of config_state_dict


def pretrained_pdeep2():
    """Load pretrained PDeep2 model.

    Returns:
        PDeep2: pretrained PDeep2 model
    """
    if not STATE_DICT:
        msg = "Please config PDeep2 state_dict path first. "
        msg += "For default PDeep2 fragment model, "
        msg += "please check out https://github.com/npz7yyk/PepDream, "
        msg += "download released model state pdeep2.pth and "
        msg += "config the path using --spectrum-model-weight argument."
        raise ValueError(msg)

    model = PDeep2()
    try:
        model.load_state_dict(torch.load(STATE_DICT))
    except RuntimeError:
        print("PDeep2 model is not compatible with provided state dict. ")
        exit(1)
    return model
