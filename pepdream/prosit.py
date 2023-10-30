# PepDream v0.1.0
# Written by Yikang Yue <yyk2020@mail.ustc.edu.cn>
#
# Copyright (C) 2023 Yikang Yue
# Licensed under MIT License with third-party code under other licenses.
# See LICENSE and README for more details

import torch
import torch.nn as nn

# Prosit fragment model
# copied from Pept3 and modified
# link: https://github.com/gusye1234/pept3
# license: MIT License


class AttentalSum(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.w = nn.Linear(input_dim, 1)
        self.act = nn.Tanh()
        self.soft = nn.Softmax(dim=0)

    def forward(self, x, src_mask=None):
        weight = self.w(x)
        weight = self.act(weight).clone()

        if src_mask is not None:
            weight[src_mask.transpose(0, 1)] = -torch.inf
        weight = self.soft(weight)

        weighted_embed = torch.sum(x * weight, dim=0)
        return weighted_embed


class Prosit(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.peptide_dim = kwargs.pop("peptide_dim", 22)
        self.peptide_embed_dim = kwargs.pop("peptide_embed_dim", 32)
        self.percursor_dim = kwargs.pop("peptide_embed_dim", 6)
        self.hidden_size = kwargs.pop("bi_dim", 256)
        self.max_sequence = kwargs.pop("max_lenght", 30)

        self.embedding = nn.Embedding(self.peptide_dim, self.peptide_embed_dim)
        self.bi = nn.GRU(input_size=self.peptide_embed_dim,
                         hidden_size=self.hidden_size,
                         bidirectional=True)
        self.drop3 = nn.Dropout(p=0.3)
        self.gru = nn.GRU(input_size=self.hidden_size * 2,
                          hidden_size=self.hidden_size * 2)
        self.agg = AttentalSum(self.hidden_size * 2)
        self.leaky = nn.LeakyReLU()

        self.side_encoder = nn.Linear(self.percursor_dim + 1,
                                      self.hidden_size * 2)

        self.gru_decoder = nn.GRU(input_size=self.hidden_size * 2,
                                  hidden_size=self.hidden_size * 2)
        self.in_frag = nn.Linear(self.max_sequence - 1, self.max_sequence - 1)
        self.final_decoder = nn.Linear(self.hidden_size * 2, 6)

    def __str__(self):
        return "Prosit"

    def forward(self, x):
        self.bi.flatten_parameters()
        self.gru.flatten_parameters()
        self.gru_decoder.flatten_parameters()

        peptides = x["sequences"]
        nce = x["normalized_collision_energy"].float()
        charge = x["charges"].float()
        B = peptides.shape[0]
        x = self.embedding(peptides)
        x = x.transpose(0, 1)
        x, _ = self.bi(x)
        x = self.drop3(x)
        x, _ = self.gru(x)
        x = self.drop3(x)
        x = self.agg(x)

        side_input = torch.cat([charge, nce], dim=1)
        side_info = self.side_encoder(side_input)
        side_info = self.drop3(side_info)

        x = x * side_info
        x = x.expand(self.max_sequence - 1, x.shape[0], x.shape[1])
        x, _ = self.gru_decoder(x)
        x = self.drop3(x)
        x_d = self.in_frag(x.transpose(0, 2))

        x = x * x_d.transpose(0, 2)
        x = self.final_decoder(x)
        x = self.leaky(x)
        x = x.transpose(0, 1).reshape(B, -1)
        return x


# pretrained model
STATE_DICT = ""


def config_state_dict(state_dict_path):
    """Config Prosit state dict path.

    Args:
        state_dict_path (str): state dict path.
    """
    global STATE_DICT
    STATE_DICT = state_dict_path
# end of config_state_dict


def pretrained_prosit():
    """Load pretrained Prosit model.

    Returns:
        PrositFrag: pretrained Prosit model.
    """
    if not STATE_DICT:
        msg = "Please config Prosit state_dict path first. "
        msg += "For default Prosit fragment model, "
        msg += "please check out https://github.com/npz7yyk/PepDream, "
        msg += "download released model state prosit.pth and "
        msg += "config the path using --spectrum-model-weight argument."
        raise ValueError(msg)

    model = Prosit()
    try:
        model.load_state_dict(torch.load(STATE_DICT))
    except RuntimeError:
        print("Prosit model is not compatible with provided state dict. ")
        exit(1)
    return model
