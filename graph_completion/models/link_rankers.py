"""
Module containing the implementation of the link prediction models from related work.
"""
import attr
import torch as pt
from attr.validators import and_, ge, in_, instance_of, le
from torch import nn
from torch.linalg import norm
from torch.nn.functional import relu
from torch_geometric.nn import MLP

from graph_completion.utils import AbstractConf


def distance_TransE(e_s: pt.Tensor, e_r: pt.Tensor, e_t: pt.Tensor, p: int = 2) -> pt.Tensor:
    return norm(e_s + e_r - e_t, ord=p, dim=-1)


def score_DistMult(e_s: pt.Tensor, e_r: pt.Tensor, e_t: pt.Tensor) -> pt.Tensor:
    return pt.sum((e_s * e_r) * e_t, dim=-1)


def score_ComplEx(e_s: pt.Tensor, e_r: pt.Tensor, e_t: pt.Tensor) -> pt.Tensor:
    e_s_real, e_s_imag = pt.chunk(e_s, 2, dim=-1)
    e_r_real, e_r_imag = pt.chunk(e_r, 2, dim=-1)
    e_t_real, e_t_imag = pt.chunk(e_t, 2, dim=-1)

    e_sr_real = e_s_real * e_r_real - e_s_imag * e_r_imag
    e_sr_imag = e_s_real * e_r_imag + e_s_imag * e_r_real

    score_real = e_sr_real * e_t_real + e_sr_imag * e_t_imag
    return pt.sum(score_real, dim=-1)


def distance_RotatE(e_s: pt.Tensor, e_r: pt.Tensor, e_t: pt.Tensor) -> pt.Tensor:
    e_s_real, e_s_imag = pt.chunk(e_s, 2, dim=-1)
    e_r_real, e_r_imag = pt.cos(e_r), pt.sin(e_r)
    e_t_real, e_t_imag = pt.chunk(e_t, 2, dim=-1)

    diff_real = e_s_real * e_r_real - e_s_imag * e_r_imag - e_t_real
    diff_imag = e_s_real * e_r_imag + e_s_imag * e_r_real - e_t_imag
    diff_abs = norm(pt.stack([diff_real, diff_imag], dim=-1), dim=-1)

    return norm(diff_abs, ord=1, dim=-1)


def distance_Context(e: pt.Tensor, e_ctx: pt.Tensor) -> pt.Tensor:
    return -pt.einsum("...i,...i", e, e_ctx)


class LinkPredictorMLP(nn.Module):
    """
    Class representing a general MLP edge scorer.
    """

    def __init__(self, embedding_dim: int, mlp_hidden_dim: int, mlp_num_hidden_layers: int):
        super().__init__()
        self.hidden_dim = mlp_hidden_dim
        self.num_hidden_layers = mlp_num_hidden_layers

        self.mlp = MLP([2 * embedding_dim, ] + [mlp_hidden_dim, ] * mlp_num_hidden_layers + [1, ])

    def forward(self, edge_emb: pt.Tensor) -> pt.Tensor:
        e, e_ctx = edge_emb[0], edge_emb[1]
        return self.mlp(pt.column_stack((e, e_ctx))).squeeze(1)


class LinkPredictorTransE(nn.Module):
    """
    Class implementing the TransE scoring function.
    """

    def __init__(self, dummy_margin: float):
        super().__init__()
        self.margin = dummy_margin

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        return self.margin - distance_TransE(edge_emb[0], edge_attr, edge_emb[1])


class LinkPredictorDistMult(nn.Module):
    """
    Class implementing the DistMult scoring function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        return score_DistMult(edge_emb[0], edge_attr, edge_emb[1])


class LinkPredictorComplex(nn.Module):
    """
    Class implementing the ComplEx scoring function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        return score_ComplEx(edge_emb[0], edge_attr, edge_emb[1])


class LinkPredictorRotatE(nn.Module):
    """
    Class implementing the RotatE scoring function.
    """

    def __init__(self, dummy_margin: float):
        super().__init__()
        self.margin = dummy_margin

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        return self.margin - distance_RotatE(edge_emb[0], edge_attr, edge_emb[1])


class LinkPredictorGATNE(nn.Module):
    """
    Class implementing the GATNE scoring function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, edge_emb: pt.Tensor) -> pt.Tensor:
        return -distance_Context(edge_emb[0], edge_emb[1])


class LinkPredictorSACN(nn.Module):
    """
    Class representing the SACN link scoring model.
    """

    def __init__(self, embedding_dim: int,
                 sacn_num_conv_channels: int, sacn_conv_kernel_size: int, sacn_dropout_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_conv_channels = sacn_num_conv_channels
        self.conv_kernel_size = sacn_conv_kernel_size
        self.dropout_rate = sacn_dropout_rate

        self.inp_drop = nn.Dropout(sacn_dropout_rate)
        self.hidden_drop = nn.Dropout(sacn_dropout_rate)
        self.feature_map_drop = nn.Dropout(sacn_dropout_rate)
        self.conv1 = nn.Conv1d(2, sacn_num_conv_channels, sacn_conv_kernel_size,
                               stride=1, padding=sacn_conv_kernel_size // 2)
        self.bn0 = nn.BatchNorm1d(2)
        self.bn1 = nn.BatchNorm1d(sacn_num_conv_channels)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.fc = nn.Linear(embedding_dim * sacn_num_conv_channels, embedding_dim)

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        e_s, e_t = edge_emb[0], edge_emb[1]

        stacked_inputs = pt.stack((e_s, edge_attr), dim=1)
        if stacked_inputs.size(0) > 1:
            stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        if x.size(0) > 1:
            x = self.bn1(x)
        x = relu(x)
        x = self.feature_map_drop(x)
        x = x.view(-1, self.num_conv_channels * self.embedding_dim)
        x = self.fc(x)
        x = self.hidden_drop(x)
        if x.size(0) > 1:
            x = self.bn2(x)
        x = relu(x)
        y = pt.sigmoid(pt.sum(x * e_t, dim=1))
        return y


class LinkPredictorKBGAT(nn.Module):
    """
    Class representing the KBGAT link scoring model.
    """

    def __init__(self, embedding_dim: int, kbgat_num_conv_channels: int, kbgat_dropout_rate: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_conv_channels = kbgat_num_conv_channels
        self.dropout_rate = kbgat_dropout_rate

        self.conv_layer = nn.Conv2d(1, kbgat_num_conv_channels, (1, 3))
        self.dropout = nn.Dropout(kbgat_dropout_rate)
        self.fc_layer = nn.Linear(embedding_dim * kbgat_num_conv_channels, 1)

    def forward(self, edge_emb: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        x = pt.stack((edge_emb[0], edge_attr, edge_emb[1]), dim=2).unsqueeze(1)
        x = self.dropout(relu(self.conv_layer(x)))
        x = x.view(-1, self.num_conv_channels * self.embedding_dim)
        y = self.fc_layer(x).squeeze(1)
        return y


@attr.s
class LinkRankerHpars(AbstractConf):
    OPTIONS = {"mlp": LinkPredictorMLP, "transe": LinkPredictorTransE, "distmult": LinkPredictorDistMult,
               "complex": LinkPredictorComplex, "rotate": LinkPredictorRotatE,
               "gatne": LinkPredictorGATNE, "sacn": LinkPredictorSACN, "kbgat": LinkPredictorKBGAT}
    algorithm = attr.ib(validator=in_(list(OPTIONS.keys())))
    embedding_dim = attr.ib(validator=instance_of(int))
    mlp_hidden_dim = attr.ib(default=128, validator=instance_of(int))
    mlp_num_hidden_layers = attr.ib(default=2, validator=instance_of(int))
    dummy_margin = attr.ib(default=1.0, validator=instance_of(float))
    sacn_num_conv_channels = attr.ib(default=100, validator=instance_of(int))
    sacn_conv_kernel_size = attr.ib(default=5, validator=instance_of(int))
    sacn_dropout_rate = attr.ib(default=0.2, validator=and_(instance_of(float), ge(0), le(1)))
    kbgat_num_conv_channels = attr.ib(default=500, validator=instance_of(int))
    kbgat_dropout_rate = attr.ib(default=0.0, validator=and_(instance_of(float), ge(0), le(1)))

    def __attrs_post_init__(self):
        self.name = self.algorithm
