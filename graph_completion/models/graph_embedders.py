"""
Module containing the implementation of the embedding models from related work.
"""
from math import sqrt
from typing import Tuple

import attr
import pandas as pd
import torch as pt
from attr.validators import and_, ge, in_, instance_of, le
from torch import nn
from torch.nn.functional import normalize, one_hot, softmax
from torch_geometric.nn import GAT, GCNConv, MLP

from graph_completion.utils import AbstractConf


class GraphEmbedderMLP(nn.Module):
    """
    Class representing a general MLP embedder. Node embeddings are relation-specific.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 mlp_hidden_dim: int, mlp_num_hidden_layers: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = mlp_hidden_dim
        self.num_hidden_layers = mlp_num_hidden_layers

        self.entity_embeddings = MLP([num_entities + num_relations, ]
                                     + [mlp_hidden_dim, ] * mlp_num_hidden_layers + [embedding_dim, ])

    def __apply_mlp_pipeline(self, entities: pt.Tensor, edge_attr: pt.Tensor) -> pt.Tensor:
        x = pt.column_stack((entities, edge_attr))
        e = self.entity_embeddings(x)
        return e

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor) -> Tuple[pt.Tensor, None]:
        edge_index = one_hot(edge_index, self.num_entities).float()
        e_s_r = self.__apply_mlp_pipeline(edge_index[0], edge_attr)
        e_t_r = self.__apply_mlp_pipeline(edge_index[1], edge_attr)
        edge_emb = pt.stack((e_s_r, e_t_r), dim=0)
        return edge_emb, None


class GraphEmbedderDummy(nn.Module):
    """
    Class representing a TransE or DistMult embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, dummy_margin: float):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = dummy_margin

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.r_embeddings = nn.Linear(num_relations, embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        e_s, e_t = self.entity_embeddings(edge_index[0]), self.entity_embeddings(edge_index[1])
        edge_emb = pt.stack((e_s, e_t), dim=0)
        edge_attr = self.r_embeddings(edge_attr)
        edge_emb = normalize(edge_emb, dim=-1)
        return edge_emb, edge_attr


class GraphEmbedderComplEx(nn.Module):
    """
    Class representing a ComplEx embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, dummy_margin: float):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = dummy_margin

        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
        self.r_embeddings = nn.Linear(num_relations, 2 * embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        e_s, e_t = self.entity_embeddings(edge_index[0]), self.entity_embeddings(edge_index[1])
        edge_emb = pt.stack((e_s, e_t), dim=0)
        edge_attr = self.r_embeddings(edge_attr)
        return edge_emb, edge_attr


class GraphEmbedderRotatE(nn.Module):
    """
    Class representing a RotatE embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, dummy_margin: float):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = dummy_margin

        self.entity_embeddings = nn.Embedding(num_entities, 2 * embedding_dim)
        self.r_embeddings = nn.Linear(num_relations, embedding_dim, bias=False)
        nn.init.uniform_(self.entity_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)
        nn.init.uniform_(self.r_embeddings.weight,
                         -(self.margin + 2) / embedding_dim, (self.margin + 2) / embedding_dim)

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        e_s, e_t = self.entity_embeddings(edge_index[0]), self.entity_embeddings(edge_index[1])
        edge_emb = pt.stack((e_s, e_t), dim=0)
        edge_attr = self.r_embeddings(edge_attr)
        edge_attr = edge_attr / ((self.margin + 2) / (self.embedding_dim * pt.pi))
        return edge_emb, edge_attr


class GraphEmbedderGATNE(nn.Module):
    """
    Class representing a GATNE embedder. Node embeddings are relation-specific.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 gatne_edge_embedding_dim, gatne_attention_dim):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.edge_embedding_dim = gatne_edge_embedding_dim
        self.attention_dim = gatne_attention_dim

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        nn.init.uniform_(self.entity_embeddings.weight, -1.0, 1.0)
        self.u = nn.Parameter(
            pt.empty(num_entities, num_relations, gatne_edge_embedding_dim).uniform_(-1.0, 1.0)
        )
        emd_std = 1.0 / sqrt(embedding_dim)
        self.W = nn.Parameter(
            pt.fmod(pt.empty(num_relations, gatne_edge_embedding_dim, gatne_attention_dim).normal_(std=emd_std), 2)
        )
        self.w = nn.Parameter(pt.fmod(pt.empty(num_relations, gatne_attention_dim).normal_(std=emd_std), 2))
        self.M = nn.Parameter(
            pt.fmod(pt.empty(num_relations, gatne_edge_embedding_dim, embedding_dim).normal_(std=emd_std), 2)
        )

    def __apply_gatne_pipeline(self, entities: pt.Tensor, edge_attr: pt.Tensor,
                               entities_neighbors: pt.Tensor) -> pt.Tensor:
        b = self.entity_embeddings(entities)
        u_r_neighbours = self.u[entities_neighbors]
        u_r = pt.mean(u_r_neighbours, dim=1)

        W_r = pt.einsum("ir,rua->iua", edge_attr, self.W)
        w_r = edge_attr @ self.w
        M_r = pt.einsum("ir,rue->iue", edge_attr, self.M)
        attention = softmax(pt.einsum("ira,ia->ir", pt.tanh(u_r @ W_r), w_r), dim=1)

        e = b + pt.einsum("iue,iru,ir->ie", M_r, u_r, attention)
        return e

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor, n: pt.Tensor) -> Tuple[pt.Tensor, None]:
        e_s_r = self.__apply_gatne_pipeline(edge_index[0], edge_attr, n[0])
        e_t_r = self.__apply_gatne_pipeline(edge_index[1], edge_attr, n[1])
        edge_emb = pt.stack((e_s_r, e_t_r), dim=0)
        edge_emb = normalize(edge_emb, dim=-1)
        return edge_emb, None


class GraphEmbedderSACN(nn.Module):
    """
    Class representing a SACN embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, sacn_dropout_rate: float):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.dropout_rate = sacn_dropout_rate

        self.edge_data: pt.Tensor = None

        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim // 2)
        self.r_weights = nn.Embedding(num_relations, 1)
        self.gc1 = GCNConv(embedding_dim // 2, embedding_dim, normalize=False)
        self.gc1_dropout = nn.Dropout(sacn_dropout_rate)
        self.gc2 = GCNConv(embedding_dim, embedding_dim, normalize=False)
        self.gc2_dropout = nn.Dropout(sacn_dropout_rate)
        self.r_embeddings = nn.Linear(num_relations, embedding_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(embedding_dim)
        self.bn4 = nn.BatchNorm1d(embedding_dim)

    def set_edge_data(self, edge_data: pd.DataFrame, device: str):
        self.edge_data = pt.tensor(edge_data[["s", "r", "t"]].values.T, dtype=pt.long, device=device)

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        x = self.entity_embeddings(pt.arange(self.num_entities, dtype=pt.long, device=edge_index.device))
        edge_index_full = self.edge_data[[0, 2]]
        edge_weight_full = self.r_weights(self.edge_data[1]).squeeze(1)

        x = self.gc1(x=x, edge_index=edge_index_full, edge_weight=edge_weight_full)
        if x.size(0) > 1:
            x = self.bn3(x)
        x = pt.tanh(x)
        x = self.gc1_dropout(x)
        x = self.gc2(x=x, edge_index=edge_index_full, edge_weight=edge_weight_full)
        if x.size(0) > 1:
            x = self.bn4(x)
        x = pt.tanh(x)
        x = self.gc2_dropout(x)

        e_s, e_t = x[edge_index[0]], x[edge_index[1]]
        edge_emb = pt.stack((e_s, e_t), dim=0)
        edge_attr = self.r_embeddings(edge_attr)
        return edge_emb, edge_attr


class GraphEmbedderKBGAT(nn.Module):
    """
    Class representing a KBGAT embedder. Node and relation embeddings are computed separately.
    """

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, kbgat_num_hops: int,
                 kbgat_num_attention_heads: int, kbgat_attention_dim: int,
                 kbgat_negative_slope: float, kbgat_dropout_rate: float):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.num_hops = kbgat_num_hops
        self.num_attention_heads = kbgat_num_attention_heads
        self.attention_dim = kbgat_attention_dim
        self.negative_slope = kbgat_negative_slope
        self.dropout_rate = kbgat_dropout_rate

        self.edge_data: pt.Tensor = None

        self.entity_embeddings_initial = nn.Embedding(num_entities, embedding_dim)
        self.r_embeddings_initial = nn.Embedding(num_relations, embedding_dim)
        self.multi_head_gat = GAT(in_channels=embedding_dim, edge_dim=embedding_dim,
                                  hidden_channels=kbgat_num_attention_heads * kbgat_attention_dim,
                                  out_channels=kbgat_num_attention_heads * embedding_dim, concat=True,
                                  num_layers=kbgat_num_hops, heads=kbgat_num_attention_heads,
                                  act=nn.LeakyReLU(kbgat_negative_slope), dropout=kbgat_dropout_rate)
        self.entity_embeddings_skip = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.r_embeddings = nn.Linear(num_relations, embedding_dim, bias=False)

    def set_edge_data(self, edge_data: pd.DataFrame, device: str):
        self.edge_data = pt.tensor(edge_data[["s", "r", "t"]].values.T, dtype=pt.long, device=device)

    def forward(self, edge_index: pt.Tensor, edge_attr: pt.Tensor) -> Tuple[pt.Tensor, pt.Tensor]:
        x_init = self.entity_embeddings_initial(pt.arange(self.num_entities, dtype=pt.long, device=edge_index.device))
        edge_index_full = self.edge_data[[0, 2]]
        edge_attr_full = self.r_embeddings_initial(self.edge_data[1])
        x = self.multi_head_gat(x=x_init, edge_index=edge_index_full, edge_attr=edge_attr_full)
        x = pt.mean(pt.stack(pt.chunk(x, self.num_attention_heads, dim=1), dim=0), dim=0)
        x = normalize(x + self.entity_embeddings_skip(x_init), dim=1)

        e_s, e_t = x[edge_index[0]], x[edge_index[1]]
        edge_emb = pt.stack((e_s, e_t), dim=0)
        edge_attr = self.r_embeddings(edge_attr)
        return edge_emb, edge_attr


@attr.s
class GraphEmbedderHpars(AbstractConf):
    OPTIONS = {"mlp": GraphEmbedderMLP, "transe": GraphEmbedderDummy, "distmult": GraphEmbedderDummy,
               "complex": GraphEmbedderComplEx, "rotate": GraphEmbedderRotatE,
               "gatne": GraphEmbedderGATNE, "sacn": GraphEmbedderSACN, "kbgat": GraphEmbedderKBGAT}
    algorithm = attr.ib(validator=in_(list(OPTIONS.keys())))
    num_entities = attr.ib(validator=instance_of(int))
    num_relations = attr.ib(validator=instance_of(int))
    dummy_margin = attr.ib(validator=instance_of(float))
    embedding_dim = attr.ib(default=25, validator=instance_of(int))
    mlp_hidden_dim = attr.ib(default=128, validator=instance_of(int))
    mlp_num_hidden_layers = attr.ib(default=2, validator=instance_of(int))
    gatne_edge_embedding_dim = attr.ib(default=10, validator=instance_of(int))
    gatne_attention_dim = attr.ib(default=20, validator=instance_of(int))
    sacn_dropout_rate = attr.ib(default=0.2, validator=and_(instance_of(float), ge(0), le(1)))
    kbgat_num_hops = attr.ib(default=2, validator=instance_of(int))
    kbgat_num_attention_heads = attr.ib(default=2, validator=instance_of(int))
    kbgat_attention_dim = attr.ib(default=100, validator=instance_of(int))
    kbgat_negative_slope = attr.ib(default=0.2, validator=instance_of(float))
    kbgat_dropout_rate = attr.ib(default=0.3, validator=and_(instance_of(float), ge(0), le(1)))

    def __attrs_post_init__(self):
        self.name = self.algorithm
