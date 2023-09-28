"""
Module containing the implementation of the new embedding model and its loss function.
"""

from copy import deepcopy
from typing import Tuple, Union

import numpy as np
import pandas as pd
import torch as pt
from torch import nn
from torch.nn.functional import normalize, softmax

from graph_completion.graphs.preprocess import TripletData, TripletEmbeddingData
from graph_completion.models.graph_embedders import GraphEmbedderHpars
from graph_completion.models.link_rankers import LinkRankerHpars
from graph_completion.models.loss_terms import EmbeddingLossHpars


class COINs(nn.Module):
    def __init__(self, num_nodes: int, num_node_types: int, num_relations: int,
                 num_communities: int, community_sizes: np.ndarray,
                 intra_community_map: np.ndarray, inter_community_map: np.ndarray,
                 embedder_hpars: dict):
        """
        Construct the embedding model.
        """

        super().__init__()
        self.num_nodes = num_nodes
        self.num_node_types = num_node_types
        self.num_relations = num_relations
        self.num_communities = num_communities
        self.num_inter_community_nodes = int(inter_community_map.max())
        self.node_types = None
        self.community_membership = None
        self.intra_community_map = intra_community_map
        self.inter_community_map = inter_community_map
        self.embedder_hpars = GraphEmbedderHpars.from_dict(embedder_hpars)

        self.algorithm = self.embedder_hpars.algorithm
        self.embedding_dim = self.embedder_hpars.embedding_dim
        self.mlp_num_hidden_layers = self.embedder_hpars.mlp_num_hidden_layers
        self.mlp_hidden_dim = self.embedder_hpars.mlp_hidden_dim

        self.embeddings_complex = self.embedder_hpars.algorithm in ["complex", "rotate"]
        self.embeddings_relation_specific = self.embedder_hpars.algorithm in ["mlp", "gatne"]
        self.embedder_sparse = self.embedder_hpars.algorithm in ["sacn", "kbgat"]

        community_embedder_hpars = deepcopy(self.embedder_hpars)
        community_embedder_hpars.num_entities = num_communities
        self.community_embedder = community_embedder_hpars.make()

        self.node_type_embedder = nn.Linear(num_node_types,
                                            2 * self.embedding_dim if self.embeddings_complex else self.embedding_dim,
                                            bias=False)
        if self.algorithm in ["transe", "distmult", "complex", "rotate"]:
            margin = self.embedder_hpars.dummy_margin
            nn.init.uniform_(self.node_type_embedder.weight,
                             -(margin + 2) / self.embedding_dim, (margin + 2) / self.embedding_dim)
            nn.init.uniform_(self.node_type_embedder.weight,
                             -(margin + 2) / self.embedding_dim, (margin + 2) / self.embedding_dim)
        elif self.algorithm == "gatne":
            nn.init.uniform_(self.node_type_embedder.weight, -1.0, 1.0)

        self.intra_community_embedders = []
        for i in range(num_communities):
            intra_community_embedder_hpars = deepcopy(self.embedder_hpars)
            intra_community_embedder_hpars.num_entities = int(community_sizes[i])
            self.intra_community_embedders.append(intra_community_embedder_hpars.make())
        self.intra_community_embedders = nn.ModuleList(self.intra_community_embedders)

        inter_community_embedder_hpars = deepcopy(self.embedder_hpars)
        inter_community_embedder_hpars.num_entities = 1 + self.num_inter_community_nodes
        self.inter_community_embedder = inter_community_embedder_hpars.make()

        self.final_embeddings_weights = nn.Parameter(pt.ones(3))
        if not self.embeddings_relation_specific:
            self.final_embeddings_weights_r = nn.Parameter(pt.ones(2))

    def set_graph_data(self, node_types: np.ndarray, edge_data: pd.DataFrame,
                       community_membership: np.ndarray, device: str):
        if self.embedder_sparse:
            self.community_embedder.set_edge_data(
                edge_data.assign(s=community_membership[edge_data.s], t=community_membership[edge_data.t]),
                device
            )

            for i in range(self.num_communities):
                nodes_in_community = community_membership == i
                edge_data_community = edge_data[nodes_in_community[edge_data.s] & nodes_in_community[edge_data.t]]
                self.intra_community_embedders[i].set_edge_data(
                    edge_data_community.assign(s=self.intra_community_map[edge_data_community.s],
                                               t=self.intra_community_map[edge_data_community.t]),
                    device
                )

            self.inter_community_embedder.set_edge_data(
                edge_data.assign(s=self.inter_community_map[edge_data.s], t=self.inter_community_map[edge_data.t]),
                device
            )

        self.node_types = pt.tensor(node_types, dtype=pt.long, device=device)
        self.community_membership = pt.tensor(community_membership, dtype=pt.long, device=device)
        self.intra_community_map = pt.tensor(self.intra_community_map, dtype=pt.long, device=device)
        self.inter_community_map = pt.tensor(self.inter_community_map, dtype=pt.long, device=device)

    def embed_communities(self, triplet_batch: TripletData) -> Tuple[pt.Tensor, Union[pt.Tensor, None]]:
        # Community embedding
        if self.algorithm == "gatne":
            edge_emb_c, edge_attr_emb_c = self.community_embedder(triplet_batch.edge_index_c, triplet_batch.edge_attr_c,
                                                                  triplet_batch.n_c)
        else:
            edge_emb_c, edge_attr_emb_c = self.community_embedder(triplet_batch.edge_index_c, triplet_batch.edge_attr_c)
        return edge_emb_c, edge_attr_emb_c

    def forward(self, triplet_batch: TripletData) -> Tuple[pt.Tensor, Union[pt.Tensor, None]]:

        # Node type embedding
        x_emb_unstacked = self.node_type_embedder(triplet_batch.x)
        x_emb = pt.stack((x_emb_unstacked[0::2], x_emb_unstacked[1::2]), dim=0)

        # Community embedding for nodes
        if self.algorithm == "gatne":
            c_emb, c_attr_emb = self.community_embedder(triplet_batch.c, triplet_batch.edge_attr,
                                                        self.community_membership[triplet_batch.n])
        else:
            c_emb, c_attr_emb = self.community_embedder(triplet_batch.c, triplet_batch.edge_attr)

        # Node embedding
        edge_emb, edge_attr_emb = [], []
        for c_s, c_t, community_pair_batch in triplet_batch.community_split():

            if c_s == c_t:
                # Intra-community case
                node_embedder = self.intra_community_embedders[c_s]
                edge_index = self.intra_community_map[community_pair_batch.edge_index]
                n = self.intra_community_map[community_pair_batch.n]
            else:
                # Inter-community case
                node_embedder = self.inter_community_embedder
                edge_index = self.inter_community_map[community_pair_batch.edge_index]
                n = self.inter_community_map[community_pair_batch.n]

            if self.algorithm == "gatne":
                edge_emb_c_pair, edge_attr_emb_c_pair = node_embedder(edge_index, community_pair_batch.edge_attr, n)
            else:
                edge_emb_c_pair, edge_attr_emb_c_pair = node_embedder(edge_index, community_pair_batch.edge_attr)
            edge_emb.append(edge_emb_c_pair)
            if not self.embeddings_relation_specific:
                edge_attr_emb.append(edge_attr_emb_c_pair)
        edge_emb = pt.cat(edge_emb, dim=1)
        if not self.embeddings_relation_specific:
            edge_attr_emb = pt.cat(edge_attr_emb, dim=0)

        # Final node embedding refinement
        edge_emb_final = pt.stack((x_emb, c_emb, edge_emb), dim=-1)
        edge_emb_final = (edge_emb_final @ softmax(self.final_embeddings_weights, dim=0))
        edge_attr_emb_final = None
        if not self.embeddings_relation_specific:
            edge_attr_emb_final = pt.stack((c_attr_emb, edge_attr_emb), dim=-1)
            edge_attr_emb_final = (edge_attr_emb_final @ softmax(self.final_embeddings_weights_r, dim=0))

        # Normalization
        if self.algorithm in ["transe", "distmult", "gatne", "kbgat"]:
            edge_emb_final = normalize(edge_emb_final, dim=-1)

        return edge_emb_final, edge_attr_emb_final

    def embed_supervised(self, triplet_batch: TripletData) -> TripletEmbeddingData:
        c_emb, c_attr_emb = self.embed_communities(triplet_batch)
        edge_emb, edge_attr_emb = self(triplet_batch)
        return TripletEmbeddingData(c_emb=c_emb, c_attr_emb=c_attr_emb,
                                    edge_emb=edge_emb, edge_attr_emb=edge_attr_emb,
                                    y=triplet_batch.y, sample=triplet_batch.sample)


class COINsLinkPredictor(nn.Module):
    def __init__(self, link_ranker_hpars: dict):
        """
        Construct the link prediction model.
        """

        super().__init__()
        self.link_ranker_hpars = LinkRankerHpars.from_dict(link_ranker_hpars)
        self.algorithm = link_ranker_hpars["algorithm"]
        self.embeddings_relation_specific = link_ranker_hpars["algorithm"] in ["mlp", "gatne"]

        self.community_link_ranker = self.link_ranker_hpars.make()
        self.node_link_ranker = self.link_ranker_hpars.make()

    def forward(self, edge_emb: pt.Tensor, edge_attr_emb: Union[pt.Tensor, None],
                for_communities: bool = False) -> pt.Tensor:
        # Triplet scoring
        link_ranker = self.community_link_ranker if for_communities else self.node_link_ranker
        if self.embeddings_relation_specific:
            y = link_ranker(edge_emb)
        else:
            y = link_ranker(edge_emb, edge_attr_emb)

        if self.algorithm != "sacn":
            y = pt.sigmoid(y)

        return y


class COINsLoss(nn.Module):
    def __init__(self, embedding_loss_hpars: dict, coins_alpha: float):
        """
        Construct the loss function for the embedding model.
        """

        super().__init__()
        self.embedding_loss_hpars = EmbeddingLossHpars.from_dict(embedding_loss_hpars)
        self.alpha = coins_alpha
        self.algorithm = embedding_loss_hpars["algorithm"]
        self.embeddings_relation_specific = embedding_loss_hpars["algorithm"] in ["mlp", "gatne"]
        self.learnable_link_ranker = embedding_loss_hpars["algorithm"] in ["mlp", "sacn", "kbgat"]

        self.community_embedding_loss = self.embedding_loss_hpars.make()
        self.embedding_loss = self.embedding_loss_hpars.make()

    def forward(self, triplet_emb_batch: TripletEmbeddingData, link_ranker: COINsLinkPredictor):
        (c, c_neg), (c_attr, c_attr_neg), (edge, edge_neg), (edge_attr, edge_attr_neg) = \
            triplet_emb_batch.sample_split_embeddings()

        if self.embeddings_relation_specific and self.learnable_link_ranker:
            community_loss = self.community_embedding_loss(c, c_neg, link_ranker.community_link_ranker)
            node_loss = self.embedding_loss(edge, edge_neg, link_ranker.node_link_ranker)
        elif self.learnable_link_ranker:
            community_loss = self.community_embedding_loss(c, c_attr, c_neg, c_attr_neg,
                                                           link_ranker.community_link_ranker)
            node_loss = self.embedding_loss(edge, edge_attr, edge_neg, edge_attr_neg, link_ranker.node_link_ranker)
        elif self.embeddings_relation_specific:
            community_loss = self.community_embedding_loss(c, c_neg)
            node_loss = self.embedding_loss(edge, edge_neg)
        else:
            community_loss = self.community_embedding_loss(c, c_attr, c_neg, c_attr_neg)
            node_loss = self.embedding_loss(edge, edge_attr, edge_neg, edge_attr_neg)

        return (1 - self.alpha) * community_loss + self.alpha * node_loss, (community_loss, node_loss)
