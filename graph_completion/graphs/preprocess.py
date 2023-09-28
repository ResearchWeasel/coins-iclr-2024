"""
Module containing methods required for graph preprocessing.
"""
from queue import Queue
from typing import Dict, Generator, Iterable, List, Set, Tuple, Union

import attr
import numpy as np
import pandas as pd
import torch as pt
from attr.validators import instance_of
from torch.nn.functional import one_hot

from graph_completion.graphs.random_walks import do_walk, obtain_context_indices
from graph_completion.utils import AbstractConf, one_or_many

Triplet = Tuple[int, int, int]
Sample = Tuple[Triplet, List[Triplet]]
SamplePair = Tuple[Sample, Sample]


def get_efficient_indexes(node_data: pd.DataFrame, edge_data: pd.DataFrame,
                          community_membership: np.ndarray) -> Iterable[pd.Series]:
    nodes_of_type_community = node_data.assign(c=community_membership).set_index(["type", "c"])["n"].sort_index()
    edge_data_full = edge_data.assign(s_type=node_data.type.values[edge_data.s],
                                      t_type=node_data.type.values[edge_data.t],
                                      c_s=community_membership[edge_data.s],
                                      c_t=community_membership[edge_data.t])
    adjacency_source_to_target = edge_data_full.set_index(["s", "r",
                                                           "s_type", "c_s", "c_t", "t_type"])["t"].sort_index()
    adjacency_target_to_source = edge_data_full.set_index(["t", "r",
                                                           "t_type", "c_t", "c_s", "s_type"])["s"].sort_index()
    community_edge_data = edge_data_full[["c_s", "r", "c_t"]].drop_duplicates()
    del edge_data_full
    adjacency_source_to_target_c = community_edge_data.set_index(["c_s", "r"])["c_t"].sort_index()
    adjacency_target_to_source_c = community_edge_data.set_index(["c_t", "r"])["c_s"].sort_index()
    return [nodes_of_type_community, adjacency_source_to_target, adjacency_target_to_source,
            adjacency_source_to_target_c, adjacency_target_to_source_c]


class TripletData:
    def __init__(self, x=None, y=None, edge_index=None, edge_attr=None, n=None, c=None,
                 edge_index_c=None, edge_attr_c=None, n_c=None, sample=None):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.n = n
        self.c = c
        self.edge_index_c = edge_index_c
        self.edge_attr_c = edge_attr_c
        self.n_c = n_c
        self.sample = sample

    def __repr__(self):
        return f"TripletData(x={list(self.x.size()) if self.x is not None else None}, " \
               f"y={list(self.y.size()) if self.y is not None else None}, " \
               f"edge_index={list(self.edge_index.size()) if self.edge_index is not None else None}, " \
               f"edge_attr={list(self.edge_attr.size()) if self.edge_attr is not None else None}, " \
               f"n={list(self.n.size()) if self.n is not None else None}, " \
               f"c={list(self.c.size()) if self.c is not None else None}, " \
               f"edge_index_c={list(self.edge_index_c.size()) if self.edge_index_c is not None else None}, " \
               f"edge_attr_c={list(self.edge_attr_c.size()) if self.edge_attr_c is not None else None}, " \
               f"n_c={list(self.n_c.size()) if self.n_c is not None else None}, " \
               f"sample={list(self.sample.size()) if self.sample is not None else None})"

    def __len__(self):
        return self.edge_index.size(1) if self.edge_index is not None else (
            self.edge_index_c.size(1) if self.edge_index_c is not None else 0
        )

    def sample_split(self, mini_batch_size: int) -> Generator["TripletData", None, None]:
        num_samples = self.sample.max().item() + 1
        for s in range(0, num_samples, mini_batch_size):
            batch_index = (s <= self.sample) & (self.sample < s + mini_batch_size)
            yield TripletData(x=self.x[batch_index.repeat_interleave(2)],
                              y=self.y[batch_index],
                              edge_index=self.edge_index[:, batch_index],
                              edge_attr=self.edge_attr[batch_index],
                              n=self.n[:, batch_index],
                              c=self.c[:, batch_index],
                              edge_index_c=self.edge_index_c[:, batch_index],
                              edge_attr_c=self.edge_attr_c[batch_index],
                              n_c=self.n_c[:, batch_index],
                              sample=self.sample[batch_index] - s)

    def community_split(self) -> Generator[Tuple[int, int, "TripletData"], None, None]:
        device = self.edge_index.device
        batch_keys, batch_splits_sizes = pt.unique_consecutive(self.c, dim=1, return_counts=True)
        if len(batch_splits_sizes) == 1:
            yield batch_keys[0], batch_keys[1], self
        else:
            curr_edge = pt.tensor(0, device=device)
            for (c_s, c_t), batch_split_size in zip(batch_keys.T, batch_splits_sizes):
                yield c_s, c_t, TripletData(edge_index=self.edge_index[:, curr_edge: curr_edge + batch_split_size],
                                            edge_attr=self.edge_attr[curr_edge: curr_edge + batch_split_size],
                                            n=self.n[:, curr_edge: curr_edge + batch_split_size])
                curr_edge += batch_split_size

    def batch_split(self, mini_batch_size: int) -> Generator["TripletData", None, None]:
        num_samples = len(self)
        for b in range(0, num_samples, mini_batch_size):
            yield TripletData(
                x=self.x[2 * b:2 * (b + mini_batch_size)] if self.x is not None else None,
                y=self.y[b:b + mini_batch_size] if self.y is not None else None,
                edge_index=self.edge_index[:, b:b + mini_batch_size] if self.edge_index is not None else None,
                edge_attr=self.edge_attr[b:b + mini_batch_size] if self.edge_attr is not None else None,
                n=self.n[:, b:b + mini_batch_size] if self.n is not None else None,
                c=self.c[:, b:b + mini_batch_size] if self.c is not None else None,
                edge_index_c=self.edge_index_c[:, b:b + mini_batch_size] if self.edge_index_c is not None else None,
                edge_attr_c=self.edge_attr_c[b:b + mini_batch_size] if self.edge_attr_c is not None else None,
                n_c=self.n_c[:, b:b + mini_batch_size] if self.n_c is not None else None,
                sample=self.sample[b:b + mini_batch_size] if self.sample is not None else None
            )

    def get_only_positive(self) -> Generator["TripletData", None, None]:
        positive_samples = TripletData(x=self.x[self.y.repeat_interleave(2)],
                                       edge_index=self.edge_index[:, self.y], edge_attr=self.edge_attr[self.y],
                                       n=self.n[:, self.y], c=self.c[:, self.y], sample=self.sample[self.y])
        return positive_samples.batch_split(1)


class TripletEmbeddingData:
    def __init__(self, c_emb: pt.Tensor, c_attr_emb: Union[pt.Tensor, None],
                 edge_emb: pt.Tensor, edge_attr_emb: Union[pt.Tensor, None],
                 y: pt.Tensor, sample: pt.Tensor):
        self.c_emb = c_emb
        self.c_attr_emb = c_attr_emb
        self.edge_emb = edge_emb
        self.edge_attr_emb = edge_attr_emb
        self.y = y
        self.sample = sample

    def __repr__(self):
        return f"TripletEmbeddingData(c_emb={list(self.c_emb.size())}, c_attr_emb={list(self.c_attr_emb.size())}, " \
               f"edge_emb={list(self.edge_emb.size())}, edge_attr_emb={list(self.edge_attr_emb.size())}, " \
               f"y={list(self.y.size())}, sample={list(self.sample.size())})"

    def __len__(self):
        return self.edge_emb.size(1)

    def sample_split_embeddings(self) -> List[Tuple[pt.Tensor, pt.Tensor]]:
        batch_size = self.sample.max().item() + 1
        sample_sort_index = pt.argsort(self.sample)
        y = self.y[sample_sort_index]
        sample_emb = []
        for emb in [self.c_emb, self.c_attr_emb, self.edge_emb, self.edge_attr_emb]:
            if emb is None:
                sample_emb.append((None, None))
                continue
            emb = emb[..., sample_sort_index, :]
            pos_emb = emb[..., y, :]
            neg_emb = pt.stack(pt.chunk(emb[..., ~y, :], batch_size, dim=-2), dim=-3)
            sample_emb.append((pos_emb, neg_emb))
        return sample_emb


def samples_to_tensors(samples: List[SamplePair],
                       node_types: np.ndarray, communities: np.ndarray, num_communities: int,
                       com_neighbours: np.ndarray, node_neighbours: np.ndarray,
                       num_node_types: int, num_relations: int, device: str) -> TripletData:
    x, y, edge_index, edge_attr, n, c, edge_index_c, edge_attr_c, n_c, sample = [], [], [], [], [], [], [], [], [], []
    for i, ((pos_triplet, neg_triplets), (pos_triplet_c, neg_triplets_c)) in enumerate(samples):
        s, r, t = pos_triplet
        c_s, c_r, c_t = pos_triplet_c
        edge = [s, t]
        edge_c = [c_s, c_t]
        x.extend(node_types[edge])
        y.append(True)
        edge_index.append(edge)
        edge_attr.append(r)
        n_i = node_neighbours[edge, r]
        n_i[0, communities[n_i[0]] != communities[t]] = s
        n_i[1, communities[n_i[1]] != communities[s]] = t
        n.append(n_i)
        c.append(communities[edge])
        edge_index_c.append(edge_c)
        edge_attr_c.append(c_r)
        n_c.append(com_neighbours[edge_c, c_r])
        sample.append(i)
        for neg_triplet, neg_triplet_c in zip(neg_triplets, neg_triplets_c):
            s, r, t = neg_triplet
            c_s, c_r, c_t = neg_triplet_c
            edge = [s, t]
            edge_c = [c_s, c_t]
            x.extend(node_types[edge])
            y.append(False)
            edge_index.append(edge)
            edge_attr.append(r)
            n_i = node_neighbours[edge, r]
            n_i[0, communities[n_i[0]] != communities[t]] = s
            n_i[1, communities[n_i[1]] != communities[s]] = t
            n.append(n_i)
            c.append(communities[edge])
            edge_index_c.append(edge_c)
            edge_attr_c.append(c_r)
            n_c.append(com_neighbours[edge_c, c_r])
            sample.append(i)
    c = pt.tensor(np.array(c), dtype=pt.long, device=device).T
    batch_sort_index = pt.argsort(num_communities * c[0] + c[1])
    x_sort_index = batch_sort_index.repeat_interleave(2) * 2
    x_sort_index[1::2] += 1

    return TripletData(x=one_hot(pt.tensor(x, dtype=pt.long, device=device)[x_sort_index], num_node_types).float(),
                       y=pt.tensor(y, device=device)[batch_sort_index],
                       edge_index=pt.tensor(edge_index, dtype=pt.long, device=device)[batch_sort_index].T,
                       edge_attr=one_hot(
                           pt.tensor(edge_attr, dtype=pt.long, device=device)[batch_sort_index],
                           num_relations).float(),
                       n=pt.tensor(np.array(n), dtype=pt.long, device=device)[batch_sort_index].permute(1, 0, 2),
                       c=c[:, batch_sort_index],
                       edge_index_c=pt.tensor(np.array(edge_index_c), dtype=pt.long, device=device)[batch_sort_index].T,
                       edge_attr_c=one_hot(pt.tensor(edge_attr_c, dtype=pt.long, device=device)[batch_sort_index],
                                           num_relations).float(),
                       n_c=pt.tensor(np.array(n_c), dtype=pt.long, device=device)[batch_sort_index].permute(1, 0, 2),
                       sample=pt.tensor(sample, dtype=pt.long, device=device)[batch_sort_index])


class Sampler:
    def __init__(self, num_negative_samples: int, num_neighbours: int,
                 random_walk_length: int, context_radius: int,
                 pagerank_importances: bool, walks_relation_specific: bool):
        self.num_negative_samples = num_negative_samples
        self.num_neighbours = num_neighbours
        self.random_walk_length = random_walk_length
        self.context_radius = context_radius
        self.pagerank_importances = pagerank_importances
        self.walks_relation_specific = walks_relation_specific

        self.sample_index = 0
        self.sample_index_c = 0
        self.sample_queue: "Queue[Sample]" = Queue()
        self.sample_queue_c: "Queue[Sample]" = Queue()
        self.smore_node_cut_cache: Dict[Tuple[int, int, int, int], Set[int]] = dict()
        self.smore_node_cut_cache_c: Dict[Tuple[int, int], Set[int]] = dict()
        self.invalid_neg_targets: pd.Series = None

    def set_neg_filtering_index(self, val_edges: pd.DataFrame, test_edges: pd.DataFrame):
        self.invalid_neg_targets = pd.concat((val_edges, test_edges), sort=True).set_index(["s", "r"])["t"].sort_index()

    def get_neg_samples_random_walks(self, positive_triplet: Triplet, walk: List[int], index: int,
                                     n_of_type_c: pd.Series, community_membership: np.ndarray, num_communities: int,
                                     for_nodes: bool) -> List[Triplet]:
        s, r, t = positive_triplet
        walk = np.array(walk, dtype=int)
        walk_context = walk[np.abs(index - np.arange(len(walk))) <= self.context_radius]
        negative_samples, negative_samples_c = [], []

        if for_nodes:
            c_s, c_t = community_membership[s], community_membership[t]
            candidate_pool = one_or_many(n_of_type_c[:, c_t]).values
            invalid_neg_targets = []
            if (s, r) in self.invalid_neg_targets.index:
                invalid_neg_targets = one_or_many(self.invalid_neg_targets[s, r]).values
            candidate_pool = candidate_pool[~np.isin(candidate_pool, invalid_neg_targets)]
        else:
            candidate_pool = np.arange(num_communities)
        candidate_pool = candidate_pool[~np.isin(candidate_pool, walk_context)]
        if len(candidate_pool) > 0:
            negative_samples = [(s, r, neg_t)
                                for neg_t in np.random.choice(candidate_pool, size=self.num_negative_samples)]
        return negative_samples

    def get_neg_samples_smore(self, positive_triplet: Triplet,
                              n_of_type_c: pd.Series, node_types: np.ndarray,
                              community_membership: np.ndarray, num_communities: int,
                              for_nodes: bool) -> List[Triplet]:
        anchor, r, answer = positive_triplet
        negative_samples = []

        if for_nodes:
            s_type, t_type, c_t = node_types[anchor], node_types[answer], community_membership[answer]
            invalid_neg_targets = []
            if (anchor, r) in self.invalid_neg_targets.index:
                invalid_neg_targets = one_or_many(self.invalid_neg_targets[anchor, r]).values
            cache_state = np.array(list(self.smore_node_cut_cache[(s_type, r, t_type, anchor)]))
            candidate_pool = one_or_many(n_of_type_c[:, c_t]).values
            candidate_pool = candidate_pool[(~np.isin(candidate_pool, invalid_neg_targets))
                                            & (~np.isin(candidate_pool, cache_state))]
        else:
            cache_state = np.array(list(self.smore_node_cut_cache_c[(r, anchor)]))
            candidate_pool = np.arange(num_communities)
            candidate_pool = candidate_pool[~np.isin(candidate_pool, cache_state)]
        if len(candidate_pool) > 0:
            negative_samples = [(anchor, r, neg_answer)
                                for neg_answer in np.random.choice(candidate_pool, size=self.num_negative_samples)]
        return negative_samples

    def get_triplets_generator_random_walks(self, num_samples: int,
                                            graph_indexes: Iterable[pd.Series],
                                            community_membership: np.ndarray,
                                            num_communities: int) -> Generator[SamplePair, None, None]:
        n_of_type_c, adj_s_to_t, _, adj_s_to_t_c, _ = graph_indexes

        for _ in range(num_samples):
            while self.sample_queue.empty():
                start_node, r, _, _, _, _ = adj_s_to_t.index[self.sample_index]
                walk_length, walk_nodes = do_walk(start_node, r, self.random_walk_length, adj_s_to_t)
                context_indices = [] if walk_length < 2 else obtain_context_indices(walk_length, self.context_radius)
                for node_index, context_node_index in context_indices:
                    positive_sample = (walk_nodes[node_index], r, walk_nodes[context_node_index])
                    negative_samples = self.get_neg_samples_random_walks(positive_sample, walk_nodes, node_index,
                                                                         n_of_type_c,
                                                                         community_membership, num_communities, True)
                    if len(negative_samples) == 0:
                        continue
                    self.sample_queue.put((positive_sample, negative_samples))

                self.sample_index = (self.sample_index + 1) % len(adj_s_to_t)

            while self.sample_queue_c.empty():
                start_comm, c_r = adj_s_to_t_c.index[self.sample_index_c]
                walk_length, walk_comms = do_walk(start_comm, c_r, self.random_walk_length, adj_s_to_t_c)
                context_indices = [] if walk_length < 2 else obtain_context_indices(walk_length, self.context_radius)
                for node_index, context_node_index in context_indices:
                    positive_sample_c = (walk_comms[node_index], c_r, walk_comms[context_node_index])
                    negative_samples_c = self.get_neg_samples_random_walks(positive_sample_c, walk_comms, node_index,
                                                                           n_of_type_c,
                                                                           community_membership, num_communities, False)
                    if len(negative_samples_c) == 0:
                        continue
                    self.sample_queue_c.put((positive_sample_c, negative_samples_c))

                self.sample_index_c = (self.sample_index_c + 1) % len(adj_s_to_t_c)

            yield self.sample_queue.get(), self.sample_queue_c.get()

    def get_triplets_generator_smore(self, num_samples: int,
                                     graph_indexes: Iterable[pd.Series],
                                     node_types: np.ndarray,
                                     community_membership: np.ndarray,
                                     num_communities: int) -> Generator[SamplePair, None, None]:
        n_of_type_c, adj_s_to_t, adj_t_to_s, adj_s_to_t_c, adj_t_to_s_c = graph_indexes

        for _ in range(num_samples):
            while self.sample_queue.empty():
                answer, r, t_type, c_t, c_s, s_type = adj_t_to_s.index[self.sample_index]
                anchor = adj_t_to_s.iloc[self.sample_index]
                answers = one_or_many(adj_s_to_t.loc[anchor, r, s_type, c_s, c_t, t_type]).values
                self.smore_node_cut_cache.setdefault((s_type, r, t_type, anchor), set())
                self.smore_node_cut_cache[(s_type, r, t_type, anchor)].update(answers)
                positive_sample = (anchor, r, answer)
                negative_samples = self.get_neg_samples_smore(positive_sample, n_of_type_c,
                                                              node_types, community_membership, num_communities, True)
                if len(negative_samples) > 0:
                    self.sample_queue.put((positive_sample, negative_samples))

                self.sample_index = (self.sample_index + 1) % len(adj_t_to_s)

            while self.sample_queue_c.empty():
                answer, r = adj_t_to_s_c.index[self.sample_index_c]
                anchor = adj_t_to_s_c.iloc[self.sample_index_c]
                answers = one_or_many(adj_s_to_t_c.loc[anchor, r]).values
                self.smore_node_cut_cache_c.setdefault((r, anchor), set())
                self.smore_node_cut_cache_c[(r, anchor)].update(answers)
                positive_sample_c = (anchor, r, answer)
                negative_samples_c = self.get_neg_samples_smore(positive_sample_c, n_of_type_c,
                                                                node_types,
                                                                community_membership, num_communities, False)
                if len(negative_samples_c) > 0:
                    self.sample_queue_c.put((positive_sample_c, negative_samples_c))

                self.sample_index_c = (self.sample_index_c + 1) % len(adj_t_to_s_c)

            yield self.sample_queue.get(), self.sample_queue_c.get()


@attr.s
class SamplerHpars(AbstractConf):
    OPTIONS = {"sampler": Sampler}
    num_negative_samples = attr.ib(default=5, validator=instance_of(int))
    num_neighbours = attr.ib(default=10, validator=instance_of(int))
    random_walk_length = attr.ib(default=10, validator=instance_of(int))
    context_radius = attr.ib(default=2, validator=instance_of(int))
    pagerank_importances = attr.ib(default=True, validator=instance_of(bool))
    walks_relation_specific = attr.ib(default=True, validator=instance_of(bool))
    name = "sampler"
