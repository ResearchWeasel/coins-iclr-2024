"""
Module containing the implementation of the context extraction process.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd

from graph_completion.utils import one_or_many


def do_walk(start: int, walk_relation: int, max_walk_length: int, adj_s_to_t: pd.Series) -> Tuple[int, List[int]]:
    curr_node = start
    walk_length = 1
    walk_nodes = [curr_node, ]
    while walk_length < max_walk_length:
        if (curr_node, walk_relation) in adj_s_to_t.index:
            curr_out_neighborhood = one_or_many(adj_s_to_t.loc[curr_node, walk_relation]).values
        else:
            curr_out_neighborhood = np.empty(0, dtype=int)

        num_neighbours = len(curr_out_neighborhood)
        if num_neighbours == 0:
            break
        sampled_index = np.random.choice(num_neighbours, size=1)[0]
        curr_node = curr_out_neighborhood[sampled_index]
        walk_nodes.append(curr_node)
        walk_length += 1
    return walk_length, walk_nodes


def obtain_context_indices(walk_length: int, context_radius: int) -> List[Tuple[int, int]]:
    """
    Calculate the position index pairs for every possible node-context pair in a random walk

    :param walk_length: length of the random walk
    :param context_radius: context size in one direction

    :return: list of position index pairs
    """

    node_context_indices = [(i, max(i - context_radius, 0), min(i + context_radius, walk_length - 1) + 1)
                            for i in range(walk_length)]
    context_triplet_indices = [(node_index, context_index)
                               for node_index, context_start, context_end in node_context_indices
                               for context_index in range(context_start, context_end) if node_index != context_index]
    return context_triplet_indices
