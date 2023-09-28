"""
Module containing the implementation of the evaluation procedure and the experiment pipeline.
"""
import time
from copy import deepcopy
from glob import glob
from math import ceil
from os import makedirs
from os.path import dirname, realpath
from typing import Dict

import attr
import numpy as np
import torch as pt
from attr.validators import and_, ge, in_, instance_of, le
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, \
    precision_score, recall_score, roc_auc_score
from tensorboardX import SummaryWriter
from torch.cuda import device_count
from torch.nn.functional import one_hot
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from graph_completion.graphs.load_graph import Loader, LoaderHpars
from graph_completion.graphs.preprocess import TripletData, get_efficient_indexes
from graph_completion.models.coins import COINs, COINsLinkPredictor, COINsLoss
from graph_completion.models.graph_embedders import GraphEmbedderHpars
from graph_completion.models.link_rankers import LinkRankerHpars
from graph_completion.models.loss_terms import EmbeddingLossHpars
from graph_completion.utils import AbstractConf, one_or_many, reproduce


class Experiment:
    def __init__(self, seed: int, device: str, train: bool, checkpoint_run: int, checkpoint_tag: str,
                 val_size: float, test_size: float, mini_batch_size: int,
                 lr: float, weight_decay: float, val_patience: int, val_tolerance: float,
                 max_samples: int, validation_freq: int, checkpoint_freq: int,
                 algorithm: str, leiden_resolution: float, coins_alpha: float, loader_hpars: dict,
                 embedder_hpars: dict, link_ranker_hpars: dict, embedding_loss_hpars: dict):
        self.seed = seed
        self.device = device
        self.train = train
        self.checkpoint_run = checkpoint_run
        self.checkpoint_tag = checkpoint_tag
        self.val_size = val_size
        self.test_size = test_size
        self.mini_batch_size = mini_batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_patience = val_patience
        self.val_tolerance = val_tolerance
        self.max_samples = max_samples
        self.validation_freq = validation_freq
        self.checkpoint_freq = checkpoint_freq
        self.algorithm = algorithm
        self.leiden_resolution = leiden_resolution
        self.coins_alpha = coins_alpha
        self.loader_hpars = loader_hpars
        self.embedder_hpars = embedder_hpars
        self.link_ranker_hpars = link_ranker_hpars
        self.embedding_loss_hpars = embedding_loss_hpars

        self.loader: Loader = None
        self.val_set: TripletData = None
        self.test_set: TripletData = None
        self.embedder: COINs = None
        self.link_ranker: COINsLinkPredictor = None
        self.criterion: COINsLoss = None
        self.embedder_optim: Optimizer = None
        self.link_ranker_optim: Optimizer = None

        self.run_id: int = 0
        self.dashboard: SummaryWriter = None

    def prepare(self):
        reproduce(self.seed)

        # Load data
        self.loader = LoaderHpars.from_dict(self.loader_hpars).make()
        self.loader.load_graph(self.seed, self.device, self.val_size, self.test_size, self.leiden_resolution)
        self.val_set = self.loader.get_evaluation_triplets(val=True)
        self.test_set = self.loader.get_evaluation_triplets(val=False)

        # Initialize models
        print("Constructing embedder...")
        self.embedder_hpars.update(num_entities=self.loader.num_nodes, num_relations=self.loader.num_relations)
        self.embedder_hpars.update(dummy_margin=self.embedding_loss_hpars["margin"])
        self.embedder = COINs(self.loader.num_nodes, self.loader.num_node_types, self.loader.num_nodes,
                              self.loader.num_communities, self.loader.community_sizes,
                              self.loader.intra_community_map, self.loader.inter_community_map,
                              self.embedder_hpars).to(self.device)
        self.embedder.set_graph_data(self.loader.dataset.node_data.type.values, self.loader.train_edge_data,
                                     self.loader.communities, self.device)
        self.embedder_optim = Adam(self.embedder.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print("Constructing link ranker and loss...")
        self.link_ranker_hpars.update(embedding_dim=self.embedder.embedding_dim)
        if self.algorithm in ["transe", "rotate"]:
            self.link_ranker_hpars.update(dummy_margin=self.embedding_loss_hpars["margin"])
        self.link_ranker = COINsLinkPredictor(self.link_ranker_hpars).to(self.device)
        self.criterion = COINsLoss(self.embedding_loss_hpars, self.coins_alpha).to(self.device)
        if self.criterion.learnable_link_ranker:
            self.link_ranker_optim = Adam(self.link_ranker.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Logging
        makedirs(f"graph_completion/results/{self.loader.dataset_name}/runs", exist_ok=True)
        self.run_id = len(glob(f"graph_completion/results/{self.loader.dataset_name}/runs/*")) + 1
        makedirs(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}", exist_ok=True)
        self.dashboard = SummaryWriter(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}")

    def run_training(self):
        makedirs(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/checkpoints", exist_ok=True)
        if self.train and self.checkpoint_run > 0 and len(self.checkpoint_tag) > 0:
            checkpoint = pt.load(f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                 f"{self.checkpoint_run}/checkpoints/checkpoint_{self.checkpoint_tag}.tar",
                                 map_location=self.device)
            start_time = checkpoint["start_time"]
            prev_train_loss, best_val_loss = checkpoint["prev_train_loss"], checkpoint["best_val_loss"]
            num_samples_processed = checkpoint["num_samples_processed"]
            num_batches_processed = checkpoint["num_batches_processed"]
            patience = checkpoint["patience"]
            self.embedder.load_state_dict(checkpoint["embedder_state_dict"])
            self.link_ranker.load_state_dict(checkpoint["link_ranker_state_dict"])
            self.embedder_optim.load_state_dict(checkpoint["embedder_optim_state_dict"])
            if self.criterion.learnable_link_ranker:
                self.link_ranker_optim.load_state_dict(checkpoint["link_ranker_optim_state_dict"])
        else:
            start_time = time.time()
            prev_train_loss, best_val_loss = 0, None
            num_samples_processed = 0
            num_batches_processed = 0
            patience = 0
        best_embedder, best_link_ranker = deepcopy(self.embedder.state_dict()), deepcopy(self.link_ranker.state_dict())

        print("Training...")
        while self.train and patience < self.val_patience and num_samples_processed < self.max_samples:
            self.embedder.train()
            self.embedder_optim.zero_grad()
            if self.criterion.learnable_link_ranker:
                self.link_ranker.train()
                self.link_ranker_optim.zero_grad()

            train_batch = self.loader.get_training_triplets(self.mini_batch_size)
            loss, (com_loss, node_loss) = self.criterion(self.embedder.embed_supervised(train_batch), self.link_ranker)
            loss.backward()
            self.embedder_optim.step()
            if self.criterion.learnable_link_ranker:
                self.link_ranker_optim.step()
            self.dashboard.add_scalar("Train/Loss", loss.item(), num_samples_processed)
            self.dashboard.add_scalar("Train/ComLoss", com_loss.item(), num_samples_processed)
            self.dashboard.add_scalar("Train/NodeLoss", node_loss.item(), num_samples_processed)

            if num_batches_processed % self.validation_freq == 0:
                self.embedder.eval()
                if self.criterion.learnable_link_ranker:
                    self.link_ranker.eval()
                val_metrics = self.compute_evaluation_metrics(self.val_set)
                for metric_name, metric_value in val_metrics.items():
                    self.dashboard.add_scalar(f"Validation/{metric_name}", metric_value, num_samples_processed)
                val_com_ap, val_ap = val_metrics["ComAP"], val_metrics["AP"]
                val_loss = 1 - 0.5 * (val_com_ap + val_ap)
                train_loss_change = abs(loss.item() - prev_train_loss)
                if (best_val_loss is not None and val_loss >= best_val_loss) and train_loss_change < self.val_tolerance:
                    patience += 1
                else:
                    patience = 0
                    if best_val_loss is None or val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_embedder = deepcopy(self.embedder.state_dict())
                        best_link_ranker = deepcopy(self.link_ranker.state_dict())
                prev_train_loss = loss.item()
                self.dashboard.add_scalar("Validation/Patience", patience, num_samples_processed)
                if num_batches_processed % self.checkpoint_freq == 0:
                    checkpoint = {"start_time": start_time,
                                  "prev_train_loss": prev_train_loss,
                                  "best_val_loss": best_val_loss,
                                  "num_samples_processed": num_samples_processed,
                                  "num_batches_processed": num_batches_processed,
                                  "patience": patience,
                                  "embedder_state_dict": self.embedder.state_dict(),
                                  "link_ranker_state_dict": self.link_ranker.state_dict(),
                                  "embedder_optim_state_dict": self.embedder_optim.state_dict()}
                    if self.criterion.learnable_link_ranker:
                        checkpoint.update(link_ranker_optim_state_dict=self.link_ranker_optim.state_dict())
                    pt.save(checkpoint, f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                        f"{self.run_id}/checkpoints/checkpoint_{num_batches_processed}.tar")
                self.dashboard.add_scalar("Train/Time", time.time() - start_time, num_samples_processed)

                with open(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/train_log.txt",
                          mode="a+", encoding="utf-8") as train_log:
                    hparam_dict = {
                        "MiniBatchSize": self.mini_batch_size, "LearningRate": self.lr,
                        "Algorithm": self.algorithm, "LeidenResolution": self.leiden_resolution,
                        "EmbeddingDim": self.embedder.embedding_dim,
                        "LossMargin": self.criterion.embedding_loss_hpars.margin,
                        "Dataset": self.loader.dataset_name, "NumNodes": self.loader.num_nodes,
                        "NumNodeTypes": self.loader.num_node_types,
                        "NumRelations": self.loader.num_relations,
                        "NumCommunities": self.loader.num_communities,
                        "NumNegativeSamples": self.loader.sampler.num_negative_samples
                    }
                    hparam_dict.update(**self.loader.graph_analysis_metrics)
                    train_line = "\t".join([str(v) for _, v in hparam_dict.items()])
                    train_line += f"\t{com_loss.item()}\t{node_loss.item()}\t{loss.item()}"
                    train_line += f"\t{time.time() - start_time}\t"
                    train_line += "\t".join([str(v) for _, v in val_metrics.items()])
                    train_line += f"\t{patience}\n"
                    train_log.write(train_line)

            num_samples_processed += len(train_batch.y)
            num_batches_processed += 1
            pt.cuda.empty_cache()

        if self.train:
            checkpoint = {"embedder_state_dict": best_embedder,
                          "link_ranker_state_dict": best_link_ranker}
            pt.save(checkpoint, f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                f"{self.run_id}/checkpoints/checkpoint_best.tar")
        else:
            best_checkpoint = pt.load(f"graph_completion/results/{self.loader.dataset_name}/runs/"
                                      f"{self.checkpoint_run}/checkpoints/checkpoint_{self.checkpoint_tag}.tar",
                                      map_location=self.device)
            best_embedder = best_checkpoint["embedder_state_dict"]
            best_link_ranker = best_checkpoint["link_ranker_state_dict"]
        self.embedder.load_state_dict(best_embedder)
        self.link_ranker.load_state_dict(best_link_ranker)

    def compute_evaluation_metrics(self, evaluation_triplets: TripletData,
                                   query_answering: bool = False) -> Dict[str, float]:
        metrics = {}

        # Link prediction metrics
        with pt.no_grad():
            num_batches = ceil((evaluation_triplets.sample.max().item() + 1) / self.mini_batch_size)
            loss, com_loss, node_loss = [], [], []
            for evaluation_batch in tqdm(evaluation_triplets.sample_split(self.mini_batch_size),
                                         "Computing loss", total=num_batches, leave=False):
                loss_batch, (com_loss_batch, node_loss_batch) = self.criterion(
                    self.embedder.embed_supervised(evaluation_batch), self.link_ranker
                )
                loss.append(loss_batch * len(evaluation_batch) / len(evaluation_triplets))
                com_loss.append(com_loss_batch * len(evaluation_batch) / len(evaluation_triplets))
                node_loss.append(node_loss_batch * len(evaluation_batch) / len(evaluation_triplets))
            metrics["ComLoss"] = pt.stack(com_loss).sum().item()
            metrics["NodeLoss"] = pt.stack(node_loss).sum().item()
            metrics["Loss"] = pt.stack(loss).sum().item()

            y_pred_c = []
            y_pred = []
            num_batches = ceil(len(evaluation_triplets) / self.mini_batch_size)
            for evaluation_batch in tqdm(evaluation_triplets.batch_split(self.mini_batch_size),
                                         "Classifying", total=num_batches, leave=False):
                y_pred_c.append(self.link_ranker(*self.embedder.embed_communities(evaluation_batch),
                                                 for_communities=True))
                y_pred.append(self.link_ranker(*self.embedder(evaluation_batch)))
            y_pred_c, y_pred = pt.cat(y_pred_c).cpu().numpy(), pt.cat(y_pred).cpu().numpy()
            y = evaluation_triplets.y.cpu().numpy()
            metrics["ComAccuracy"] = accuracy_score(y, y_pred_c > 0.5)
            metrics["Accuracy"] = accuracy_score(y, y_pred > 0.5)
            metrics["ComPrecision"] = precision_score(y, y_pred_c > 0.5, average="weighted", zero_division=0)
            metrics["Precision"] = precision_score(y, y_pred > 0.5, average="weighted", zero_division=0)
            metrics["ComRecall"] = recall_score(y, y_pred_c > 0.5, average="weighted", zero_division=0)
            metrics["Recall"] = recall_score(y, y_pred > 0.5, average="weighted", zero_division=0)
            metrics["ComF1"] = f1_score(y, y_pred_c > 0.5, average="weighted", zero_division=0)
            metrics["F1"] = f1_score(y, y_pred > 0.5, average="weighted", zero_division=0)
            metrics["ComROC-AUC"] = roc_auc_score(y, y_pred_c, average="weighted")
            metrics["ROC-AUC"] = roc_auc_score(y, y_pred, average="weighted")
            metrics["ComAP"] = average_precision_score(y, y_pred_c, average="weighted")
            metrics["AP"] = average_precision_score(y, y_pred, average="weighted")

            if query_answering:
                # Query answering metrics
                _, adj_s_to_t, _, adj_s_to_t_c, _ = get_efficient_indexes(self.loader.dataset.node_data,
                                                                          self.loader.dataset.edge_data,
                                                                          self.loader.communities)
                ranks_c = []
                ranks_n = []
                ranks = []
                all_communities = pt.arange(self.loader.num_communities, device=self.device)
                for pos_triplet in tqdm(evaluation_triplets.get_only_positive(), "Query answering",
                                        total=pt.sum(evaluation_triplets.y).item(), leave=False):
                    s, t = pos_triplet.edge_index[0], pos_triplet.edge_index[1]
                    c_s, c_t = pos_triplet.c[0], pos_triplet.c[1]
                    r = pos_triplet.edge_attr.squeeze().argmax()
                    s_cpu, r_cpu, t_cpu, c_s_cpu, c_t_cpu = s.item(), r.item(), t.item(), c_s.item(), c_t.item()

                    # Community ranking
                    scores_c = []
                    all_answers_c = pt.tensor(
                        one_or_many(adj_s_to_t_c.loc[c_s_cpu, r_cpu]).values,
                        dtype=pt.long, device=self.device)
                    filtered_communities = all_communities[(all_communities == c_t)
                                                           | (~pt.isin(all_communities, all_answers_c))]
                    edge_index_c = pt.stack((c_s.expand(self.loader.num_communities), all_communities))
                    edge_attr_c = pos_triplet.edge_attr.expand(self.loader.num_communities, self.loader.num_relations)
                    n_c = pt.tensor(self.loader.com_neighbours[:, r_cpu], dtype=pt.long, device=self.device)
                    n_c = n_c[edge_index_c]
                    all_triplets = TripletData(edge_index_c=edge_index_c, edge_attr_c=edge_attr_c, n_c=n_c)

                    num_batches = ceil(len(all_triplets) / self.mini_batch_size)
                    for evaluation_batch in tqdm(all_triplets.batch_split(self.mini_batch_size),
                                                 "Scoring communities", total=num_batches, leave=False):
                        scores_c.append(self.link_ranker(*self.embedder.embed_communities(evaluation_batch),
                                                         for_communities=True))
                    scores_c = pt.cat(scores_c)
                    scores_c[filtered_communities] += 1
                    rank_c_all = pt.sum(scores_c.unsqueeze(0) <= scores_c.unsqueeze(1), dim=0)
                    rank_c = rank_c_all[c_t].item()
                    ranks_c.append(rank_c)

                    # Node ranking
                    scores = []
                    nodes_in_community = (self.embedder.community_membership == c_t).nonzero().squeeze(dim=1)
                    all_answers = pt.tensor(one_or_many(adj_s_to_t.loc[s_cpu, r_cpu]).values,
                                            dtype=pt.long, device=self.device)
                    filtered_nodes = nodes_in_community[(nodes_in_community == t)
                                                        | ((~pt.isin(nodes_in_community, all_answers))
                                                           & (nodes_in_community != s))]
                    community_size_filtered = len(filtered_nodes)
                    x = pt.stack((pos_triplet.x[0].expand(community_size_filtered, self.loader.num_node_types),
                                  one_hot(self.embedder.node_types[filtered_nodes], self.loader.num_node_types)),
                                 dim=2).view(-1, self.loader.num_node_types)
                    edge_index = pt.stack((s.expand(community_size_filtered), filtered_nodes))
                    edge_attr = pos_triplet.edge_attr.expand(community_size_filtered, self.loader.num_relations)
                    n = pt.tensor(self.loader.node_neighbours[:, r_cpu], dtype=pt.long, device=self.device)
                    n = n[edge_index]
                    n[0, self.embedder.community_membership[n[0]] != c_t] = s
                    n[1, self.embedder.community_membership[n[1]] != c_s] = t
                    c = pos_triplet.c.expand(2, community_size_filtered)
                    all_triplets = TripletData(x=x, edge_index=edge_index, edge_attr=edge_attr, n=n, c=c)

                    num_batches = ceil(len(all_triplets) / self.mini_batch_size)
                    for evaluation_batch in tqdm(all_triplets.batch_split(self.mini_batch_size),
                                                 "Scoring nodes", total=num_batches, leave=False):
                        scores.append(self.link_ranker(*self.embedder(evaluation_batch)))
                    scores = pt.cat(scores)
                    intra_community_rank = pt.sum(scores[filtered_nodes == t] <= scores).item()
                    ranks_n.append(intra_community_rank)

                    c_err = self.loader.community_sizes[(rank_c_all < rank_c).cpu().numpy()].sum()
                    rank = c_err + intra_community_rank
                    ranks.append(rank)

                ranks_c = np.array(ranks_c)
                ranks_n = np.array(ranks_n)
                ranks = np.array(ranks)
                metrics["ComHits@1"] = np.mean(ranks_c <= 1)
                metrics["NodeHits@1"] = np.mean(ranks_n <= 1)
                metrics["Hits@1"] = np.mean(ranks <= 1)
                metrics["ComHits@3"] = np.mean(ranks_c <= 3)
                metrics["NodeHits@3"] = np.mean(ranks_n <= 3)
                metrics["Hits@3"] = np.mean(ranks <= 3)
                metrics["ComHits@10"] = np.mean(ranks_c <= 10)
                metrics["NodeHits@10"] = np.mean(ranks_n <= 10)
                metrics["Hits@10"] = np.mean(ranks <= 10)
                metrics["ComMR"] = np.mean(ranks_c)
                metrics["NodeMR"] = np.mean(ranks_n)
                metrics["MR"] = np.mean(ranks)
                metrics["ComMRR"] = np.mean(1 / ranks_c)
                metrics["NodeMRR"] = np.mean(1 / ranks_n)
                metrics["MRR"] = np.mean(1 / ranks)

        return metrics

    def main(self):
        self.prepare()

        self.run_training()

        print("Testing...")
        self.embedder.eval()
        if self.criterion.learnable_link_ranker:
            self.link_ranker.eval()
        for checkpoint_file in glob(f"graph_completion/results/{self.loader.dataset_name}/"
                                    f"runs/{self.run_id if self.train else self.checkpoint_run}/"
                                    f"checkpoints/checkpoint_*.tar"):
            checkpoint_tag = checkpoint_file[:-4].split("/")[-1].split("_")[-1]
            checkpoint = pt.load(checkpoint_file, map_location=self.device)
            embedder = checkpoint["embedder_state_dict"]
            link_ranker = checkpoint["link_ranker_state_dict"]
            self.embedder.load_state_dict(embedder)
            self.link_ranker.load_state_dict(link_ranker)
            start_time = time.time()
            test_metrics = self.compute_evaluation_metrics(self.test_set, query_answering=True)
            test_metrics.update(Time=time.time() - start_time)
            if checkpoint_tag == "best":
                for metric_name, metric_value in test_metrics.items():
                    self.dashboard.add_scalar(f"Test/{metric_name}", metric_value, self.loader.num_nodes)
            hparam_dict = {
                "MiniBatchSize": self.mini_batch_size, "LearningRate": self.lr,
                "Algorithm": self.algorithm, "LeidenResolution": self.leiden_resolution,
                "EmbeddingDim": self.embedder.embedding_dim,
                "LossMargin": self.criterion.embedding_loss_hpars.margin,
                "Dataset": self.loader.dataset_name, "NumNodes": self.loader.num_nodes,
                "NumNodeTypes": self.loader.num_node_types,
                "NumRelations": self.loader.num_relations,
                "NumCommunities": self.loader.num_communities,
                "NumNegativeSamples": self.loader.sampler.num_negative_samples
            }
            hparam_dict.update(**self.loader.graph_analysis_metrics)
            if checkpoint_tag == "best":
                self.dashboard.add_hparams(hparam_dict=hparam_dict, metric_dict=test_metrics,
                                           name=f"{dirname(realpath(__file__))}/results/"
                                                f"{self.loader.dataset_name}/runs/{self.run_id}")
            with open(f"graph_completion/results/{self.loader.dataset_name}/runs/{self.run_id}/test_log.txt",
                      mode="a+", encoding="utf-8") as test_log:
                test_line = f"{checkpoint_tag}\t"
                test_line += "\t".join([str(v) for _, v in hparam_dict.items()])
                test_line += "\t" + "\t".join([str(v) for _, v in test_metrics.items()]) + "\n"
                test_log.write(test_line)


@attr.s
class ExperimentHpars(AbstractConf):
    OPTIONS = {"experiment": Experiment}
    seed = attr.ib(default=123456789, validator=instance_of(int))
    device = attr.ib(default="cpu", validator=in_(["cpu", ] + [f"cuda:{i}" for i in range(device_count())]))
    train = attr.ib(default=True, validator=instance_of(bool))
    checkpoint_run = attr.ib(default=0, validator=instance_of(int))
    checkpoint_tag = attr.ib(default="", validator=instance_of(str))
    val_size = attr.ib(default=0.01, validator=and_(instance_of(float), ge(0), le(1)))
    test_size = attr.ib(default=0.02, validator=and_(instance_of(float), ge(0), le(1)))
    mini_batch_size = attr.ib(default=25, validator=instance_of(int))
    lr = attr.ib(default=1e-3, validator=instance_of(float))
    weight_decay = attr.ib(default=1e-6, validator=instance_of(float))
    val_patience = attr.ib(default=50, validator=instance_of(int))
    val_tolerance = attr.ib(default=1e-4, validator=instance_of(float))
    max_samples = attr.ib(default=int(2e8), validator=instance_of(int))
    validation_freq = attr.ib(default=1, validator=instance_of(int))
    checkpoint_freq = attr.ib(default=1000, validator=instance_of(int))
    algorithm = attr.ib(default="transe", validator=in_(["mlp", "transe", "distmult", "complex", "rotate",
                                                         "gatne", "sacn", "kbgat"]))
    leiden_resolution = attr.ib(default=0.0, validator=instance_of(float))
    coins_alpha = attr.ib(default=0.5, validator=instance_of(float))
    loader_hpars = attr.ib(factory=LoaderHpars, validator=lambda i, a, v: type(v) is LoaderHpars)
    embedder_hpars = attr.ib(factory=GraphEmbedderHpars, validator=lambda i, a, v: type(v) is GraphEmbedderHpars)
    link_ranker_hpars = attr.ib(factory=LinkRankerHpars, validator=lambda i, a, v: type(v) is LinkRankerHpars)
    embedding_loss_hpars = attr.ib(factory=EmbeddingLossHpars, validator=lambda i, a, v: type(v) is EmbeddingLossHpars)
    name = "experiment"

    def __attrs_post_init__(self):
        self.embedder_hpars["algorithm"] = self.algorithm
        self.link_ranker_hpars["algorithm"] = self.algorithm
        self.embedding_loss_hpars["algorithm"] = self.algorithm
