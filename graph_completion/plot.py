from glob import glob
from os.path import exists
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from graph_analysis.metrics import Subgraphing
from graph_completion.experiments import ExperimentHpars
from graph_completion.graphs.load_graph import LoaderHpars

sns.set(context="paper", font_scale=5, style="darkgrid")


def parse_baseline_metrics(dataset: str, algorithm: str) -> Tuple[float, float, float, float, float]:
    with open(f"graph_completion/results_baseline/{algorithm}_{dataset}_0/train.log",
              mode="r", encoding="utf-8") as baseline_log_file:
        baseline_log = baseline_log_file.read().splitlines()
        baseline_hits_at_1 = float(baseline_log[-3].split(" ")[-1])
        baseline_hits_at_3 = float(baseline_log[-2].split(" ")[-1])
        baseline_hits_at_10 = float(baseline_log[-1].split(" ")[-1])
        baseline_mr = float(baseline_log[-4].split(" ")[-1])
        baseline_mrr = float(baseline_log[-5].split(" ")[-1])
    return baseline_hits_at_1, baseline_hits_at_3, baseline_hits_at_10, baseline_mr, baseline_mrr


if __name__ == "__main__":
    datasets_names = {"freebase": "FB15k-237", "wordnet": "WN18RR", "nell": "NELL-995"}
    algs_names = {"transe": "TransE", "distmult": "DistMult", "complex": "ComplEx", "rotate": "RotatE"}

    num_leiden_samples, num_random_samples = 1000, 10000
    N, K, V_size, V_star_size, total_eval_embeddings = dict(), dict(), dict(), dict(), dict()
    datasets_resolution, datasets_modularity = dict(), dict()
    datasets_speed_up, datasets_bulk_up = dict(), dict()
    leiden_resolution_scales = np.logspace(-5, 5, num_leiden_samples)
    plot_data_leiden = []
    plot_data_random = []
    for dataset_key, dataset_name in datasets_names.items():
        with open(f"graph_completion/configs/{dataset_key}.yml", "r", encoding="utf-8") as config_file:
            dataset_conf = ExperimentHpars.from_dict(yaml.safe_load(config_file)).make()
        dataset_loader = LoaderHpars.from_dict(dataset_conf.loader_hpars).make()
        dataset_loader.load_graph(dataset_conf.seed, "cpu",
                                  dataset_conf.val_size, dataset_conf.test_size, dataset_conf.leiden_resolution)
        N[dataset_key] = len(dataset_loader.test_edge_data)
        K[dataset_key] = dataset_loader.num_communities
        V_size[dataset_key] = dataset_loader.num_nodes
        V_star_size[dataset_key] = dataset_loader.inter_community_map.max()
        com_test_edges = dataset_loader.test_edge_data.assign(
            c_t=dataset_loader.communities[dataset_loader.test_edge_data.t]
        ).groupby("c_t").size().reindex(np.arange(dataset_loader.num_communities), fill_value=0).values
        com_test_embeddings = dataset_loader.num_communities + dataset_loader.community_sizes
        total_eval_embeddings[dataset_key] = (com_test_edges * com_test_embeddings).sum()
        datasets_speed_up[dataset_key] = (N[dataset_key] * V_size[dataset_key]) / total_eval_embeddings[dataset_key]
        datasets_bulk_up[dataset_key] = (V_size[dataset_key] + K[dataset_key]
                                         + V_star_size[dataset_key] + 1) / V_size[dataset_key]
        print(f"Acceleration for {datasets_names[dataset_key]}: {round(datasets_speed_up[dataset_key], 4)}")
        print(f"Overparametrization for {datasets_names[dataset_key]}: {round(datasets_bulk_up[dataset_key], 4)}")

        subgraphing = Subgraphing(dataset_loader.graph, dataset_loader.num_nodes, dataset_conf.leiden_resolution)
        subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
        subgraphing.compute_metrics()
        datasets_resolution[dataset_key] = subgraphing.community_resolution_disjoint
        datasets_modularity[dataset_key] = subgraphing.metric_values['DisjointCommunityModularity'][-1]
        leiden_resolution_values = subgraphing.community_resolution_disjoint * leiden_resolution_scales
        for leiden_resolution in tqdm(leiden_resolution_values, "Validating Leiden resolution", leave=False):
            subgraphing.community_resolution_disjoint = leiden_resolution
            subgraphing.recursive_updates(dataset_loader.dataset.node_data, dataset_loader.dataset.edge_data)
            subgraphing.compute_metrics()
            communities = subgraphing.metric_values["DisjointCommunityMembership"][-1]
            num_communities = int(subgraphing.metric_values['DisjointCommunityNumber'][-1])
            community_sizes = np.rint(
                dataset_loader.num_nodes * subgraphing.metric_values["DisjointCommunitySizeDist"][-1][:num_communities]
            ).astype(int)
            com_test_edges = dataset_loader.test_edge_data.assign(
                c_t=communities[dataset_loader.test_edge_data.t]
            ).groupby("c_t").size().reindex(np.arange(num_communities), fill_value=0).values
            std_community_sizes, std_com_test_edges = np.std(community_sizes), np.std(com_test_edges)
            modularity = subgraphing.metric_values['DisjointCommunityModularity'][-1]
            speed_up = ((N[dataset_key] * V_size[dataset_key])
                        / ((num_communities + community_sizes) * com_test_edges).sum())
            inter_community_edges = (communities[dataset_loader.dataset.edge_data.s]
                                     != communities[dataset_loader.dataset.edge_data.t])
            inter_community_nodes = np.unique(np.concatenate((
                dataset_loader.dataset.edge_data[inter_community_edges].s.values,
                dataset_loader.dataset.edge_data[inter_community_edges].t.values)
            ))
            bulk_up = (V_size[dataset_key] + num_communities + len(inter_community_nodes) + 1) / V_size[dataset_key]
            plot_data_leiden.append((dataset_key, leiden_resolution,
                                     std_community_sizes, std_com_test_edges, modularity,
                                     speed_up, bulk_up))

        communities_basis = np.repeat(np.arange(dataset_loader.num_communities), dataset_loader.community_sizes)
        for sample_id in tqdm(range(num_random_samples), "Validating random community partition", leave=False):
            communities_random = np.random.permutation(communities_basis)
            num_communities = dataset_loader.num_communities
            community_sizes = np.unique(communities_random, return_counts=True)[1]
            com_test_edges = dataset_loader.test_edge_data.assign(
                c_t=communities_random[dataset_loader.test_edge_data.t]
            ).groupby("c_t").size().reindex(np.arange(num_communities), fill_value=0).values
            std_community_sizes, std_com_test_edges = np.std(community_sizes), np.std(com_test_edges)
            modularity = dataset_loader.graph.graph.modularity(communities_random)
            speed_up = ((N[dataset_key] * V_size[dataset_key])
                        / ((num_communities + community_sizes) * com_test_edges).sum())
            inter_community_edges = (communities_random[dataset_loader.dataset.edge_data.s]
                                     != communities_random[dataset_loader.dataset.edge_data.t])
            inter_community_nodes = np.unique(np.concatenate((
                dataset_loader.dataset.edge_data[inter_community_edges].s.values,
                dataset_loader.dataset.edge_data[inter_community_edges].t.values)
            ))
            bulk_up = (V_size[dataset_key] + num_communities + len(inter_community_nodes) + 1) / V_size[dataset_key]
            plot_data_random.append((dataset_name, sample_id,
                                     std_community_sizes, std_com_test_edges, modularity,
                                     speed_up, bulk_up))
    plot_data_leiden = pd.DataFrame(plot_data_leiden, columns=["Dataset", "Resolution",
                                                               "StdCommunitySizes", "StdCommunityNumTestEdges",
                                                               "Modularity", "Acceleration", "Overparametrization"])
    plot_data_leiden = plot_data_leiden.melt(id_vars=["Dataset", "Resolution",
                                                      "StdCommunitySizes", "StdCommunityNumTestEdges", "Modularity"],
                                             value_vars=["Acceleration", "Overparametrization"],
                                             var_name="Factor", value_name="Value")
    plot_data_random = pd.DataFrame(plot_data_random, columns=["Dataset", "SampleId",
                                                               "StdCommunitySizes", "StdCommunityNumTestEdges",
                                                               "Modularity", "Acceleration", "Overparametrization"])

    plt.figure(figsize=(3 * 11.7, 8.3))
    g = sns.relplot(x="Resolution", y="Value", hue="Factor", col="Dataset", data=plot_data_leiden,
                    palette="Set1", linewidth=5, kind="line",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    for dataset_key, ax in g.axes_dict.items():
        ax.axvline(datasets_resolution[dataset_key], 0, 100, color="black", linestyle="dashed", lw=5)
        ax.set_title(datasets_names[dataset_key])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Leiden resolution")
        ax.set_ylabel("Factor value")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/scalability_leiden_resolution.pdf", format="pdf")

    plt.figure(figsize=(3 * 11.7, 8.3))
    g = sns.relplot(x="Modularity", y="Value", hue="Factor", col="Dataset", data=plot_data_leiden,
                    palette="Set1", linewidth=5, kind="line",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    for dataset_key, ax in g.axes_dict.items():
        ax.axvline(datasets_modularity[dataset_key], 0, 100, color="black", linestyle="dashed", lw=5)
        ax.set_title(datasets_names[dataset_key])
        ax.set_yscale("log")
        ax.set_xlabel("Modularity")
        ax.set_ylabel("Factor value")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/scalability_leiden_modularity.pdf", format="pdf")

    train_logs = [pd.read_csv(f"{run_folder}/train_log.txt", sep="\t", encoding="utf-8", header=None)
                  for dataset in datasets_names for run_folder in glob(f"graph_completion/results/{dataset}/runs/*")
                  if exists(f"{run_folder}/train_log.txt")]
    test_logs = [pd.read_csv(f"{run_folder}/test_log.txt", sep="\t", encoding="utf-8",
                             header=None, index_col=None)
                 for dataset in datasets_names for run_folder in glob(f"graph_completion/results/{dataset}/runs/*")
                 if exists(f"{run_folder}/test_log.txt")]
    train_logs = pd.concat(train_logs)
    test_logs = pd.concat(test_logs, ignore_index=True)

    train_logs.columns = ["MiniBatchSize", "LearningRate", "Algorithm", "LeidenResolution",
                          "EmbeddingDim", "LossMargin", "Dataset", "NumNodes", "NumNodeTypes", "NumRelations",
                          "NumCommunities", "NumNegativeSamples", "EdgeDensity", "NodeTypeAssortativity",
                          "GiantWCC", "GiantSCC", "AveragePathLength", "Diameter", "AverageClustering",
                          "TrainComLoss", "TrainNodeLoss", "TrainLoss", "TrainTime",
                          "ValComLoss", "ValNodeLoss", "ValLoss",
                          "ComAccuracy", "Accuracy", "ComPrecision", "Precision", "ComRecall", "Recall", "ComF1", "F1",
                          "ComROC-AUC", "ROC-AUC", "ComAP", "AP", "Patience"]
    train_logs = train_logs.reset_index().rename(columns={"index": "Batch"})
    train_logs = train_logs.assign(
        NumSamples=(train_logs.Batch + 1) * train_logs.MiniBatchSize * (train_logs.NumNegativeSamples + 1)
    )
    train_logs.loc[train_logs.Algorithm.isin(["distmult", "complex"]), "NumSamples"] *= 10
    test_logs.columns = ["Checkpoint", "MiniBatchSize", "LearningRate", "Algorithm", "LeidenResolution",
                         "EmbeddingDim", "LossMargin", "Dataset", "NumNodes", "NumNodeTypes", "NumRelations",
                         "NumCommunities", "NumNegativeSamples", "EdgeDensity", "NodeTypeAssortativity",
                         "GiantWCC", "GiantSCC", "AveragePathLength", "Diameter", "AverageClustering",
                         "TestComLoss", "TestNodeLoss", "TestLoss",
                         "ComAccuracy", "Accuracy", "ComPrecision", "Precision", "ComRecall", "Recall", "ComF1", "F1",
                         "ComROC-AUC", "ROC-AUC", "ComAP", "AP", "ComHits@1", "NodeHits@1", "Hits@1",
                         "ComHits@3", "NodeHits@3", "Hits@3", "ComHits@10", "NodeHits@10", "Hits@10",
                         "ComMR", "NodeMR", "MR", "ComMRR", "NodeMRR", "MRR", "TestTime"]
    test_logs = test_logs.assign(Value="COINs")

    test_logs_baseline = []
    for dataset_key, dataset_name in datasets_names.items():
        for algorithm_key, algorithm_name in algs_names.items():
            test_logs_baseline.append((dataset_key, algorithm_key,
                                       *parse_baseline_metrics(dataset_key, algorithm_name)))
    test_logs_baseline = pd.DataFrame(test_logs_baseline, columns=["Dataset", "Algorithm",
                                                                   "Hits@1", "Hits@3", "Hits@10", "MR", "MRR"])
    test_logs_baseline = test_logs_baseline.assign(Value="Baseline")

    results_table = pd.concat((test_logs_baseline, test_logs))
    results_table = results_table.groupby(["Dataset", "Algorithm", "Value"])[
        ["Hits@1", "Hits@3", "Hits@10", "MRR"]
    ].max().round(decimals=3)
    results_table = results_table.reindex(list(datasets_names.keys()), level=0)
    results_table = results_table.reindex(list(algs_names.keys()), level=1)
    results_table = results_table.reindex(["Baseline", "COINs"], level=2)
    print(results_table)
    results_table_2 = test_logs.groupby(["Dataset", "Algorithm"], sort=False)[
        ["ComHits@1", "NodeHits@1", "ComHits@3", "NodeHits@3", "ComHits@10", "NodeHits@10", "ComMRR", "NodeMRR"]
    ].max().reset_index()
    results_table_2_parts = []
    for metric in ["Hits@1", "Hits@3", "Hits@10", "MRR"]:
        results_table_2_part = results_table_2.melt(id_vars=["Dataset", "Algorithm"],
                                                    value_vars=[f"Com{metric}", f"Node{metric}"],
                                                    var_name="Value",
                                                    value_name=metric)
        results_table_2_part.Value = results_table_2_part.Value.map({f"Com{metric}": "Community",
                                                                     f"Node{metric}": "Node"})
        results_table_2_part = results_table_2_part.set_index(["Dataset", "Algorithm", "Value"])
        results_table_2_parts.append(results_table_2_part)
    results_table_2 = pd.concat(results_table_2_parts, axis="columns").round(decimals=3)
    results_table_2 = results_table_2.reindex(list(datasets_names.keys()), level=0)
    results_table_2 = results_table_2.reindex(list(algs_names.keys()), level=1)
    results_table_2 = results_table_2.reindex(["Community", "Node"], level=2)
    print(results_table_2)
    results_table_3 = test_logs.groupby(["Dataset", "Algorithm"], sort=False)[
        ["ComAccuracy", "Accuracy", "ComF1", "F1", "ComROC-AUC", "ROC-AUC", "ComAP", "AP"]
    ].max().reset_index()
    results_table_3_parts = []
    for metric in ["Accuracy", "F1", "ROC-AUC", "AP"]:
        results_table_3_part = results_table_3.melt(id_vars=["Dataset", "Algorithm"],
                                                    value_vars=[f"Com{metric}", metric],
                                                    var_name="Value",
                                                    value_name=metric)
        results_table_3_part.Value = results_table_3_part.Value.map({f"Com{metric}": "Community",
                                                                     metric: "Overall"})
        results_table_3_part = results_table_3_part.set_index(["Dataset", "Algorithm", "Value"])
        results_table_3_parts.append(results_table_3_part)
    results_table_3 = pd.concat(results_table_3_parts, axis="columns").round(decimals=3)
    results_table_3 = results_table_3.reindex(list(datasets_names.keys()), level=0)
    results_table_3 = results_table_3.reindex(list(algs_names.keys()), level=1)
    results_table_3 = results_table_3.reindex(["Community", "Overall"], level=2)
    print(results_table_3)

    test_logs = test_logs.groupby(["Dataset", "Algorithm"], sort=False)[
        ["NodeHits@1", "NodeHits@3", "NodeHits@10", "Hits@1", "Hits@3", "Hits@10"]
    ].max()
    test_logs_baseline = test_logs_baseline.set_index(["Dataset", "Algorithm"])[["Hits@1", "Hits@3", "Hits@10"]]
    relative_error = (test_logs_baseline - test_logs[["Hits@1", "Hits@3", "Hits@10"]]) / test_logs_baseline
    relative_error_node = (test_logs_baseline - test_logs[
        ["NodeHits@1", "NodeHits@3", "NodeHits@10"]
    ].rename(columns={f"Node{hits_metric}": hits_metric for hits_metric in ["Hits@1", "Hits@3", "Hits@10"]})
                           ) / test_logs_baseline

    relative_error, relative_error_node = relative_error.reset_index(), relative_error_node.reset_index()
    plot_data = relative_error.assign(Acceleration=relative_error.Dataset.map(datasets_speed_up),
                                      Value="Overall")
    speed_up = plot_data.groupby("Dataset")["Acceleration"].mean()
    print(f"Average acceleration: {speed_up.mean().round(decimals=4)} (+/- {speed_up.std().round(decimals=4)})")
    rho = plot_data[["Hits@1", "Hits@3", "Hits@10"]].values
    rho = rho[np.abs(rho) <= 1]  # Ignore outliers
    print(f"Average relative error: {rho.mean().round(decimals=4)} (+/- {rho.std().round(decimals=4)})")

    plot_data_node = relative_error_node.assign(Acceleration=relative_error_node.Dataset.map(datasets_speed_up),
                                                Value="Node")
    plot_data = pd.concat((plot_data, plot_data_node), ignore_index=True)
    plot_data.Dataset = plot_data.Dataset.map(datasets_names)
    plot_data.Algorithm = plot_data.Algorithm.map(algs_names)
    plot_data = plot_data.melt(id_vars=["Dataset", "Algorithm", "Acceleration", "Value"],
                               value_vars=["Hits@1", "Hits@3", "Hits@10"],
                               var_name="Metric", value_name="RelativeError")
    x_boundary = np.linspace(-20, 1, 10000, endpoint=False)
    y_boundary = 1 / (1 - x_boundary)

    plt.figure(figsize=(3 * 11.7, 2 * 8.3))
    g = sns.relplot(x="RelativeError", y="Acceleration", hue="Dataset", style="Algorithm",
                    row="Value", col="Metric", data=plot_data,
                    palette="Set1", s=1000, kind="scatter",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"legend_out": True})
    for (value, metric), ax in g.axes_dict.items():
        ax.set_title(f"{value} {metric}")
        ax.fill_between(x_boundary, y_boundary, 15, color="gray", alpha=0.25)
        ax.plot(x_boundary, y_boundary, color="black", linestyle="dashed", lw=5)
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 15)
        ax.set_xlabel("Relative error")
    for lh in g.legend.legendHandles:
        lh.set_sizes([1000])
    g.tight_layout()
    plt.savefig("graph_completion/results/feasibility.pdf", format="pdf")

    plot_data_train = train_logs[["Dataset", "Algorithm", "NumSamples", "TrainComLoss", "TrainNodeLoss", "TrainLoss"]]
    plot_data_train = plot_data_train.rename(
        columns={"TrainComLoss": "Community", "TrainNodeLoss": "Node", "TrainLoss": "Overall"}
    ).melt(id_vars=["Dataset", "Algorithm", "NumSamples"], value_vars=["Community", "Node", "Overall"],
           var_name="Value", value_name="Loss").assign(Subset="Training")
    plot_data_valid = train_logs[["Dataset", "Algorithm", "NumSamples", "ValComLoss", "ValNodeLoss", "ValLoss"]]
    plot_data_valid = plot_data_valid.rename(
        columns={"ValComLoss": "Community", "ValNodeLoss": "Node", "ValLoss": "Overall"}
    ).melt(id_vars=["Dataset", "Algorithm", "NumSamples"], value_vars=["Community", "Node", "Overall"],
           var_name="Value", value_name="Loss").assign(Subset="Validation")
    plot_data = pd.concat((plot_data_train, plot_data_valid), ignore_index=True)

    plt.figure(figsize=(4 * 11.7, 2 * 8.3))
    g = sns.relplot(x="NumSamples", y="Loss", hue="Value",
                    row="Subset", col="Algorithm", data=plot_data,
                    palette="Set1", linewidth=5, kind="line",
                    height=8.3, aspect=11.7 / 8.3, facet_kws={"sharex": False, "legend_out": True})
    for (subset, algorithm_key), ax in g.axes_dict.items():
        ax.set_title(f"{algs_names[algorithm_key]} {subset}")
        ax.set_xscale("log")
        if algorithm_key in ["transe", "rotate"]:
            ax.set_xlim(right=2e8)
        ax.set_xlabel("Number of training samples")
    for lh in g.legend.get_lines():
        lh.set_linewidth(5)
    g.tight_layout()
    plt.savefig("graph_completion/results/convergence.pdf", format="pdf")
