"""
Main script.
"""
import argparse
from copy import deepcopy
from pprint import pprint
from typing import List

import yaml

from graph_completion.experiments import ExperimentHpars
from graph_completion.models.graph_embedders import GraphEmbedderHpars


def get_validation_configs(base_conf: dict) -> List[dict]:
    new_configs = []
    for algorithm in GraphEmbedderHpars.OPTIONS:
        for embedding_dim in [5, 10, 25, 50, 100]:
            for num_negative_samples in [1, 5, 10, 50, 100]:
                algorithm_margin_loss = algorithm in ["transe", "rotate", "kbgat"]
                algorithm_random_walks = algorithm in ["mlp", "gatne"]
                new_config = deepcopy(base_conf)
                new_config["algorithm"] = algorithm
                new_config["embedder_hpars"]["embedding_dim"] = embedding_dim
                new_config["loader_hpars"]["sampler_hpars"]["num_negative_samples"] = num_negative_samples
                new_config["loader_hpars"]["sample_source"] = "random_walks" if algorithm_random_walks else "smore"

                if algorithm_margin_loss:
                    for margin in [0.01, 0.1, 1.0, 2.0, 5.0]:
                        new_config_2 = deepcopy(new_config)
                        new_config_2["embedding_loss_hpars"]["margin"] = margin
                        new_configs.append(new_config_2)
                else:
                    new_configs.append(new_config)
    return new_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate model.')
    parser.add_argument("-cf", "--config_file", metavar="config_file", type=str, required=True,
                        help="Path to the YAML config file containing the parameter settings.")
    parser.add_argument("-v", "--validate", metavar="validate", type=bool, required=False, default=False,
                        help="Whether to run hyperparameter validation.")
    args = parser.parse_args()
    config_filepath, validate = args.config_file, args.validate
    with open(config_filepath, "r", encoding="utf-8") as config_file:
        base_conf = yaml.safe_load(config_file)

    experiment_configs = get_validation_configs(base_conf) if validate else [base_conf, ]
    for experiment_config in experiment_configs:
        experiment_hpars = ExperimentHpars.from_dict(experiment_config)
        pprint(experiment_config)
        experiment = experiment_hpars.make()
        experiment.main()
