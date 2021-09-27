import argparse
import json
import os
import random
import time

import mlflow
import numpy
import torch

from data_utils import load_dataset, flatten_dictionary
from eval_utils import get_performance_metrics
from models import get_model_class


def run_experiment(data_root: str,
                   dataset_name: str,
                   model_name: str,
                   model_parameters: dict,
                   training_parameters: dict,
                   run_parameters: dict,
                   seed: int,
                   device: int,
                   ) -> None:
    # Load data
    x, y_true = load_dataset(data_root=data_root,
                             dataset_name=dataset_name,
                             )
    num_dims = x.shape[1]
    num_clusters = numpy.unique(y_true[y_true != -1]).size
    if 'elki' in run_parameters.keys():
        x = os.path.join(data_root, f'{dataset_name}.csv')

    # Fix random seed
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Get model
    if 'initial_data_mean' in model_parameters.keys():
        model_parameters = model_parameters.copy()
        model_parameters['initial_data_mean'] = x.mean(axis=0)
    model = get_model_class(model_name=model_name)(**model_parameters,
                                                   num_dims=num_dims,
                                                   num_clusters=num_clusters,
                                                   )

    # Move to device
    if 'gpu' in run_parameters.keys() and device is not None:
        device = torch.device(device)
        model = model.to(device)

    # Run
    runtime_seconds = []
    accs = []
    aris = []
    nmis = []
    for run_idx in range(run_parameters['num_runs']):
        model.reset_parameters()

        start_time = time.time()
        y_pred = model.fit_predict(x=x,
                                   **training_parameters,
                                   )
        end_time = time.time()

        acc, ari, nmi = get_performance_metrics(y_true=y_true,
                                                y_pred=y_pred
                                                )

        accs += [acc]
        aris += [ari]
        nmis += [nmi]
        runtime_seconds += [end_time - start_time]

        mlflow.log_metric('acc', accs[run_idx], step=run_idx)
        mlflow.log_metric('ari', aris[run_idx], step=run_idx)
        mlflow.log_metric('nmi', nmis[run_idx], step=run_idx)
        mlflow.log_metric('runtime_seconds', runtime_seconds[run_idx], step=run_idx)

    # Log aggregated results over all runs
    acc_mean = numpy.mean(accs)
    acc_std = numpy.std(accs)
    ari_mean = numpy.mean(aris)
    ari_std = numpy.std(aris)
    nmi_mean = numpy.mean(nmis)
    nmi_std = numpy.std(nmis)
    runtime_seconds_mean = numpy.mean(runtime_seconds)
    runtime_seconds_std = numpy.std(runtime_seconds)
    mlflow.log_metric('acc_mean', acc_mean)
    mlflow.log_metric('acc_std', acc_std)
    mlflow.log_metric('ari_mean', ari_mean)
    mlflow.log_metric('ari_std', ari_std)
    mlflow.log_metric('nmi_mean', nmi_mean)
    mlflow.log_metric('nmi_std', nmi_std)
    mlflow.log_metric('runtime_seconds_mean', runtime_seconds_mean)
    mlflow.log_metric('runtime_seconds_std', runtime_seconds_std)


if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        help="Path to config file",
                        type=str,
                        required=True,
                        )
    parser.add_argument('--data_root',
                        help="Path to data",
                        type=str,
                        default='../data',
                        )
    parser.add_argument('--dataset_name',
                        help="Name of the dataset",
                        type=str,
                        required=True,
                        )
    parser.add_argument('--device',
                        help="Device index",
                        type=int,
                        required=True,
                        )
    parser.add_argument('--mlflow_uri',
                        help="MLflow tracking URI",
                        type=str,
                        default='../mlflow',
                        )
    parser.add_argument('--mlflow_experiment_name',
                        help="Experiment name used for MLflow results tracking",
                        type=str,
                        default='houghnet',
                        )
    args = parser.parse_args()

    # Parse config file
    with open(args.config_path, 'r') as config_file:
        config = json.load(config_file)
    config['dataset_name'] = args.dataset_name

    # Set up MLflow results tracking
    mlflow.set_tracking_uri(args.mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name=args.mlflow_experiment_name)
    if experiment is None:
        mlflow.create_experiment(args.mlflow_experiment_name)
    mlflow.set_experiment(args.mlflow_experiment_name)

    with mlflow.start_run():
        # Log params
        params_flat = flatten_dictionary(config)
        mlflow.log_params(params_flat)

        # Run experiment
        run_experiment(data_root=args.data_root,
                       dataset_name=config['dataset_name'],
                       model_name=config['model_name'],
                       model_parameters=config['model_parameters'],
                       training_parameters=config['training_parameters'],
                       run_parameters=config['run_parameters'],
                       seed=config['seed'],
                       device=args.device,
                       )
