import json
import os

import mlflow
from sklearn.model_selection import ParameterGrid

from data_utils import flatten_dictionary
from run_experiment import run_experiment

# Parameter grids
MODEL_PARAMETERS_GRID = {
    'CASH': {
        'minPts': [250, 500],
        'maxLevel': [25, 50],
        # For high-dimensional datasets
        # 'minPts': [1000],
        # 'maxLevel': [1],
    },
    'FourC': {
        'minPts': [3, 5, 8, 13],
        'epsilon': [0.05, 0.1, 0.2],
    },
    'KPlanes': {
    },
    'HoughNet': {
        'lambda_activation': [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'ORCLUS': {
        'alpha': [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'SSC': {
        'lambda_regularization': [0.0001, 0.001, 0.01, 0.1, 1.],
    },
}
TRAINING_PARAMETERS_GRID = {
    'CASH': {
    },
    'FourC': {
    },
    'KPlanes': {
    },
    'HoughNet': {
        'num_epochs': [10, 20, 50, 100],
    },
    'ORCLUS': {
    },
    'SSC': {
    },
}


def run_grid_search(data_root: str,
                    config_root: str,
                    mlflow_uri: str,
                    mlflow_experiment_name: str,
                    model_name: str,
                    dataset_name: str,
                    device: int,
                    ) -> None:
    # Load default config
    config_path = os.path.join(config_root, f'{model_name.lower()}.json')
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    config['dataset_name'] = dataset_name

    # Set up MLflow results tracking
    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(name=mlflow_experiment_name)
    if experiment is None:
        mlflow.create_experiment(mlflow_experiment_name)
    mlflow.set_experiment(mlflow_experiment_name)

    # Iterate over grid
    count = 1
    for model_parameters in ParameterGrid(MODEL_PARAMETERS_GRID[model_name]):
        config['model_parameters'].update(model_parameters)
        for training_parameters in ParameterGrid(TRAINING_PARAMETERS_GRID[model_name]):
            config['training_parameters'].update(training_parameters)

            print(f'{model_name}, {dataset_name}')
            print(f'Trying parameter configuration {count}:')
            print(f'Model parameters: {model_parameters}')
            print(f'Training parameters: {training_parameters}')
            count += 1

            with mlflow.start_run():
                # Log params
                params_flat = flatten_dictionary(config)
                mlflow.log_params(params_flat)

                # Run experiment
                run_experiment(data_root=data_root,
                               dataset_name=config['dataset_name'],
                               model_name=config['model_name'],
                               model_parameters=config['model_parameters'],
                               training_parameters=config['training_parameters'],
                               run_parameters=config['run_parameters'],
                               seed=config['seed'],
                               device=device,
                               )


if __name__ == '__main__':
    data_root = '../data'
    config_root = '../config'

    mlflow_uri = '../mlflow'
    mlflow_experiment_name = 'houghnet_gridsearch'

    dataset_name = '3lines_0.01_0.30'
    model_name = 'HoughNet'

    device = 0

    run_grid_search(data_root=data_root,
                    config_root=config_root,
                    mlflow_uri=mlflow_uri,
                    mlflow_experiment_name=mlflow_experiment_name,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    device=device,
                    )
