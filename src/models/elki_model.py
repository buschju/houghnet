import datetime
import os
import subprocess

import numpy

ELKI_PATH = 'elki/elki-bundle-0.7.5.jar'
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.relpath(__file__)), '../../output')


class ElkiModel:
    def __init__(self):
        super().__init__()

        self.elki_path = ELKI_PATH
        self.output_path = OUTPUT_PATH
        self.id = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S_%f")
        self.results_path = os.path.join(self.output_path, f'{self.__class__.__name__.lower()}_{self.id}')
        self.random_seed = 0

    def get_java_command(self,
                         data_path: str,
                         ) -> str:
        raise NotImplementedError()

    def fit_predict(self,
                    x: str,
                    ) -> numpy.ndarray:
        command = self.get_java_command(data_path=x)
        try:
            subprocess.call(command, shell=True)
        except subprocess.CalledProcessError as error:
            print(error.output)

        labels = {}
        cluster_id = 0

        for file_name in os.listdir(self.results_path):
            with open(os.path.join(self.results_path, file_name), 'r') as file:
                for line in file.readlines():
                    if line.startswith('#'):
                        continue
                    idx = int(line.split(' ')[0][3:]) - 1
                    if 'cluster' in file_name:
                        labels[idx] = cluster_id
                    elif 'noise' in file_name:
                        labels[idx] = -1
                if 'cluster' in file_name:
                    cluster_id += 1

        labels = numpy.array([labels[key] for key in range(len(labels))], dtype=numpy.int32)

        return labels

    def reset_parameters(self):
        self.random_seed += 1
