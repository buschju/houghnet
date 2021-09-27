import os
import random
from typing import Tuple, Dict, Any, Optional, List

import numpy
import pandas


def load_dataset(data_root: str,
                 dataset_name: str,
                 ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    x = pandas.read_csv(os.path.join(data_root, f'{dataset_name}.csv'),
                        header=None,
                        index_col=None,
                        ).values.astype(numpy.float32)
    y = numpy.load(os.path.join(data_root, f'{dataset_name}_labels.npy'))

    return x, y


def generate_3lines(num_points_per_cluster: int,
                    jitter_factor: float,
                    noise_fraction: float,
                    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    step_size = 2. / num_points_per_cluster
    num_noise_points = int((noise_fraction * 3. * num_points_per_cluster) / (1. - noise_fraction))

    x = numpy.concatenate(
        [
            numpy.array([[i, 0.6] for i in numpy.arange(-1, 1, step_size)], dtype=numpy.float32),
            numpy.array([[-10 * i / 100 + .3, i] for i in numpy.arange(-1, 1, step_size)], dtype=numpy.float32),
            numpy.array([[i, i] for i in numpy.arange(-1, 1, step_size)], dtype=numpy.float32),
            (numpy.random.rand(num_noise_points, 2).astype(numpy.float32) - .5) * 2.,
        ],
        axis=0,
    )
    x += numpy.random.randn(x.shape[0], 2) * jitter_factor

    y = numpy.concatenate(
        [
            numpy.zeros(num_points_per_cluster),
            numpy.ones(num_points_per_cluster),
            2 * numpy.ones(num_points_per_cluster),
            -1 * numpy.ones(num_noise_points),
        ]
    ).astype(numpy.int32)

    return x, y


def generate_2p1l(num_points_per_cluster: int,
                  jitter_factor: float,
                  noise_fraction: float,
                  ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    cluster_dims = numpy.array(
        [
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
        ],
    )
    num_clusters = cluster_dims.shape[0]
    num_dims = cluster_dims.shape[1]
    num_noise_points = int((noise_fraction * 3. * num_points_per_cluster) / (1. - noise_fraction))

    xs = []
    ys = []
    for i in range(num_clusters):
        dims = []
        for j in range(num_dims):
            dims.append(numpy.random.uniform(-1, 1, num_points_per_cluster).reshape(num_points_per_cluster, 1))
        for (y,), value in numpy.ndenumerate(cluster_dims[i, :]):
            if value == 1:
                dims[y] = dims[y - 1] * random.random()
        xs += [numpy.concatenate(dims, axis=1)]
        ys += [numpy.ones(shape=(num_points_per_cluster, 1)) * i]

    xs[1] += [0., 0., 0.3]
    xs[2] -= [0.2, 0., 0.]

    x = numpy.concatenate(xs, axis=0)
    y = numpy.concatenate(ys, axis=0).squeeze()

    x = x + numpy.random.randn(x.shape[0], num_dims) * jitter_factor

    x = numpy.concatenate([x, (numpy.random.rand(num_noise_points, num_dims).astype(numpy.float32) - .5) * 2.], axis=0)
    y = numpy.concatenate([y, -1 * numpy.ones(num_noise_points)]).astype(numpy.int32)

    return x, y


def generate_highd_dataset(num_points_per_cluster: int,
                           num_dims: int,
                           cluster_dims: List[int],
                           jitter_factor: float,
                           noise_fraction: float,
                           ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    num_clusters = len(cluster_dims)
    num_noise_points = int((noise_fraction * num_clusters * num_points_per_cluster) / (1. - noise_fraction))

    x = numpy.concatenate(
        [
            numpy.random.rand(num_points_per_cluster, cluster_dim) @ numpy.random.rand(cluster_dim, num_dims)
            for cluster_dim in cluster_dims
        ],
        axis=0,
    )
    y = numpy.concatenate(
        [
            float(i) * numpy.ones(num_points_per_cluster) for i in range(num_clusters)
        ],
    )

    x -= x.min(axis=0, keepdims=True)
    x /= x.max(axis=0, keepdims=True)
    x = (x - 0.5) * 2.

    x += numpy.random.rand(x.shape[0], num_dims) * jitter_factor

    x = numpy.concatenate([x, numpy.random.uniform(x.min(axis=0), x.max(axis=0), (num_noise_points, num_dims))], axis=0)
    y = numpy.concatenate([y, -1 * numpy.ones(num_noise_points)])

    return x, y


def flatten_dictionary(dictionary: Dict[str, Any],
                       separator: str = '.',
                       prefix: Optional[str] = '',
                       ) -> Dict[str, Any]:
    flat_dict = {prefix + separator + k if prefix else k: v
                 for kk, vv in dictionary.items()
                 for k, v in flatten_dictionary(vv, separator, kk).items()
                 } if isinstance(dictionary, dict) else {prefix: dictionary}
    return flat_dict
