from typing import Tuple

import numpy
from munkres import Munkres
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder


def match_labels(y_1: numpy.ndarray,
                 y_2: numpy.ndarray,
                 ) -> numpy.ndarray:
    labels_1 = numpy.unique(y_1)
    num_labels_1 = labels_1.size
    labels_2 = numpy.unique(y_2)
    num_labels_2 = labels_2.size

    assert (num_labels_1 >= num_labels_2)

    graph = numpy.zeros((num_labels_2, num_labels_1))
    for i in range(num_labels_1):
        idx_label_1 = y_1 == labels_1[i]
        for j in range(num_labels_2):
            idx_label_2 = y_2 == labels_2[j]
            graph[j, i] = -(idx_label_1 * idx_label_2).sum().astype(numpy.float32)

    matching = numpy.array(Munkres().compute(graph))[:, 1]
    matched_y_2 = numpy.zeros_like(y_2)
    for i in range(num_labels_2):
        matched_y_2[y_2 == labels_2[i]] = labels_1[matching[i]]

    return matched_y_2


def get_clustering_accuracy(y_true: numpy.ndarray,
                            y_pred: numpy.ndarray,
                            ) -> float:
    if numpy.unique(y_true).size > numpy.unique(y_pred).size:
        y_pred_matched = match_labels(y_true, y_pred)
        accuracy = (y_true == y_pred_matched).sum().astype(numpy.float32) / y_true.shape[0]
    else:
        y_true_matched = match_labels(y_pred, y_true)
        accuracy = (y_true_matched == y_pred).sum().astype(numpy.float32) / y_true.shape[0]

    return accuracy


def get_performance_metrics(y_true: numpy.ndarray,
                            y_pred: numpy.ndarray,
                            ) -> Tuple[float, float, float]:
    non_noise_mask = y_true != -1
    y_true = y_true[non_noise_mask]
    y_pred = y_pred[non_noise_mask]

    # Points detected by the algorithm as noise are collected in a new noise cluster
    y_pred[y_pred == -1] = y_pred.max() + 1

    # Relabel in case an entire cluster was masked out in y_pred
    y_pred = LabelEncoder().fit_transform(y_pred)

    acc = get_clustering_accuracy(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')

    return acc, ari, nmi
