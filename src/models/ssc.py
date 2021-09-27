from typing import Optional

import cvxpy
import numpy
from joblib import Parallel, delayed
from sklearn.cluster import SpectralClustering


class SSC:
    """
    Based on the Matlab CVX Implementation provided at http://vision.jhu.edu/code/
    """

    def __init__(self,
                 num_dims: int,
                 num_clusters: int,
                 lambda_regularization: float,
                 max_subspace_dimensionality: Optional[int] = None,
                 ):
        super().__init__()

        self.num_dims = num_dims
        self.num_clusters = num_clusters
        self.lambda_regularization = lambda_regularization
        if max_subspace_dimensionality is None:
            self.maximum_subspace_dimension = num_dims - 1
        else:
            self.maximum_subspace_dimension = max_subspace_dimensionality

    def fit_predict(self,
                    x: numpy.ndarray,
                    ) -> numpy.ndarray:
        num_points = x.shape[0]

        # Recover reconstruction coefficient matrix

        coefficient_matrix = numpy.zeros((num_points, num_points))

        def compute_coefficients_point(i):
            x_row = x[i, :]
            x_rest = x[[j for j in range(num_points) if j != i], :]

            coefficients_row = cvxpy.Variable(num_points - 1)
            objective = cvxpy.Minimize(
                cvxpy.norm(coefficients_row, p=1)
                + self.lambda_regularization * cvxpy.norm(coefficients_row @ x_rest - x_row, p=2)
            )
            affine_constraint = [cvxpy.sum(coefficients_row) == 1.]
            optimization_problem = cvxpy.Problem(objective, affine_constraint)
            optimization_problem.solve()
            coefficient_matrix[i, :] = numpy.concatenate(
                [
                    coefficients_row.value[:i],
                    numpy.zeros(1),
                    coefficients_row.value[i:],
                ]
            )

        Parallel(n_jobs=-1, backend='threading')(delayed(compute_coefficients_point)(i) for i in range(num_points))

        # Set small entries to zero and check for NaNs
        coefficient_matrix[numpy.abs(coefficient_matrix) < 1e-12] = 0.
        if numpy.isnan(coefficient_matrix).any():
            print('Found NaN-values in coefficient matrix')

        # Build symmetric adjacency matrix

        coefficient_matrix = numpy.abs(coefficient_matrix)
        coefficient_matrix /= numpy.max(coefficient_matrix, axis=1, keepdims=True)
        coefficient_matrix = (coefficient_matrix + coefficient_matrix.T) / 2.

        adjacency_matrix = numpy.zeros_like(coefficient_matrix)
        sorted_indices = numpy.argsort(coefficient_matrix, axis=1)[:, ::-1]
        for i in range(num_points):
            max_idx = sorted_indices[i, :self.maximum_subspace_dimension + 1]
            adjacency_matrix[i, max_idx] = coefficient_matrix[i, max_idx] / coefficient_matrix[i, max_idx[0]]
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2.

        # Perform spectral clustering
        y_pred = SpectralClustering(n_clusters=self.num_clusters,
                                    affinity='precomputed',
                                    eigen_solver='arpack',
                                    assign_labels='discretize',
                                    random_state=0,
                                    n_jobs=-1,
                                    ).fit_predict(adjacency_matrix)

        return y_pred

    def reset_parameters(self):
        pass
