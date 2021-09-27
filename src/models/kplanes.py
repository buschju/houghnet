import numpy.random
import scipy
from scipy.sparse.linalg import ArpackNoConvergence


class KPlanes:
    def __init__(self,
                 num_dims: int,
                 num_clusters: int,
                 initial_data_mean: numpy.ndarray,
                 max_iterations: int,
                 ):
        super().__init__()

        self.num_dims = num_dims
        self.num_clusters = num_clusters
        self.initial_data_mean = initial_data_mean
        self.max_iterations = max_iterations

        self.weight = None
        self.bias = None

        self.reset_parameters()

    def fit(self,
            x: numpy.ndarray,
            ) -> None:
        previous_assignments = -1. * numpy.ones(x.shape[0])
        previous_loss = -1.

        for _ in range(self.max_iterations):
            # Cluster assignment step
            projection_distances = numpy.abs(self.get_projection_distance(x))
            assignments = projection_distances.argmin(axis=1)
            loss = projection_distances.min(axis=1).mean()

            # Stop if converged
            if (assignments == previous_assignments).all() or numpy.abs(previous_loss - loss) < 1e-12:
                break
            else:
                previous_assignments = assignments
                previous_loss = loss

            # Cluster update step
            for k in range(self.num_clusters):
                x_cluster = x[assignments == k, :]
                n_cluster = x_cluster.shape[0]
                if n_cluster > 0:
                    B = x_cluster.T \
                        @ (numpy.eye(n_cluster) - numpy.ones((n_cluster, n_cluster)) / n_cluster) \
                        @ x_cluster
                    try:
                        eig = scipy.sparse.linalg.eigsh(B,
                                                        k=1,
                                                        which='SM',
                                                        )[1]
                    except ArpackNoConvergence:
                        try:
                            B += 1e-8 * numpy.eye(self.num_dims)
                            eig = scipy.sparse.linalg.eigsh(B,
                                                            k=1,
                                                            which='SM',
                                                            )[1]
                        except ArpackNoConvergence:
                            break
                    self.weight[:, k] = eig.squeeze()
                    self.bias[0, k] = numpy.ones((1, n_cluster)) @ x_cluster @ eig / n_cluster

    def predict(self,
                x: numpy.ndarray,
                ) -> numpy.ndarray:
        return numpy.abs(self.get_projection_distance(x)).argmin(axis=1)

    def fit_predict(self,
                    x: numpy.ndarray,
                    ) -> numpy.ndarray:
        self.fit(x)
        return self.predict(x)

    def get_projection_distance(self,
                                x: numpy.ndarray,
                                ) -> numpy.ndarray:
        weights_normalized = self.weight / numpy.linalg.norm(self.weight, axis=0, keepdims=True)
        projection_distances = x @ weights_normalized - self.bias

        return projection_distances

    def reset_parameters(self):
        self.weight = numpy.random.uniform(0., 1., (self.num_dims, self.num_clusters))
        self.weight /= numpy.linalg.norm(self.weight, axis=0, keepdims=True)

        self.bias = self.initial_data_mean[None, :] @ self.weight
