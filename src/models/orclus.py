from typing import Optional

from models.elki_model import ElkiModel


class ORCLUS(ElkiModel):
    def __init__(self,
                 num_clusters: int,
                 num_dims: int,
                 alpha: float,
                 k_i: Optional[int] = None,
                 max_subspace_dimensionality: Optional[int] = None,
                 ):
        super().__init__()

        self.k = str(num_clusters)
        if k_i is None:
            k_i = 10 * num_clusters
        self.k_i = str(k_i)
        if max_subspace_dimensionality is None:
            max_subspace_dimensionality = num_dims - 1
        self.l = str(max_subspace_dimensionality)
        self.alpha = str(alpha)

    def get_java_command(self,
                         data_path: str,
                         ) -> str:
        command = (
            f'java -jar {self.elki_path} KDDCLIApplication  '
            f'-dbc.in {data_path}  '
            '-time -algorithm clustering.correlation.ORCLUS  '
            f'-projectedclustering.k {self.k}  '
            f'-projectedclustering.l {self.l}  '
            f'-projectedclustering.k_i {self.k_i}  '
            f'-orclus.alpha {self.alpha}  '
            f'-orclus.seed  {self.random_seed}  '
            f'-resulthandler ResultWriter  '
            f'-out {self.output_path}{self.id}'
        )

        return command
