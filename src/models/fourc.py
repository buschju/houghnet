from typing import Optional

from models.elki_model import ElkiModel


class FourC(ElkiModel):
    def __init__(self,
                 num_dims: int,
                 minPts: int,
                 epsilon: float,
                 absolute: bool,
                 delta: float,
                 kappa: float,
                 num_clusters: int = None,
                 max_subspace_dimensionality: Optional[int] = None,
                 ):
        super().__init__()

        self.minPts = str(minPts)
        self.epsilon = str(epsilon)
        self.absolute = str(absolute).lower()
        self.delta = delta
        self.kappa = kappa
        if max_subspace_dimensionality is None:
            max_subspace_dimensionality = num_dims - 1
        self.max_subspace_dimensionality = max_subspace_dimensionality

    def get_java_command(self,
                         data_path: str,
                         ) -> str:
        command = (
            f'java -jar {self.elki_path} KDDCLIApplication  '
            f'-dbc.in {data_path}  '
            '-time -algorithm clustering.correlation.FourC  '
            f'-dbscan.epsilon {self.epsilon}  '
            f'-dbscan.minpts {self.minPts}  '
            f'-pca.filter.absolute {self.absolute}  '
            f'-pca.filter.delta {self.delta}  '
            f'-predecon.kappa {self.kappa}  '
            f'-predecon.lambda {self.max_subspace_dimensionality}  '
            f'-resulthandler ResultWriter  '
            f'-out {self.output_path}{self.id}'
        )

        return command
