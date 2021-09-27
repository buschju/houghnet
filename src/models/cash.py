from models.elki_model import ElkiModel


class CASH(ElkiModel):
    def __init__(self,
                 minPts: int,
                 maxLevel: int,
                 jitter: float,
                 minDim: int,
                 adjust: bool,
                 num_clusters: int = None,
                 num_dims: int = None,
                 ):
        super().__init__()

        self.minPts = str(minPts)
        self.maxLevel = str(maxLevel)
        self.minDim = str(minDim)
        self.jitter = str(jitter)
        self.adjust = str(adjust).lower()

    def get_java_command(self,
                         data_path: str,
                         ) -> str:
        command = (
            f'java -jar {self.elki_path} KDDCLIApplication  '
            f'-dbc.in {data_path}  '
            '-time -algorithm clustering.correlation.CASH  '
            f'-cash.minpts {self.minPts}  '
            f'-cash.maxlevel {self.maxLevel}  '
            f'-cash.mindim {self.minDim}  '
            f'-cash.jitter {self.jitter}  '
            f'-cash.adjust {self.adjust}  '
            f'-resulthandler ResultWriter  '
            f'-out {self.results_path}'
        )

        return command
