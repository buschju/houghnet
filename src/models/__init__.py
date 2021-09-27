from typing import Type

from models.cash import CASH
from models.fourc import FourC
from models.houghnet import HoughNet
from models.kplanes import KPlanes
from models.orclus import ORCLUS
from models.ssc import SSC

_MODELS = {
    'CASH': CASH,
    'FourC': FourC,
    'HoughNet': HoughNet,
    'KPlanes': KPlanes,
    'ORCLUS': ORCLUS,
    'SSC': SSC,
}


def get_model_class(model_name: str,
                    ) -> Type:
    try:
        return _MODELS[model_name]
    except KeyError:
        raise ValueError(f'Unknown model: {model_name}')
