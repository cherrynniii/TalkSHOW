from .smplx_face import TrainWrapper as s2g_face
from .smplx_body_vq import TrainWrapper as s2g_body_vq
from .smplx_body_pixel import TrainWrapper as s2g_body_pixel
from .smplx_simple import TrainWrapper as s2g_simple
from .body_ae import TrainWrapper as s2g_body_ae
from .LS3DCG import TrainWrapper as LS3DCG
from .smplx_simple_mlp import TrainWrapper as s2g_simple_mlp
from .base import TrainWrapperBaseClass
from .smplx_simple_transformer import TrainWrapper as s2g_simple_transformer

from .utils import normalize, denormalize