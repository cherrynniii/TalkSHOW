from .smplx_face import TrainWrapper as s2g_face
from .smplx_body_vq import TrainWrapper as s2g_body_vq
from .smplx_body_pixel import TrainWrapper as s2g_body_pixel
from .smplx_body_rnn import TrainWrapper as s2g_body_rnn
from .body_ae import TrainWrapper as s2g_body_ae
from .LS3DCG import TrainWrapper as LS3DCG
from .base import TrainWrapperBaseClass
from .smplx_body_rnn2 import TrainWrapper as s2g_body_rnn2
from .smplx_body_rnn3 import TrainWrapper as s2g_body_rnn3

from .utils import normalize, denormalize