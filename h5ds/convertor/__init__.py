from .convertor import Convertor

from .bc_z import bc_z_data_spec
from .bridge import bridge_spec
from .cmu_play_fusion import cmu_play_fusion_spec
from .dobbe import dobbe_spec
from .fractal20220817_data import fractal20220817_data_spec

DATASET_SPECS = {
    "bc_z": bc_z_data_spec,
    "bridge": bridge_spec,
    "cmu_play_fusion": cmu_play_fusion_spec,
    "dobbe": dobbe_spec,
    "fractal20220817_data": fractal20220817_data_spec,
}
