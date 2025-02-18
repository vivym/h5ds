from .convertor import Convertor

from .bc_z import bc_z_data_spec
from .bridge import bridge_spec
from .bridge_v2 import bridge_v2_spec
from .cmu_play_fusion import cmu_play_fusion_spec
from .dobbe import dobbe_spec
from .droid import droid_spec
from .fmb import fmb_spec
from .fractal20220817_data import fractal20220817_data_spec
from .kuka import kuka_spec
from .maniskill_dataset_converted_externally_to_rlds import (
    maniskill_dataset_converted_externally_to_rlds_spec,
)

# TODO: robot_set, rh20t, kuka, berkeley_rpt_converted_externally_to_rlds, maniskill_dataset_converted_externally_to_rlds

DATASET_SPECS = {
    "bc_z": bc_z_data_spec,
    "bridge": bridge_spec,
    "bridge_v2": bridge_v2_spec,
    "cmu_play_fusion": cmu_play_fusion_spec,
    "dobbe": dobbe_spec,
    "droid": droid_spec,
    "fmb": fmb_spec,
    "fractal20220817_data": fractal20220817_data_spec,
    "kuka": kuka_spec,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_converted_externally_to_rlds_spec,
}
