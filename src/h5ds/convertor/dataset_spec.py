from dataclasses import dataclass
from typing import Callable

from .robot_spec import RobotSpec


@dataclass
class DatasetSpec:
    rgb_obs_keys: dict[str, str] | None = None

    depth_obs_keys: dict[str, str] | None = None

    proprio_obs_key: str | None = None

    language_key: str | None = None

    standardize_func: Callable | None = None

    robot_spec: RobotSpec | None = None
