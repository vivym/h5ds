from dataclasses import dataclass


@dataclass
class RobotSpec:
    name: str

    num_arms: int

    action_space: str
