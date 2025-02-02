from numpy import ndarray
from dataclasses import dataclass, field


@dataclass
class Keyframe:
    descriptor: ndarray
    descriptor_size: int = field(init=False)

    def __post_init__(self) -> None:
        self.descriptor_size = len(self.descriptor)
