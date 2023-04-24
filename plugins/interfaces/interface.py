from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel, Field


class Interface(BaseModel, ABC):
    name: str
    params: Dict = Field({}, description="custom parameters")

    @abstractmethod
    def set_channels(self, channel_inputs: Dict[str, float]):
        pass

    @abstractmethod
    def get_channels(self, channels: List[str]) -> Dict[str, float]:
        pass
