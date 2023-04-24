from typing import Dict, List

from pydantic import Field

from plugins.interfaces.camera import AWACamera
from plugins.interfaces.epics_interface import EPICSInterface


class TestAWAInterface(EPICSInterface):
    name = "awa_interface"
    camera_app: AWACamera = Field(
        AWACamera(None), description="awa frame grabber object"
    )
    x: float = None

    class Config:
        arbitrary_types_allowed = True

    def set_channels(self, channel_inputs: Dict[str, float]):
        for name, val in channel_inputs.items():
            if name == "Q6:BCTRL":
                self.x = val**2

    def get_channels(self, channels: List[str]) -> Dict[str, float]:
        data = {}
        for name in channels:
            if name == "AWAICTMon:Ch1":
                data[name] = 1.0
            elif name == "YAG1:XRMS":
                data[name] = self.x

        return data
