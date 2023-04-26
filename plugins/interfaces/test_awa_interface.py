from typing import Dict, List

from plugins.interfaces.epics_interface import EPICSInterface


class TestAWAInterface(EPICSInterface):
    name = "awa_interface"
    x: float = None
    y: float = None

    class Config:
        arbitrary_types_allowed = True

    def set_channels(self, channel_inputs: Dict[str, float]):
        for name, val in channel_inputs.items():
            if name == "AWA:Bira3Ctrl:Ch00":
                self.x = abs(val)
                self.y = -abs(val) + 0.5 

    def get_channels(self, channels: List[str]) -> Dict[str, float]:
        data = {}
        for name in channels:
            if name == "AWAICTMon:Ch1":
                data[name] = 1.0
            elif name == "YAG1:XRMS":
                data[name] = self.x
            elif name == "YAG1:YRMS":
                data[name] = self.y

        return data
