from typing import Dict, List

from epics import caget_many, caput

from plugins.interfaces.interface import Interface


class EPICSInterface(Interface):
    name = "EPICS"

    def set_channels(self, channel_inputs: Dict[str, float]):
        for name, val in channel_inputs.items():
            caput(name, val)

    def get_channels(self, channels: List[str]) -> Dict[str, float]:
        return dict(zip(channels, caget_many(channels)))
