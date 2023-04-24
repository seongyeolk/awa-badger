from copy import deepcopy
from typing import Dict, List

from epics import caget_many
from plugins.interfaces.camera import AWACamera

from plugins.interfaces.epics_interface import EPICSInterface
from pydantic import Field


class AWAInterface(EPICSInterface):
    name = "awa_interface"
    camera_app: AWACamera = Field(
        AWACamera(None), description="awa frame grabber object"
    )

    class Config:
        arbitrary_types_allowed = True

    def get_channels(self, channels: List[str]) -> Dict:
        """make measurements of requested observables"""

        # separate out frame grabber names from epics names
        epics_obs = deepcopy(channels)
        fg_obs = []
        for idx, name in enumerate(epics_obs):
            if name in self.camera_app.channel_names:
                fg_obs += [epics_obs.pop(idx)]

        # if any of the observables are provided by the frame grabber use the frame
        # grabber application, otherwise use caget_many from epics
        if len(fg_obs) > 0:
            measurements = self.camera_app.get_measurement(epics_pvs=epics_obs)

        else:
            measurements = dict(zip(channels, caget_many(channels)))

        return measurements
