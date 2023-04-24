from typing import List, Dict

from epics import caget, caput, caget_many

from plugins.interfaces.interface import Interface
from camera import AWACamera


class AWAInterface(Interface):
    name = "awa_interface"
    variables = {}
    observables = []

    def __init__(self):
        super().__init__()

        # initalize frame grabber camera
        self.camera_app = AWACamera()

    def get_variables(self, variable_names: List[str]) -> Dict:
        values = {}
        for name in variable_names:
            values[name] = caget(name)

        return values

    def set_variables(self, variable_inputs: Dict[str, float]):
        # use epics to set variables
        for name, val in variable_inputs.items():
            caput(name, val)

    def get_observables(self, observable_names: List[str]) -> Dict:

        camera_observables = []

        # if any of the observables are provided by the frame grabber use the frame
        # grabber application, otherwise use caget_many from epics
        if any(name in camera_observables for name in observable_names):
            measurements = self.camera_app.get_measurement()

        else:
            measurements = caget_many(observable_names)


        return measurements

