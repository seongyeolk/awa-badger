from typing import Dict, List, Optional

import pandas as pd
from pydantic import Field, PositiveFloat

from plugins.environments.environment import Environment, validate_observable_names
from plugins.interfaces.interface import Interface


# from plugins.interfaces.awa_interface import AWAInterface
# from plugins.interfaces.camera import AWACamera


class AWAEnvironment(Environment):
    name = "awa_environment"
    interface: Interface# = AWAInterface(camera_app=AWACamera(None))

    target_charge_PV: str = "AWAICTMon:Ch1"
    target_charge: Optional[PositiveFloat] = Field(
        None, description="magnitude of target charge in nC"
    )
    fractional_charge_deviation: PositiveFloat = Field(
        0.1, description="fractional deviation from target charge allowed"
    )

    def __init__(self, varaible_file: str, observable_file: str, interface: Interface,
                 **kwargs):

        # process variable and observable files to det variables and observables
        variable_info = pd.read_csv(varaible_file).set_index(
            "NAME")
        observable_info = pd.read_csv(
            observable_file
        ).set_index("NAME").T

        _variables = variable_info[["MIN", "MAX"]].T.to_dict()
        _observables = list(observable_info.keys())

        for name in _variables:
            _variables[name] = [_variables[name]["MIN"], _variables[name]["MAX"]]

        super(AWAEnvironment, self).__init__(
            variables=_variables,
            observables=_observables,
            interface=interface,
            **kwargs
        )

    @validate_observable_names
    def get_observables(self, observable_names: List[str]) -> Dict:
        """make measurements until charge range is within bounds"""

        while True:
            measurement = self.interface.get_channels(
                observable_names + [self.target_charge_PV]
            )

            charge_value = measurement[self.target_charge_PV]
            if self.is_inside_charge_bounds(charge_value):
                break
            else:
                print(f"charge value {charge_value} is outside bounds")

        return measurement

    def is_inside_charge_bounds(self, value):
        """test to make sure that charge value is within bounds"""
        if self.target_charge is not None:
            return (
                self.target_charge * (1.0 - self.fractional_charge_deviation)
                <= value
                <= self.target_charge * (1.0 + self.fractional_charge_deviation)
            )
        else:
            return True
