from typing import Union, List, Dict, Optional

from pydantic import PositiveFloat, Field

from plugins.environments.environment import Environment
from plugins.interfaces.awa_interface import AWAInterface


class AWAEnvironment(Environment):
    name = "awa_environment"
    interface = AWAInterface()

    target_charge_PV: str = "AWAICTMon:Ch1"
    target_charge: Optional[PositiveFloat] = Field(
        None,
        description="magnitude of target charge in nC"
    )
    fractional_charge_deviation: PositiveFloat = Field(
        0.1,
        description="fractional deviation from target charge allowed"
    )

    def get_observables(self, observable_names: List[str]) -> Dict:
        """make measurements until charge range is within bounds"""

        while True:
            measurement = self.interface.get_observables(
                observable_names + [self.target_charge_PV]
            )

            charge_value = measurement[self.target_charge_PV]
            if self.is_inside_charge_bounds(charge_value):
                break
            else:
                print(f"charge value {charge_value} is outside bounds")

        return measurement

    def is_inside_charge_bounds(self, value):
        """ test to make sure that charge value is within bounds """
        if self.target_charge is not None:
            return self.target_charge * (
                    1.0 - self.fractional_charge_deviation) <= value\
                   <= self.target_charge * (1.0 + self.fractional_charge_deviation)
        else:
            return True
