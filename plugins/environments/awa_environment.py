from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from plugins.environments.environment import Environment, validate_observable_names
from plugins.interfaces.interface import Interface
from pydantic import Field, PositiveFloat


# from plugins.interfaces.awa_interface import AWAInterface
# from plugins.interfaces.camera import AWACamera


class AWAEnvironment(Environment):
    name = "awa_environment"
    interface: Interface  # = AWAInterface(camera_app=AWACamera(None))

    target_charge_PV: str = "AWAICTMon:Ch1"
    target_charge: Optional[PositiveFloat] = Field(
        None, description="magnitude of target charge in nC"
    )
    fractional_charge_deviation: PositiveFloat = Field(
        0.1, description="fractional deviation from target charge allowed"
    )

    roi_pvs: List[str] = [
        "13ARV1:ROI1:MinX_RBV",
        "13ARV1:ROI1:MinY_RBV",
        "13ARV1:ROI1:SizeX_RBV",
        "13ARV1:ROI1:SizeY_RBV",
    ]

    centroid_pvs: List[str] = [
        "13ARV1:Stats1:CentroidX_RBV",
        "13ARV1:Stats1:CentroidY_RBV",
        "13ARV1:Stats1:SigmaX_RBV",
        "13ARV1:Stats1:SigmaY_RBV",
    ]

    virtual_observables: List[str] = ["MAX_EXCURSION"]

    def __init__(
        self, varaible_file: str, observable_file: str, interface: Interface, **kwargs
    ):
        # process variable and observable files to det variables and observables
        variable_info = pd.read_csv(varaible_file).set_index("NAME")
        observable_info = pd.read_csv(observable_file).set_index("NAME").T

        _variables = variable_info[["MIN", "MAX"]].T.to_dict()
        _observables = list(observable_info.keys())

        for name in _variables:
            _variables[name] = [_variables[name]["MIN"], _variables[name]["MAX"]]

        super(AWAEnvironment, self).__init__(
            variables=_variables,
            observables=_observables,
            interface=interface,
            **kwargs,
        )

        self.observables = sorted(
            list(
                set(
                    self.roi_pvs
                    + self.centroid_pvs
                    + self.virtual_observables
                    + self.observables
                )
            )
        )

    @validate_observable_names
    def get_observables(self, observable_names: List[str]) -> Dict:
        """make measurements until charge range is within bounds"""

        while True:
            # add PV's to measure the maximum excursion of the beam outside the ROI
            if "MAX_EXCURSION" in observable_names:
                observable_names += self.roi_pvs
                observable_names += self.centroid_pvs

            if self.target_charge is not None:
                observable_names += [self.target_charge_PV]

            # remove duplicates
            observable_names = list(set(observable_names))

            # do measurements
            measurement = self.interface.get_channels(observable_names)

            # calculate the maximum excursion
            if "MAX_EXCURSION" in observable_names:
                measurement["MAX_EXCURSION"] = calculate_maximum_excursion(
                    *[measurement[name] for name in self.roi_pvs + self.centroid_pvs]
                )

            if self.target_charge is not None:
                charge_value = measurement[self.target_charge_PV]
                if self.is_inside_charge_bounds(charge_value):
                    break
                else:
                    print(f"charge value {charge_value} is outside bounds")
            else:
                break

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


def calculate_maximum_excursion(
    min_x, min_y, size_x, size_y, c_x, c_y, s_x, s_y, sigma_scale_factor=3
):
    # calculate the center of the ROI
    roi_c = np.array([min_x + size_x / 2.0, min_y + size_y / 2.0])
    beam_env_min = np.array(
        [c_x - sigma_scale_factor * s_x, c_y - sigma_scale_factor * s_y]
    )
    beam_env_max = np.array(
        [c_x + sigma_scale_factor * s_x, c_y + sigma_scale_factor * s_y]
    )

    roi_sizes = np.array([size_x, size_y])

    return np.max(
        (
            np.abs(beam_env_max - roi_c) - roi_sizes / 2,
            np.abs(roi_c - beam_env_min) - roi_sizes / 2,
        )
    )
