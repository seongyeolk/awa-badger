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

    @validate_observable_names
    def get_observables(self, observable_names: List[str]) -> Dict:
        """make measurements until charge range is within bounds"""

        while True:
            if self.target_charge is not None:
                observable_names += [self.target_charge_PV]

            # remove duplicates
            observable_names = list(set(observable_names))

            # if a screen measurement is involved
            base_observable_names = [ele.split(":")[0] for ele in observable_names]
            screen_name = "13ARV1"

            if screen_name in base_observable_names:
                measurement = self.get_screen_measurement(screen_name, observable_names)
            else:
                # otherwise do normal epics communication
                measurement = self.interface.get_channels(observable_names)

            if self.target_charge is not None:
                charge_value = measurement[self.target_charge_PV]
                if self.is_inside_charge_bounds(charge_value):
                    break
                else:
                    print(f"charge value {charge_value} is outside bounds")
            else:
                break

        return measurement

    def get_screen_measurement(self, screen_name, extra_pvs=None):
        roi_readbacks = [
            "ROI1:MinX_RBV",
            "ROI1:MinY_RBV",
            "ROI1:SizeX_RBV",
            "ROI1:SizeY_RBV",
        ]
        centroid_readbacks = [
            "Stats1:CentroidX_RBV",
            "Stats1:CentroidY_RBV",
            "Stats1:SigmaX_RBV",
            "Stats1:SigmaY_RBV",
        ]

        extra_pvs = extra_pvs or []

        # construct list of all PVs necessary for measurement
        observation_pvs = [
            f"{screen_name}:{pv_name}" for pv_name in roi_readbacks + centroid_readbacks
        ] + extra_pvs

        # get rid of duplicate PVs
        observation_pvs = list(set(observation_pvs))

        # do measurement and sort data
        measurement = self.interface.get_channels(observation_pvs)
        roi_data = np.array([measurement[ele] for ele in observation_pvs[:4]])
        beam_data = np.array([measurement[ele] for ele in observation_pvs[4:]])

        # validate measurement
        ll_penalty, ur_penalty = validate_screen_measurement(
            roi_data[:2], roi_data[2:], beam_data[:2], beam_data[2:]
        )

        # remove data that is invalid
        if ll_penalty > 0 or ur_penalty > 0:
            for name in observation_pvs[4:]:
                measurement[name] = np.nan

        measurement["LL_BOX_PENALTY"] = ll_penalty
        measurement["UR_BOX_PENALTY"] = ur_penalty

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


def validate_screen_measurement(
    roi_min_pt, roi_sizes, beam_centroid, beam_rms, sigma_scale_factor=3
):
    # inscribe a circle inside the roi_rectangle
    roi_c = roi_min_pt + roi_sizes / 2.0
    roi_radius = np.min(roi_sizes) / 2.0

    # calculate the beam rectangle
    beam_ll = beam_centroid - beam_rms * sigma_scale_factor
    beam_ur = beam_centroid + beam_rms * sigma_scale_factor

    # calculate the distance from the center of the roi circle to each corner of the
    # beam rectangle
    ll_distance = np.linalg.norm((beam_ll - roi_c))
    ur_distance = np.linalg.norm((beam_ur - roi_c))

    ll_penalty = ll_distance - roi_radius  # if > 0 then the measurement is invalid
    ur_penalty = ur_distance - roi_radius  # if > 0 then the measurement is invalid

    return ll_penalty, ur_penalty


def rectangle_union_area(llc1, s1, llc2, s2):
    # Compute the intersection of the two rectangles
    x1, y1 = llc1
    x2, y2 = llc2
    w1, h1 = s1
    w2, h2 = s2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_area = x_overlap * y_overlap

    # Compute the areas of the two rectangles
    rect1_area = w1 * h1
    rect2_area = w2 * h2

    # Compute the area of the union
    union_area = rect1_area + rect2_area - overlap_area

    return union_area
