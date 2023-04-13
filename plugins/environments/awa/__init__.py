from typing import List

from badger import environment


class Environment(environment.Environment):
    name = "awa"
    variable_ranges = {"quad1": [0, 1], "quad2": [0, 5]}
    observables = ["YAG1:FWHMX", "YAG2:FWHMY"]

    def __init__(self, interface, params=None):
        params = params or {}

        self.interface = interface

        # target charge in nC
        self.target_charge = params.get("target_charge", 1.0)

        # n_samples
        self.n_samples = params.get("n_samples", 1)

    @staticmethod
    def list_vars() -> List[str]:
        return ["quad1", "quad2"]

    @staticmethod
    def list_obses() -> List[str]:
        return ["YAG1:FWHMX", "YAG2:FWHMY"]

    def _get_vrange(self, var: str):
        return self.variable_ranges[var]

    def _get_var(self, var: str):
        self.interface.get_value(var)

    def _set_var(self, var: str, x):
        self.interface.set_value(var, x)

    def _get_obs(self, obs: str):
        # parse observable string
        screen_name, obs_property = obs.split(":")

        return self.interface.get_camera_measurement(
            screen_name, self.target_charge, self.n_samples
        )[obs_property]
