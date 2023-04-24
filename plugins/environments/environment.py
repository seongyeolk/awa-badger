from abc import ABC
from typing import List, Dict

import numpy as np
from pydantic import BaseModel, Field

from plugins.interfaces.interface import Interface


def validate_observable_names(func):
    def validate(environment, observable_names: List[str]):
        for name in observable_names:
            if name not in environment.observables:
                raise ValueError(f"Variable name: {name} not found in environment")
        return func(environment, observable_names)

    return validate


class Environment(BaseModel, ABC):
    name: str
    interface: Interface
    virtual_observables: List = Field([], description="custom virtual objectives "
                                                      "implemented "
                                                      "by this environment")
    params: Dict = Field({}, description="custom parameters")

    class Config:
        validate_assignment = True
        use_enum_values = True
        arbitrary_types_allowed = True

    @property
    def bounds(self) -> np.ndarray:
        return self.interface.bounds

    @property
    def variables(self) -> Dict[str, List[float]]:
        return self.interface.variables

    @property
    def observables(self) -> List:
        return sorted(self.interface.observables + self.virtual_observables)

    @property
    def variable_names(self) -> List[str]:
        """Returns a sorted list of variable names"""
        return list(sorted(self.variables.keys()))

    @property
    def n_variables(self) -> int:
        """Returns the number of variables"""
        return len(self.variables)

    @property
    def n_observables(self) -> int:
        """Returns the number of variables"""
        return len(self.observables)

    def get_variables(self, variable_names: List[str]) -> Dict:
        return self.interface.get_variables(variable_names)

    def set_variables(self, variable_inputs: Dict[str, float]):
        return self.interface.set_variables(variable_inputs)

    @validate_observable_names
    def get_observables(self, observable_names: List[str]) -> Dict:
        return self.interface.get_observables(observable_names)
