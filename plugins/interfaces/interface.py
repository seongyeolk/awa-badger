from abc import ABC, abstractmethod
from typing import List, Dict
import pickle

import numpy as np
from pydantic import validator, Field, conlist, BaseModel


def validate_variable_names(func):
    def validate(interface, variable_names: List[str]):
        for name in variable_names:
            if name not in interface.variable_names:
                raise ValueError(f"Variable name: {name} not found in interface")
        return func(interface, variable_names)

    return validate


def validate_setpoints(func):
    def validate(interface, variable_inputs: Dict[str, float]):
        for name, value in variable_inputs.items():
            if name not in interface.variable_names:
                raise ValueError(f"Variable name: {name} not found in interface")

            lower = interface.variables[name][0]
            upper = interface.variables[name][1]

            if np.any(value > upper) or np.any(value < lower):
                raise ValueError(f"input point for {name} is outside interface "
                                 f"bounds")

        return func(interface, variable_inputs)

    return validate


def validate_observable_names(func):
    def validate(interface, observable_names: List[str]):
        for name in observable_names:
            if name not in interface.observables:
                raise ValueError(f"Observable name: {name} not found in interface")
        return func(interface, observable_names)

    return validate


class Interface(BaseModel, ABC):
    name: str
    variables: Dict[str, conlist(float, min_items=2, max_items=2)]
    observables: List
    params: Dict = Field({}, description="custom parameters")

    class Config:
        validate_assignment = True
        use_enum_values = True
        arbitrary_types_allowed = True

    @validator("variables")
    def validate_variable_bounds(cls, v):
        for name, item in v.items():
            lower = item[0]
            upper = item[1]

            if not upper > lower:
                raise ValueError("bounds for variable are not valid!")

        return v

    @property
    def bounds(self) -> np.ndarray:
        """
        Returns a bounds array (mins, maxs) of shape (2, n_variables)
        Arrays of lower and upper bounds can be extracted by:
            mins, maxs = vocs.bounds
        """
        return np.array([v for _, v in sorted(self.variables.items())]).T

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

    @abstractmethod
    @validate_variable_names
    def get_variables(self, variable_names: List[str]) -> Dict:
        pass

    @abstractmethod
    @validate_setpoints
    def set_variables(self, variable_inputs: Dict[str, float]):
        pass

    @abstractmethod
    @validate_observable_names
    def get_observables(self, observable_names: List[str]) -> Dict:
        pass
