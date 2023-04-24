from typing import List, Dict

import pytest

from plugins.environments.environment import Environment
from plugins.interfaces.interface import Interface


class DummyInterface(Interface):
    def get_variables(self, variable_names: List[str]) -> Dict:
        pass

    def set_variables(self, variable_inputs: Dict[str, float]):
        pass

    def get_observables(self, observable_names: List[str]) -> Dict:
        pass


class TestEnvironment:
    def test_environment_creation(self):
        variables = {"x": [0, 1], "y": [0, 2]}
        observables = ["z"]
        params = {"a": 1, "b": 2}

        interface = DummyInterface(name="test_interface", variables=variables,
                                   observables=observables)
        env = Environment(name="test_env", interface=interface, params=params)

        assert env.name == "test_env"
        assert env.interface == interface
        assert env.variables == variables
        assert env.observables == observables
        assert env.params == params

    def test_environment_creation_without_params(self):
        variables = {"x": [0, 1], "y": [0, 2]}
        observables = ["z"]

        interface = DummyInterface(name="test_interface", variables=variables,
                                   observables=observables)
        env = Environment(name="test_env", interface=interface)

        assert env.name == "test_env"
        assert env.interface == interface
        assert env.variables == variables
        assert env.observables == observables

    def test_interface_creation_with_invalid_bounds(self):
        variables = {"x": [1, 0], "y": [0, 2]}
        observables = ["z"]

        with pytest.raises(ValueError):
            DummyInterface(name="test_interface", variables=variables,
                           observables=observables)

    def test_get_variables(self):
        variables = {"x": [0, 1], "y": [0, 2]}
        observables = ["z"]
        params = {"a": 1, "b": 2}

        interface = DummyInterface(name="test_interface", variables=variables,
                                   observables=observables)
        env = Environment(name="test_env", interface=interface, params=params)

        # Test getting multiple variables
        env.get_variables(["x", "y"])

        # Test getting an invalid variable
        with pytest.raises(ValueError):
            env.get_variables(["x", "z"])
