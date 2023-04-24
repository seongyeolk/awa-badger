from plugins.environments.awa_environment import AWAEnvironment


class TestAWAEnvironment:
    def test_init(self):
        AWAEnvironment()

    # Test to ensure the is_inside_charge_bounds function is working as expected
    def test_is_inside_charge_bounds(self):
        env = AWAEnvironment(target_charge=10.0, fractional_charge_deviation=0.1)
        assert env.is_inside_charge_bounds(10.0)
        assert env.is_inside_charge_bounds(11.0)
        assert env.is_inside_charge_bounds(9.0)
        assert not env.is_inside_charge_bounds(12.0)
        assert not env.is_inside_charge_bounds(8.0)

    # Test all of the observables
    def test_get_observables_returns_expected_keys(self):
        env = AWAEnvironment()
        observables = env.observables
        result = env.get_observables(observables)
        assert all(key in result for key in observables)
        assert env.target_charge_PV in result

