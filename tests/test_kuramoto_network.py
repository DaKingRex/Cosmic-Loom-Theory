"""
Tests for KuramotoNetwork — the coupled-oscillator synchronization primitive
(CLT Phase 3.1).
"""

import numpy as np
import pytest


class TestKuramotoBasics:
    """Construction, state, stepping, reset."""

    def test_import_and_init(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=50, seed=0)
        assert net.n == 50
        assert net.phases.shape == (50,)
        assert net.omega.shape == (50,)

    def test_step_advances_time(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=50, seed=0)
        net.step(10)
        assert net.time > 0.0
        assert len(net.coherence_history) == 11

    def test_run_duration(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=50, dt=0.01, seed=0)
        net.run(1.0)
        assert len(net.coherence_history) == 101

    def test_reset(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=50, seed=0)
        net.run(1.0)
        net.reset(seed=1)
        assert net.time == 0.0
        assert len(net.coherence_history) == 1

    def test_invalid_coupling_mode(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        with pytest.raises(ValueError):
            KuramotoNetwork(coupling_mode="bogus")


class TestOrderParameter:
    """Order parameter and its dependence on coupling."""

    def test_bounds(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=100, seed=0)
        net.run(2.0)
        assert 0.0 <= net.order_parameter() <= 1.0

    def test_increases_with_coupling(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        weak = KuramotoNetwork(n_oscillators=200, coupling=0.1, gamma=0.5, seed=0)
        strong = KuramotoNetwork(n_oscillators=200, coupling=6.0, gamma=0.5, seed=0)
        weak.run(25.0)
        strong.run(25.0)
        assert strong.order_parameter() > weak.order_parameter()

    def test_synchronize_raises_order(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=100, seed=0)
        before = net.order_parameter()
        net.synchronize(1.0)
        assert net.order_parameter() > before


class TestRegimes:
    """Regime coverage across coupling strengths."""

    def test_incoherent_is_chaos(self):
        from simulations.emergence.kuramoto_network import create_incoherent_network
        net = create_incoherent_network(seed=1)
        net.run(20.0)
        assert net.map_to_er_space()["regime"] == "chaos"

    def test_hypersync_is_rigidity(self):
        from simulations.emergence.kuramoto_network import create_hypersync_network
        net = create_hypersync_network(seed=1)
        net.run(20.0)
        assert net.map_to_er_space()["regime"] == "rigidity"


class TestERMapping:
    """éR phase-space readout."""

    def test_keys(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        er = KuramotoNetwork(n_oscillators=50, seed=0).map_to_er_space()
        for key in ("energy_present", "frequency", "energy_resistance",
                    "coherence", "order_parameter", "regime"):
            assert key in er

    def test_er_formula(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=50, seed=0)
        net.run(2.0)
        er = net.map_to_er_space()
        assert np.isclose(er["energy_resistance"],
                          er["energy_present"] / er["frequency"] ** 2, rtol=1e-3)


class TestCriticalCoupling:
    """Analytic critical coupling."""

    def test_scales_with_gamma(self):
        from simulations.emergence.kuramoto_network import critical_coupling
        assert critical_coupling(1.0) > critical_coupling(0.5)

    def test_positive(self):
        from simulations.emergence.kuramoto_network import critical_coupling
        assert critical_coupling(0.5) > 0.0


class TestLatticeMode:
    """Nearest-neighbor lattice coupling runs."""

    def test_lattice_runs(self):
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        net = KuramotoNetwork(n_oscillators=64, coupling=2.0,
                              coupling_mode="lattice", seed=0)
        net.run(3.0)
        assert 0.0 <= net.order_parameter() <= 1.0


class TestPresets:
    """Preset factory functions."""

    def test_incoherent_preset(self):
        from simulations.emergence.kuramoto_network import create_incoherent_network
        assert create_incoherent_network(seed=0).coupling < 1.0

    def test_partial_sync_preset(self):
        from simulations.emergence.kuramoto_network import (
            create_partial_sync_network, critical_coupling,
        )
        net = create_partial_sync_network(seed=0)
        assert net.coupling > critical_coupling(net.gamma)

    def test_hypersync_preset(self):
        from simulations.emergence.kuramoto_network import create_hypersync_network
        assert create_hypersync_network(seed=0).coupling >= 5.0
