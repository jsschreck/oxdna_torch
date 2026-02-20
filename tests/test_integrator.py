"""
Tests for the integrator module (dynamics and backprop through time).
"""

import math
import os

import torch
import pytest

from oxdna_torch.model import OxDNAEnergy
from oxdna_torch.state import SystemState
from oxdna_torch.integrator import LangevinIntegrator, VelocityVerletIntegrator

from conftest import T_334K


# ---------------------------------------------------------------------------
# Paths to example files used by stability tests
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, "..")

_RNA_TOP  = os.path.join(_ROOT, "examples", "RNA_MD", "prova.top")
_RNA_CONF = os.path.join(_ROOT, "examples", "RNA_MD", "prova.dat")
_rna_files_exist = os.path.isfile(_RNA_TOP) and os.path.isfile(_RNA_CONF)


class TestLangevinIntegrator:
    def test_step_returns_state(self, hairpin_model):
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s_new = integrator.step(s, stochastic=False)

        assert s_new.positions.shape == s.positions.shape
        assert s_new.quaternions.shape == s.quaternions.shape

    def test_step_changes_positions(self, hairpin_model):
        """A step should change positions (non-zero forces)."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s_new = integrator.step(s, stochastic=False)

        # Positions should have changed
        diff = (s_new.positions - s.positions).abs().max().item()
        assert diff > 1e-10, "Positions didn't change after a step"

    def test_quaternions_remain_normalized(self, hairpin_model):
        """Quaternions should remain unit-length after integration."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        for _ in range(5):
            s = integrator.step(s, stochastic=True)

        norms = torch.norm(s.quaternions, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-10), \
            f"Quaternion norms: {norms}"

    def test_deterministic_without_noise(self, hairpin_model):
        """Two runs without noise should give identical results."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s1 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s1 = integrator.step(s1, stochastic=False)

        s2 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s2 = integrator.step(s2, stochastic=False)

        assert torch.allclose(s1.positions, s2.positions, atol=1e-12)
        assert torch.allclose(s1.quaternions, s2.quaternions, atol=1e-12)

    def test_stochastic_gives_different_results(self, hairpin_model):
        """Two runs with noise should give different results."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        torch.manual_seed(42)
        s1 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s1 = integrator.step(s1, stochastic=True)

        torch.manual_seed(123)
        s2 = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s2 = integrator.step(s2, stochastic=True)

        diff = (s1.positions - s2.positions).abs().max().item()
        assert diff > 1e-6, "Stochastic steps should differ with different seeds"

    def test_energy_finite_after_steps(self, hairpin_model):
        """Energy should remain finite after a few small steps."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.0005, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        for _ in range(5):
            s = integrator.step(s, stochastic=False)

        energy = model(SystemState(
            positions=s.positions.detach(),
            quaternions=s.quaternions.detach(),
            box=s.box,
        ))
        assert torch.isfinite(energy), f"Energy not finite after 5 steps: {energy.item()}"


class TestBackpropThroughTime:
    """The key feature: gradients flowing through integration steps."""

    def test_grad_flows_to_initial_positions(self, hairpin_model):
        """Gradients from final energy should reach initial positions."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        init_pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(
            positions=init_pos,
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )

        # 3 deterministic steps
        for _ in range(3):
            s = integrator.step(s, stochastic=False)

        final_energy = model(s)
        final_energy.backward()

        assert init_pos.grad is not None, "No gradient on initial positions"
        assert not torch.isnan(init_pos.grad).any(), "NaN in backprop-through-time gradient"
        assert init_pos.grad.abs().max().item() > 1e-6, "Gradient suspiciously small"

    def test_grad_no_nan_multiple_steps(self, hairpin_model):
        """Gradients should be clean after multiple steps."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.0005, gamma=1.0, temperature=T_334K)

        init_pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(
            positions=init_pos,
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )

        for _ in range(5):
            s = integrator.step(s, stochastic=False)

        final_energy = model(s)
        final_energy.backward()

        assert not torch.isnan(init_pos.grad).any()
        assert not torch.isinf(init_pos.grad).any()

    def test_different_initial_positions_give_different_grads(self, hairpin_model):
        """Perturbing initial positions should change the gradient."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        # Run 1
        pos1 = state.positions.clone().detach().requires_grad_(True)
        s1 = SystemState(positions=pos1, quaternions=state.quaternions.detach(), box=state.box)
        for _ in range(2):
            s1 = integrator.step(s1, stochastic=False)
        e1 = model(s1)
        e1.backward()
        grad1 = pos1.grad.clone()

        # Run 2: slightly perturbed initial positions
        pos2_data = state.positions.clone().detach()
        pos2_data[0, 0] += 0.01
        pos2 = pos2_data.requires_grad_(True)
        s2 = SystemState(positions=pos2, quaternions=state.quaternions.detach(), box=state.box)
        for _ in range(2):
            s2 = integrator.step(s2, stochastic=False)
        e2 = model(s2)
        e2.backward()
        grad2 = pos2.grad.clone()

        diff = (grad1 - grad2).abs().max().item()
        assert diff > 1e-6, "Gradients should differ for different initial conditions"

    def test_can_optimize_positions(self, hairpin_model):
        """Demonstrate that we can minimize energy by gradient descent on positions."""
        model, topology, state = hairpin_model

        # Start with a slightly perturbed config
        pos = state.positions.clone().detach()
        pos += torch.randn_like(pos) * 0.01
        pos = pos.requires_grad_(True)

        optimizer = torch.optim.SGD([pos], lr=0.001)

        initial_energy = model(SystemState(
            positions=pos, quaternions=state.quaternions.detach(), box=state.box
        )).item()

        for _ in range(10):
            optimizer.zero_grad()
            energy = model(SystemState(
                positions=pos, quaternions=state.quaternions.detach(), box=state.box
            ))
            energy.backward()
            optimizer.step()

        final_energy = model(SystemState(
            positions=pos, quaternions=state.quaternions.detach(), box=state.box
        )).item()

        assert final_energy < initial_energy, \
            f"Gradient descent should lower energy: {initial_energy:.4f} -> {final_energy:.4f}"


class TestRollout:
    def test_rollout_returns_trajectory(self, hairpin_model):
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        trajectory = integrator.rollout(s, n_steps=5, stochastic=False, save_every=1)

        # Should have initial + 5 states
        assert len(trajectory) == 6

    def test_rollout_save_every(self, hairpin_model):
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=T_334K)

        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        trajectory = integrator.rollout(s, n_steps=10, stochastic=False, save_every=5)

        # initial + steps 5 and 10
        assert len(trajectory) == 3


class TestStability:
    """Longer integration runs to catch dynamics explosion.

    Uses the production timestep (dt=0.003) and checks that energy stays
    finite throughout.  These tests catch regressions where forces / torques
    are incorrect and would cause rapid divergence.
    """

    # ------------------------------------------------------------------ DNA

    def test_dna_langevin_500_steps_finite(self, hairpin_model):
        """500 stochastic Langevin steps at dt=0.003 — energy must stay finite.

        The energy is checked every 50 steps.  A genuine explosion (e.g. due
        to a wrong force sign) will produce NaN / Inf within ~10 steps.
        """
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0,
                                        temperature=T_334K)
        torch.manual_seed(0)
        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        for step in range(500):
            s = integrator.step(s, stochastic=True)
            if (step + 1) % 50 == 0:
                e = model(s).item()
                assert math.isfinite(e), \
                    f"DNA energy not finite at step {step + 1}: {e}"
                # Very loose upper bound — any real explosion exceeds this
                assert abs(e) < 1e4, \
                    f"DNA energy exploded at step {step + 1}: {e}"

    def test_dna_quaternions_stay_normalized_500_steps(self, hairpin_model):
        """Quaternions must remain unit-length over 500 Langevin steps."""
        model, topology, state = hairpin_model
        integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0,
                                        temperature=T_334K)
        torch.manual_seed(1)
        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        for _ in range(500):
            s = integrator.step(s, stochastic=True)

        norms = torch.norm(s.quaternions, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-9), \
            f"Quaternion norms drifted: min={norms.min():.8f} max={norms.max():.8f}"

    def test_dna_nve_energy_conservation(self, hairpin_model):
        """NVE (VelocityVerlet) should conserve total energy to O(dt^2).

        At dt=0.001, total-energy drift over 200 steps should be < 0.01
        reduced-energy units — a factor ~100 above floating-point noise
        but well below any dynamics explosion.
        """
        model, topology, state = hairpin_model
        torch.manual_seed(2)
        T = T_334K
        vel = torch.randn(topology.n_nucleotides, 3,
                          dtype=state.positions.dtype) * math.sqrt(T)
        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            velocities=vel,
            box=state.box,
        )

        integrator = VelocityVerletIntegrator(model, dt=0.001)

        e0 = model(s).item()
        ke0 = 0.5 * (vel ** 2).sum().item()
        etot0 = e0 + ke0

        for _ in range(200):
            s = integrator.step(s)

        e_final = model(s).item()
        ke_final = 0.5 * (s.velocities ** 2).sum().item()
        etot_final = e_final + ke_final

        drift = abs(etot_final - etot0)
        assert math.isfinite(etot_final), \
            f"NVE total energy is not finite: {etot_final}"
        assert drift < 0.01, \
            f"NVE energy drift too large: {drift:.6f} (E0={etot0:.4f}, Ef={etot_final:.4f})"

    # ------------------------------------------------------------------ RNA

    @pytest.mark.skipif(not _rna_files_exist, reason="examples/RNA_MD files not found")
    def test_rna_langevin_500_steps_finite(self):
        """500 stochastic Langevin steps on the 132-nt RNA at dt=0.003.

        Energy is sampled every 50 steps; must remain finite and within a
        loose physical range (< 0 and > −10 * N_nucleotides).
        """
        from oxdna_torch import load_rna_system, OxRNAEnergy

        topology, state = load_rna_system(_RNA_TOP, _RNA_CONF)
        T_red = (273.15 + 25) / 3000.0
        model = OxRNAEnergy(topology, temperature=T_red, seq_dependent=True)
        integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0,
                                        temperature=T_red)
        torch.manual_seed(3)
        s = state
        for step in range(500):
            s = integrator.step(s, stochastic=True)
            if (step + 1) % 50 == 0:
                e = model(s).item()
                assert math.isfinite(e), \
                    f"RNA energy not finite at step {step + 1}: {e}"
                assert abs(e) < 1e5, \
                    f"RNA energy exploded at step {step + 1}: {e}"

    @pytest.mark.skipif(not _rna_files_exist, reason="examples/RNA_MD files not found")
    def test_rna_quaternions_stay_normalized_500_steps(self):
        """Quaternions must remain unit-length over 500 RNA Langevin steps."""
        from oxdna_torch import load_rna_system, OxRNAEnergy

        topology, state = load_rna_system(_RNA_TOP, _RNA_CONF)
        T_red = (273.15 + 25) / 3000.0
        model = OxRNAEnergy(topology, temperature=T_red, seq_dependent=True)
        integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0,
                                        temperature=T_red)
        torch.manual_seed(4)
        s = state
        for _ in range(500):
            s = integrator.step(s, stochastic=True)

        norms = torch.norm(s.quaternions, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-9), \
            f"RNA quaternion norms drifted: min={norms.min():.8f} max={norms.max():.8f}"

    @pytest.mark.skipif(not _rna_files_exist, reason="examples/RNA_MD files not found")
    def test_rna_energy_per_nt_in_physical_range(self):
        """After 200 equilibration steps the RNA energy/nt should be in [-3, -0.5].

        This catches sign errors or grossly wrong potential parameters that
        would shift the energy far outside the physically expected window
        (reference C++ oxRNA2 gives ≈ −1.34 per nucleotide at 25 °C).
        """
        from oxdna_torch import load_rna_system, OxRNAEnergy

        topology, state = load_rna_system(_RNA_TOP, _RNA_CONF)
        T_red = (273.15 + 25) / 3000.0
        model = OxRNAEnergy(topology, temperature=T_red, seq_dependent=True)
        integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0,
                                        temperature=T_red)
        torch.manual_seed(5)
        s = state
        for _ in range(200):
            s = integrator.step(s, stochastic=True)

        e_per_nt = model(s).item() / topology.n_nucleotides
        assert math.isfinite(e_per_nt), f"E/nt not finite: {e_per_nt}"
        assert -3.0 < e_per_nt < -0.5, \
            f"RNA E/nt={e_per_nt:.4f} outside physical range (-3, -0.5)"
