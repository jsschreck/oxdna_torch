"""
GPU/CUDA tests for oxdna_torch.

All tests are skipped if CUDA is not available.
"""

import torch
import pytest

from oxdna_torch import load_system, OxDNAEnergy
from oxdna_torch.state import SystemState
from oxdna_torch.integrator import LangevinIntegrator

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

HAIRPIN_TOP = "examples/HAIRPIN/initial.top"
HAIRPIN_CONF = "examples/HAIRPIN/initial.conf"
T_334K = 334.0 / 3000.0


@requires_cuda
class TestCUDADevicePlacement:
    """Test that .to('cuda') moves all tensors correctly."""

    def test_model_buffers_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model = model.to('cuda')

        assert model.bonded_pairs.device.type == 'cuda'
        assert model.base_types.device.type == 'cuda'
        assert model.strand_ids.device.type == 'cuda'
        assert model.stacking_eps.device.type == 'cuda'
        assert model.hbond_eps_matrix.device.type == 'cuda'

    def test_topology_moved_with_model(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K)
        model = model.to('cuda')

        assert model.topology.bonded_pairs.device.type == 'cuda'
        assert model.topology.base_types.device.type == 'cuda'
        assert model.topology.strand_ids.device.type == 'cuda'
        assert model.topology.bonded_neighbors.device.type == 'cuda'

    def test_state_to_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        state = state.to('cuda')

        assert state.positions.device.type == 'cuda'
        assert state.quaternions.device.type == 'cuda'
        assert state.box.device.type == 'cuda'

    def test_load_system_with_device(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF, device='cuda')

        assert state.positions.device.type == 'cuda'
        assert state.quaternions.device.type == 'cuda'
        assert topology.bonded_pairs.device.type == 'cuda'
        assert topology.base_types.device.type == 'cuda'


@requires_cuda
class TestCUDAEnergy:
    """Test energy computation on GPU."""

    def test_energy_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=False)
        model = model.to('cuda')
        state = state.to('cuda')

        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        energy = model(s)
        assert energy.device.type == 'cuda'
        assert torch.isfinite(energy)

    def test_energy_matches_cpu(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)

        # CPU energy
        model_cpu = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=False)
        s_cpu = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        e_cpu = model_cpu(s_cpu).item()

        # GPU energy - create a fresh model to avoid shared topology
        topology2, state2 = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model_gpu = OxDNAEnergy(topology2, temperature=T_334K, seq_dependent=False)
        model_gpu = model_gpu.to('cuda')
        state2 = state2.to('cuda')
        s_gpu = SystemState(
            positions=state2.positions.detach(),
            quaternions=state2.quaternions.detach(),
            box=state2.box,
        )
        e_gpu = model_gpu(s_gpu).item()

        assert abs(e_cpu - e_gpu) < 1e-8, f"CPU={e_cpu}, GPU={e_gpu}"

    def test_energy_components_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model = model.to('cuda')
        state = state.to('cuda')

        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        components = model.energy_components(s)
        for name, val in components.items():
            assert val.device.type == 'cuda', f"{name} not on CUDA"
            assert torch.isfinite(val), f"{name} is not finite"

    def test_seq_dependent_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model = model.to('cuda')
        state = state.to('cuda')

        s = SystemState(
            positions=state.positions.detach(),
            quaternions=state.quaternions.detach(),
            box=state.box,
        )
        energy = model(s)
        assert energy.device.type == 'cuda'
        assert torch.isfinite(energy)


@requires_cuda
class TestCUDAGradients:
    """Test autograd works on GPU."""

    def test_forces_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K)
        model = model.to('cuda')
        state = state.to('cuda')

        forces = model.compute_forces(state)
        assert forces.device.type == 'cuda'
        assert not torch.isnan(forces).any()
        assert not torch.isinf(forces).any()

    def test_backward_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K)
        model = model.to('cuda')
        state = state.to('cuda')

        pos = state.positions.detach().requires_grad_(True)
        s = SystemState(positions=pos, quaternions=state.quaternions.detach(), box=state.box)
        energy = model(s)
        energy.backward()

        assert pos.grad is not None
        assert pos.grad.device.type == 'cuda'
        assert not torch.isnan(pos.grad).any()


@requires_cuda
class TestCUDATemperature:
    """Test temperature update works on GPU."""

    def test_update_temperature_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=True)
        model = model.to('cuda')

        # This should not raise a device mismatch
        model.update_temperature(370.0 / 3000.0)
        assert model.stacking_eps.device.type == 'cuda'

    def test_update_temperature_avg_seq_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K, seq_dependent=False)
        model = model.to('cuda')

        model.update_temperature(370.0 / 3000.0)
        assert model.stacking_eps.device.type == 'cuda'


@requires_cuda
class TestCUDAIntegrator:
    """Test integrator works on GPU."""

    def test_langevin_step_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K)
        model = model.to('cuda')
        state = state.to('cuda')

        integrator = LangevinIntegrator(model, dt=0.001, temperature=T_334K)
        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        s_new = integrator.step(s, stochastic=False)
        assert s_new.positions.device.type == 'cuda'
        assert s_new.quaternions.device.type == 'cuda'

    def test_backprop_through_time_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K)
        model = model.to('cuda')
        state = state.to('cuda')

        integrator = LangevinIntegrator(model, dt=0.001, temperature=T_334K)
        init_pos = state.positions.clone().detach().requires_grad_(True)
        s = SystemState(
            positions=init_pos,
            quaternions=state.quaternions.detach(),
            box=state.box,
        )

        for _ in range(3):
            s = integrator.step(s, stochastic=False)

        final_energy = model(s)
        final_energy.backward()

        assert init_pos.grad is not None
        assert init_pos.grad.device.type == 'cuda'
        assert not torch.isnan(init_pos.grad).any()

    def test_rollout_on_cuda(self):
        topology, state = load_system(HAIRPIN_TOP, HAIRPIN_CONF)
        model = OxDNAEnergy(topology, temperature=T_334K)
        model = model.to('cuda')
        state = state.to('cuda')

        integrator = LangevinIntegrator(model, dt=0.001, temperature=T_334K)
        s = SystemState(
            positions=state.positions.clone().detach(),
            quaternions=state.quaternions.clone().detach(),
            box=state.box,
        )
        trajectory = integrator.rollout(s, n_steps=5, stochastic=False, save_every=5)
        assert len(trajectory) == 2  # initial + final
        assert trajectory[-1].positions.device.type == 'cuda'
