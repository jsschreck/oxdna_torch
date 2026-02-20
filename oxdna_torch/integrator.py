"""
Differentiable integrators for oxDNA dynamics.

Supports backpropagation through time via:
1. Standard autograd (for short trajectories)
2. Gradient checkpointing (for long trajectories, trades compute for memory)

The integrator computes forces via autograd (F = -dE/dpos) and torques via
autograd (tau = -dE/d(orientation)), then propagates positions/velocities
using velocity-Verlet + Langevin thermostat, matching the reference oxDNA
MD_CPUBackend + LangevinThermostat implementation.

Algorithm (per step):
  1. First half-kick: vel += F*dt/2, L += tau*dt/2
  2. Drift: pos += vel*dt, orientation updated from L
  3. Compute forces/torques at new positions
  4. Second half-kick: vel += F*dt/2, L += tau*dt/2
  5. Langevin thermostat: vel = c1*vel + c2*xi_v, L = c1r*L + c2r*xi_L

Torque derivation from quaternion gradients:
  For rotation parametrized by unit quaternion q, the torque around lab axis e_i is:
    tau_i = -dE/dtheta_i = -(dE/dq) · (dq/dtheta_i)
  where dq/dtheta_i = 0.5 * [0, e_i] ⊗ q  (generator of rotation).
  Verified against finite differences.
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, List, Tuple
import math

from .state import SystemState
from .model import OxDNAEnergy
from .quaternion import quat_normalize, quat_multiply, quat_angular_velocity_update


def _torque_from_quat_grad(quaternions: Tensor, dEdq: Tensor) -> Tensor:
    """Convert quaternion-space energy gradient to lab-frame torque.

    For a rotation parameterized by unit quaternion q = [w, x, y, z], the
    torque around lab axis e_i is:

        tau_i = -dE/dtheta_i = -(dE/dq) · (dq/dtheta_i)

    where dq/dtheta_i = 0.5 * [0, e_i] ⊗ q.

    Expanding [0, e_i] ⊗ [w, x, y, z] for each axis gives a closed form:
        dq/dtheta_x = 0.5 * [-x,  w, -z,  y]
        dq/dtheta_y = 0.5 * [-y,  z,  w, -x]
        dq/dtheta_z = 0.5 * [-z, -y,  x,  w]

    So tau_i = -(dEdq · dq/dtheta_i), fully vectorized.

    Fully vectorized - no Python loop. Verified against finite differences.

    Args:
        quaternions: (N, 4) unit quaternions [w, x, y, z]
        dEdq:        (N, 4) dE/dq (positive sign, autograd gradient)

    Returns:
        (N, 3) torque in lab frame
    """
    w, x, y, z = quaternions.unbind(-1)          # (N,) each
    gw, gx, gy, gz = dEdq.unbind(-1)             # (N,) each

    # tau_x = -(dEdq · dq/dtheta_x) = -0.5*(-gw*x + gx*w - gy*z + gz*y)
    # tau_y = -(dEdq · dq/dtheta_y) = -0.5*(-gw*y + gx*z + gy*w - gz*x)
    # tau_z = -(dEdq · dq/dtheta_z) = -0.5*(-gw*z - gx*y + gy*x + gz*w)
    tau_x = -0.5 * (-gw * x + gx * w - gy * z + gz * y)
    tau_y = -0.5 * (-gw * y + gx * z + gy * w - gz * x)
    tau_z = -0.5 * (-gw * z - gx * y + gy * x + gz * w)

    return torch.stack([tau_x, tau_y, tau_z], dim=-1)


class LangevinIntegrator(nn.Module):
    """Langevin dynamics integrator for oxDNA.

    Implements inertial Langevin dynamics (velocity-Verlet + Langevin
    thermostat), matching the reference oxDNA MD_CPUBackend:
      - Forces and torques via autograd from the energy model
      - Translational and rotational momenta (vel, L) evolved explicitly
      - Thermostat: vel *= c1 + c2*noise (same for L with rot. coefficients)
      - D_rot = 3 * D_trans  (as in oxDNA LangevinThermostat.cpp)

    Args:
        energy_model: OxDNAEnergy module for computing potential energy
        dt: integration timestep in oxDNA reduced units
        gamma: translational friction coefficient (gamma_trans)
        temperature: temperature in oxDNA reduced units (T_K / 3000)
        mass: particle mass (default 1.0, oxDNA convention)
        inertia: moment of inertia (default 1.0, oxDNA convention)
    """

    def __init__(
        self,
        energy_model: OxDNAEnergy,
        dt: float = 0.003,
        gamma: float = 1.0,
        temperature: float = 0.1,
        mass: float = 1.0,
        inertia: float = 1.0,
    ):
        super().__init__()
        self.energy_model = energy_model
        self.dt = dt
        self.gamma = gamma
        self.temperature = temperature
        self.mass = mass
        self.inertia = inertia

        # Rotational friction: gamma_rot = gamma_trans / 3
        # (D_rot = 3*D_trans, so gamma_rot = T/D_rot = T/(3*D_trans) = gamma_trans/3)
        self.gamma_rot = gamma / 3.0

        # Langevin thermostat coefficients (BAOAB-style, applied at end of step)
        # c1 = exp(-gamma * dt),  c2 = sqrt((1 - c1^2) * T)  [mass=1 convention]
        # Using half-step form used by oxDNA builtin Langevin:
        #   c1 = exp(-gamma * dt/2), c2 = sqrt((1-c1^2)*T)
        self._c1_trans = math.exp(-gamma * dt / 2.0)
        self._c2_trans = math.sqrt((1.0 - self._c1_trans ** 2) * temperature)

        self._c1_rot = math.exp(-self.gamma_rot * dt / 2.0)
        self._c2_rot = math.sqrt((1.0 - self._c1_rot ** 2) * temperature)

    def _compute_forces_and_torques(
        self,
        state: SystemState,
        create_graph: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """Compute forces (N,3) and lab-frame torques (N,3) via autograd."""
        pos_grad = state.positions.detach().requires_grad_(True)
        quat_grad = state.quaternions.detach().requires_grad_(True)

        state_grad = SystemState(
            positions=pos_grad,
            quaternions=quat_grad,
            box=state.box,
        )

        energy = self.energy_model(state_grad)

        grads = torch.autograd.grad(
            energy, [pos_grad, quat_grad],
            create_graph=create_graph,
        )

        forces = -grads[0]   # (N, 3)  F = -dE/dpos
        dEdq = grads[1]      # (N, 4)  dE/dq (positive sign)

        # Torque in lab frame: tau_i = -dE/dtheta_i (already negated inside)
        torques = _torque_from_quat_grad(quat_grad, dEdq)  # (N, 3)

        return forces, torques

    def step(
        self,
        state: SystemState,
        stochastic: bool = True,
        create_graph: bool = False,
    ) -> SystemState:
        """Perform one inertial Langevin step.  Returns the new SystemState."""
        new_state, _ = self._step_with_forces(state, stochastic=stochastic,
                                               create_graph=create_graph)
        return new_state

    def _step_with_forces(
        self,
        state: SystemState,
        stochastic: bool = True,
        create_graph: bool = False,
        _forces: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[SystemState, Tuple[Tensor, Tensor]]:
        """Like step(), but also returns forces at the new state for reuse.

        Args:
            state: current SystemState
            stochastic: whether to apply Langevin noise
            create_graph: whether to build autograd graph (for BPTT)
            _forces: (forces, torques) pre-computed at current state from the
                     previous call - avoids one redundant energy evaluation.

        Returns:
            (new_state, (forces, torques)) at the new state
        """
        dt = self.dt
        positions = state.positions
        quaternions = state.quaternions

        # Initialize velocities/angular momenta if not present
        if state.velocities is None:
            velocities = torch.zeros_like(positions)
        else:
            velocities = state.velocities

        if state.ang_velocities is None:
            ang_momenta = torch.zeros(positions.shape[0], 3,
                                     dtype=positions.dtype, device=positions.device)
        else:
            ang_momenta = state.ang_velocities  # stored as L (angular momentum)

        # === Step 1: Forces/torques at current positions ===
        # Reuse from previous step if provided (saves one energy evaluation per step)
        if _forces is not None:
            forces, torques = _forces
        else:
            forces, torques = self._compute_forces_and_torques(state, create_graph=create_graph)

        # === Step 2: First half-kick (B step) ===
        velocities = velocities + forces * (dt * 0.5 / self.mass)
        ang_momenta = ang_momenta + torques * (dt * 0.5 / self.inertia)

        # === Step 3: Langevin thermostat on velocities (O step, first half) ===
        if stochastic:
            c1, c2 = self._c1_trans, self._c2_trans
            velocities = c1 * velocities + c2 * torch.randn_like(velocities)

            c1r, c2r = self._c1_rot, self._c2_rot
            ang_momenta = c1r * ang_momenta + c2r * torch.randn_like(ang_momenta)

        # === Step 4: Drift (A step) - update positions and orientations ===
        new_positions = positions + velocities * dt

        # Apply PBC wrapping to positions
        new_box = state.box
        if new_box is not None:
            new_positions = new_positions - new_box * torch.floor(new_positions / new_box)

        # Update orientation: rotate by angular velocity omega = L / I for dt
        # Uses the exact Rodrigues rotation (as in oxDNA _first_step)
        omega = ang_momenta / self.inertia  # (N, 3) angular velocity
        new_quaternions = quat_angular_velocity_update(quaternions, omega, dt)

        # === Step 5: Compute forces/torques at new positions ===
        new_state_tmp = SystemState(
            positions=new_positions,
            quaternions=new_quaternions,
            box=new_box,
        )
        new_forces, new_torques = self._compute_forces_and_torques(
            new_state_tmp, create_graph=create_graph
        )

        # === Step 6: Second half-kick (B step) ===
        velocities = velocities + new_forces * (dt * 0.5 / self.mass)
        ang_momenta = ang_momenta + new_torques * (dt * 0.5 / self.inertia)

        # === Step 7: Langevin thermostat (O step, second half) ===
        if stochastic:
            velocities = c1 * velocities + c2 * torch.randn_like(velocities)
            ang_momenta = c1r * ang_momenta + c2r * torch.randn_like(ang_momenta)

        new_state = SystemState(
            positions=new_positions,
            quaternions=new_quaternions,
            velocities=velocities,
            ang_velocities=ang_momenta,
            box=new_box,
        )
        # Return forces at new positions - reused as step N+1's initial forces
        return new_state, (new_forces, new_torques)

    def rollout(
        self,
        state: SystemState,
        n_steps: int,
        stochastic: bool = True,
        checkpoint_every: int = 0,
        save_every: int = 1,
        create_graph: bool = False,
    ) -> List[SystemState]:
        """Integrate for multiple steps, returning trajectory.

        Args:
            state: initial SystemState
            n_steps: number of integration steps
            stochastic: whether to add Langevin noise
            checkpoint_every: if > 0, use gradient checkpointing every N steps
                (saves memory at cost of recomputation during backward pass)
            save_every: save state every N steps (1 = save all)
            create_graph: build autograd graph for backprop through time

        Returns:
            List of SystemState snapshots along the trajectory
        """
        trajectory = [state]

        if checkpoint_every > 0:
            # Use gradient checkpointing for memory efficiency
            current_state = state
            for start in range(0, n_steps, checkpoint_every):
                end = min(start + checkpoint_every, n_steps)
                n_chunk = end - start
                device = current_state.positions.device

                # Checkpoint this chunk
                current_state = grad_checkpoint(
                    self._integrate_chunk,
                    current_state.positions,
                    current_state.quaternions,
                    current_state.velocities if current_state.velocities is not None
                        else torch.zeros_like(current_state.positions),
                    current_state.ang_velocities if current_state.ang_velocities is not None
                        else torch.zeros(current_state.n_nucleotides, 3,
                                        dtype=current_state.positions.dtype, device=device),
                    current_state.box if current_state.box is not None else torch.zeros(3, device=device),
                    torch.tensor(n_chunk, device=device),
                    torch.tensor(stochastic, device=device),
                    use_reentrant=False,
                )

                if (start + checkpoint_every) % save_every == 0 or end == n_steps:
                    trajectory.append(current_state)
        else:
            current_state = state
            cached_forces = None  # carry forces from step N → step N+1
            for i in range(n_steps):
                current_state, cached_forces = self._step_with_forces(
                    current_state, stochastic=stochastic,
                    create_graph=create_graph, _forces=cached_forces,
                )
                if (i + 1) % save_every == 0:
                    trajectory.append(current_state)

        return trajectory

    def _integrate_chunk(
        self,
        positions: Tensor,
        quaternions: Tensor,
        velocities: Tensor,
        ang_momenta: Tensor,
        box: Tensor,
        n_steps_tensor: Tensor,
        stochastic_tensor: Tensor,
    ) -> SystemState:
        """Integrate a chunk of steps (used with gradient checkpointing).

        Args are tensors to work with torch.utils.checkpoint.
        """
        n_steps = int(n_steps_tensor.item())
        stochastic = bool(stochastic_tensor.item())
        box_actual = box if box.any() else None

        state = SystemState(
            positions=positions,
            quaternions=quaternions,
            velocities=velocities,
            ang_velocities=ang_momenta,
            box=box_actual,
        )

        cached_forces = None
        for _ in range(n_steps):
            state, cached_forces = self._step_with_forces(
                state, stochastic=stochastic, _forces=cached_forces
            )

        return state


class VelocityVerletIntegrator(nn.Module):
    """Velocity-Verlet integrator for NVE dynamics.

    Useful for testing energy conservation and validating forces.
    Does NOT add Langevin noise.

    WARNING: This integrator handles only translational degrees of freedom
    properly with velocity-Verlet. Quaternion integration uses a simplified
    first-order method. For production dynamics, use LangevinIntegrator.

    Args:
        energy_model: OxDNAEnergy module
        dt: timestep
        mass: particle mass
    """

    def __init__(
        self,
        energy_model: OxDNAEnergy,
        dt: float = 0.001,
        mass: float = 1.0,
    ):
        super().__init__()
        self.energy_model = energy_model
        self.dt = dt
        self.mass = mass

    def step(self, state: SystemState) -> SystemState:
        """Perform one velocity-Verlet step.

        Args:
            state: current SystemState (must have velocities)

        Returns:
            New SystemState
        """
        assert state.velocities is not None, "VelocityVerlet requires velocities"

        dt = self.dt
        positions = state.positions
        velocities = state.velocities
        quaternions = state.quaternions

        # Compute forces at current position
        pos_grad = positions.detach().requires_grad_(True)
        state_tmp = SystemState(positions=pos_grad, quaternions=quaternions, box=state.box)
        energy = self.energy_model(state_tmp)
        forces = -torch.autograd.grad(energy, pos_grad, create_graph=True)[0]

        # Half-step velocity update
        velocities_half = velocities + 0.5 * dt * forces / self.mass

        # Full-step position update
        new_positions = positions + dt * velocities_half

        # Apply PBC
        if state.box is not None:
            new_positions = new_positions - state.box * torch.floor(new_positions / state.box)

        # Compute forces at new position
        new_pos_grad = new_positions.detach().requires_grad_(True)
        state_tmp2 = SystemState(
            positions=new_pos_grad, quaternions=quaternions, box=state.box
        )
        energy2 = self.energy_model(state_tmp2)
        new_forces = -torch.autograd.grad(energy2, new_pos_grad, create_graph=True)[0]

        # Full velocity update
        new_velocities = velocities_half + 0.5 * dt * new_forces / self.mass

        return SystemState(
            positions=new_positions,
            quaternions=quaternions,  # Not updated in this simple version
            velocities=new_velocities,
            ang_velocities=state.ang_velocities,
            box=state.box,
        )
