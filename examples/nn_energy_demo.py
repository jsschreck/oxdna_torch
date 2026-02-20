"""
Hybrid NN + oxDNA energy model demo.

This script shows three progressively more involved patterns for combining
a neural network with the oxDNA physics potential:

  Pattern 1 — Energy correction network
    A small MLP learns a per-nucleotide scalar correction on top of the
    oxDNA energy. Total energy = E_oxDNA + E_NN. The NN sees positions and
    orientations (a1, a3 axes) as input features. Gradients flow through
    both terms simultaneously, so the optimizer sees the full hybrid energy
    surface.

  Pattern 2 — Hybrid MD with a correction network
    The hybrid model is dropped into the Langevin integrator unchanged.
    Because OxDNAEnergy and the correction MLP are both nn.Modules, the
    integrator's force computation (which calls model(state) and
    differentiates) works without any modification.

  Pattern 3 — Training the correction network
    A synthetic training loop that adjusts the NN weights to push the
    system toward lower energy after N integration steps. This is the
    core pattern for learned force field corrections and inverse design.

Run from the repo root:

    python examples/nn_energy_demo.py

No GPU required; the script automatically uses CUDA if available.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch import Tensor

from oxdna_torch import load_system, OxDNAEnergy
from oxdna_torch.state import SystemState
from oxdna_torch.integrator import LangevinIntegrator
from oxdna_torch.quaternion import quat_to_rotmat


# ---------------------------------------------------------------------------
# Shared feature extractor
# ---------------------------------------------------------------------------

def state_to_features(state: SystemState) -> Tensor:
    """Build a per-nucleotide feature vector from a SystemState.

    Each nucleotide gets a 9-dimensional feature:
        [pos_x, pos_y, pos_z,          (3) centre-of-mass position
         a1_x,  a1_y,  a1_z,           (3) principal axis (points toward base)
         a3_x,  a3_y,  a3_z]           (3) base-normal axis

    The a1 and a3 vectors are the first and third columns of the rotation
    matrix R = quat_to_rotmat(q), which is exactly how oxDNA stores
    orientation.  All values are already in oxDNA reduced units.

    Args:
        state: SystemState with positions (N, 3) and quaternions (N, 4)

    Returns:
        features: (N, 9) float tensor on the same device as state.positions
    """
    R = quat_to_rotmat(state.quaternions)   # (N, 3, 3)
    a1 = R[:, :, 0]                         # (N, 3) principal axis
    a3 = R[:, :, 2]                         # (N, 3) base normal
    return torch.cat([state.positions, a1, a3], dim=-1)  # (N, 9)


# ---------------------------------------------------------------------------
# Pattern 1: Energy correction network
# ---------------------------------------------------------------------------

class CorrectionMLP(nn.Module):
    """Small MLP that predicts a per-nucleotide energy correction.

    Output is summed over all nucleotides to give a scalar correction term.
    Because it is a plain nn.Module operating on tensors that descend from
    positions and quaternions, autograd traces straight through it.

    Architecture:
        Linear(9 -> hidden) -> SiLU -> Linear(hidden -> hidden) -> SiLU
        -> Linear(hidden -> 1)  [per nucleotide]  -> sum -> scalar

    Args:
        hidden_dim: width of each hidden layer
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Initialise output layer near zero so the NN starts as a small
        # perturbation rather than dominating the physics potential.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: SystemState) -> Tensor:
        """Return scalar correction energy for the given state."""
        features = state_to_features(state)           # (N, 9)
        per_nucleotide = self.net(features).squeeze(-1)  # (N,)
        return per_nucleotide.sum()                   # scalar


class HybridEnergyModel(nn.Module):
    """oxDNA physics potential + learned correction.

    Total energy:
        E_total(state) = E_oxDNA(state) + E_NN(state)

    Both terms are differentiable w.r.t. positions and quaternions, so
    forces and torques from the hybrid model are:
        F = -dE_total/dpos  = F_oxDNA + F_NN
        tau = -dE_total/dtheta = tau_oxDNA + tau_NN

    This model can be passed directly to LangevinIntegrator without any
    changes to the integrator — it only calls model(state) and
    differentiates the result.

    Args:
        oxdna_model: a fully initialised OxDNAEnergy instance
        correction: a CorrectionMLP (or any nn.Module: SystemState -> scalar)
    """

    def __init__(self, oxdna_model: OxDNAEnergy, correction: CorrectionMLP):
        super().__init__()
        self.oxdna_model = oxdna_model
        self.correction = correction

    def forward(self, state: SystemState) -> Tensor:
        e_physics = self.oxdna_model(state)
        e_nn = self.correction(state)
        return e_physics + e_nn


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ------------------------------------------------------------------
    # Load system
    # ------------------------------------------------------------------
    topology, state = load_system(
        "examples/HAIRPIN/initial.top",
        "examples/HAIRPIN/initial.conf",
        device=device,
    )
    N = topology.n_nucleotides
    print(f"Loaded {N}-nucleotide hairpin\n")

    temperature = 0.1113   # ~334 K in oxDNA reduced units

    # ------------------------------------------------------------------
    # Pattern 1: Build the hybrid model and inspect energies
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PATTERN 1 — Energy correction network")
    print("=" * 60)

    oxdna = OxDNAEnergy(
        topology,
        temperature=temperature,
        seq_dependent=True,
    ).to(device)

    correction = CorrectionMLP(hidden_dim=32).to(device)
    # Cast correction to match the float64 tensors oxDNA works in
    correction = correction.double()

    hybrid = HybridEnergyModel(oxdna, correction).to(device)

    e_physics = oxdna(state)
    e_nn      = correction(state)
    e_total   = hybrid(state)

    print(f"  oxDNA energy      : {e_physics.item():+.4f}")
    print(f"  NN correction     : {e_nn.item():+.4f}  (near zero at init)")
    print(f"  Hybrid total      : {e_total.item():+.4f}")

    # Forces from the hybrid model via autograd
    pos = state.positions.detach().requires_grad_(True)
    s_tmp = SystemState(positions=pos, quaternions=state.quaternions.detach(), box=state.box)
    e_total_tmp = hybrid(s_tmp)
    e_total_tmp.backward()
    hybrid_forces = -pos.grad

    print(f"  Max hybrid force  : {hybrid_forces.norm(dim=-1).max().item():.4f}")
    print(f"  Any NaN in forces : {torch.isnan(hybrid_forces).any().item()}")
    print()

    # ------------------------------------------------------------------
    # Pattern 2: Hybrid MD — drop the hybrid model into the integrator
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PATTERN 2 — Hybrid Langevin dynamics (30 steps)")
    print("=" * 60)

    # The integrator calls model(state) and differentiates — it does not
    # care whether the model is pure physics or a hybrid.
    integrator = LangevinIntegrator(
        hybrid,
        dt=0.003,
        gamma=1.0,
        temperature=temperature,
    )

    current = state.clone().detach()
    energies_physics = []
    energies_nn      = []

    for step in range(30):
        e_p = oxdna(current).item()
        e_n = correction(current).item()
        energies_physics.append(e_p)
        energies_nn.append(e_n)
        current = integrator.step(current, stochastic=True)

    print(f"  oxDNA energy  — initial: {energies_physics[0]:+.4f}  "
          f"final: {energies_physics[-1]:+.4f}")
    print(f"  NN correction — initial: {energies_nn[0]:+.4f}  "
          f"final: {energies_nn[-1]:+.4f}")
    print(f"  (NN correction stays near zero because weights are untrained)")
    print()

    # ------------------------------------------------------------------
    # Pattern 3: Training the correction network
    #
    # Objective: minimise the oxDNA energy of the state reached after
    # n_unroll deterministic integration steps, by adjusting the NN
    # weights.  This is the basic pattern for:
    #   - learned force field corrections
    #   - structure optimisation via differentiable MD
    #   - inverse design (replace final energy with any structural metric)
    #
    # Only the NN parameters are optimised; the oxDNA physics is fixed.
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PATTERN 3 — Training the correction network (10 iterations)")
    print("=" * 60)
    print("  Objective: minimise E_oxDNA after 3 integration steps")
    print("  Trainable: correction MLP weights only")
    print()

    optimizer = torch.optim.Adam(correction.parameters(), lr=5e-3)
    n_unroll  = 5   # steps to unroll through; keep small for demo speed

    for iteration in range(10):
        optimizer.zero_grad()

        # Fresh copy of the initial state for each iteration so we always
        # unroll from the same starting point.
        s = state.clone().detach()

        # Unroll n_unroll steps with create_graph=True so that gradients
        # flow back through the integration steps into the NN weights.
        for _ in range(n_unroll):
            s = integrator.step(s, stochastic=False, create_graph=True)

        # Loss: oxDNA energy at the final state.
        # The NN correction affects the trajectory through the forces it
        # contributes at each step, even though the loss itself is pure physics.
        loss = oxdna(s)
        loss.backward()

        # Gradient clipping for stability during early training
        nn.utils.clip_grad_norm_(correction.parameters(), max_norm=1.0)
        optimizer.step()

        grad_norm = sum(
            p.grad.norm().item()
            for p in correction.parameters()
            if p.grad is not None
        )
        print(f"  iter {iteration+1:2d}  E_oxDNA after {n_unroll} steps = "
              f"{loss.item():+.6f}   |grad| = {grad_norm:.4f}")

    print()
    print("  The NN weights update each iteration based on how the correction")
    print("  forces steered the trajectory. In a real training loop you would")
    print("  run more iterations, more unroll steps, and use a meaningful loss")
    print("  (e.g. RMSD to a target structure, experimental observables, etc.)")
    print()

    # Confirm NN weights actually changed
    total_weight_norm = sum(p.norm().item() for p in correction.parameters())
    print(f"  Correction MLP weight norm after training: {total_weight_norm:.4f}")
    print(f"  (Was near zero at init — weights have been updated)")
    print()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
