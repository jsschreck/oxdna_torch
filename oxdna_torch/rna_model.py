"""
OxRNA energy model.

Implements the full oxRNA1 potential as a PyTorch nn.Module:
  E = E_FENE + E_bonded_excl + E_nonbonded_excl
    + E_stack + E_hbond + E_cross_stack + E_coaxial_stack

The model is a drop-in replacement for OxDNAEnergy when working with RNA:
  - Uses RNATopology instead of Topology
  - Computes all interactions with RNA-specific geometry and constants
  - Compatible with LangevinIntegrator (same SystemState interface)
  - Supports set_nl_skin(), set_nl_backend(), compile()

Usage::

    from oxdna_torch.rna_io import load_rna_system
    from oxdna_torch.rna_model import OxRNAEnergy

    topology, state = load_rna_system('hairpin.top', 'hairpin.conf')
    model = OxRNAEnergy(topology, temperature=0.1113)
    energy = model(state)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict

from .state import SystemState
from .rna_topology import RNATopology
from . import rna_constants as RC
from .pairs import find_nonbonded_pairs, _torchmdnet_available
from .interactions.rna_fene import rna_fene_energy
from .interactions.rna_excl_vol import (
    rna_bonded_excluded_volume_energy,
    rna_nonbonded_excluded_volume_energy,
)
from .interactions.rna_stacking import rna_stacking_energy
from .interactions.rna_hbond import rna_hbond_energy
from .interactions.rna_cross_stacking import rna_cross_stacking_energy
from .interactions.rna_coaxial_stacking import rna_coaxial_stacking_energy


class OxRNAEnergy(nn.Module):
    """Full oxRNA1 potential energy as a differentiable nn.Module.

    Args:
        topology:                    RNATopology object (connectivity + base types)
        temperature:                 simulation temperature in oxDNA reduced units
                                     (T_reduced = T_kelvin / 3000)
        seq_dependent:               if True, use sequence-dependent stacking/HB/
                                     cross-stacking parameters
        mismatch_repulsion:          if True, add a repulsive bump for non-Watson-
                                     Crick pairs (matches oxRNA2 mismatch_repulsion)
        mismatch_repulsion_strength: strength of the mismatch repulsion (default 1.0);
                                     corresponds to C++ mismatch_repulsion_strength
    """

    def __init__(
        self,
        topology: RNATopology,
        temperature: float,
        seq_dependent: bool = True,
        mismatch_repulsion: bool = False,
        mismatch_repulsion_strength: float = 1.0,
    ):
        super().__init__()

        self.topology    = topology
        self.temperature = temperature
        self.seq_dependent = seq_dependent
        self.mismatch_repulsion = mismatch_repulsion
        self.mismatch_repulsion_strength = mismatch_repulsion_strength

        # Interaction cutoff
        self.cutoff: float = RC.compute_rna_rcut()

        # Register bonded_pairs as a buffer so .to(device) moves it
        self.register_buffer('bonded_pairs',
                             topology.bonded_pairs.clone())

        # Pre-compute per-bonded-pair stacking epsilon and shift
        stk_eps   = topology.compute_stacking_eps(temperature, seq_dependent)
        stk_shift = topology.compute_stacking_shift(stk_eps)
        self.register_buffer('stacking_eps',   stk_eps)
        self.register_buffer('stacking_shift', stk_shift)

        # Pre-compute HB epsilon table  (4, 4)
        hbond_eps = topology.compute_hbond_eps(seq_dependent)
        self.register_buffer('hbond_eps_table', hbond_eps)

        # Pre-compute cross-stacking K table  (4, 4)
        cross_k = topology.compute_cross_stacking_k(seq_dependent)
        self.register_buffer('cross_k_table', cross_k)

        # Register base_types as buffer
        self.register_buffer('base_types', topology.base_types.clone())

        # Neighbor-list cache
        self.nl_skin: float = 0.0
        self._nl_pairs:     Optional[Tensor] = None
        self._nl_positions: Optional[Tensor] = None
        self.nl_backend: str = 'oxdna'

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def set_nl_skin(self, skin: float):
        """Enable neighbor-list caching with the given skin distance."""
        self.nl_skin = skin
        self._nl_pairs = None

    def set_nl_backend(self, backend: str):
        """Set the neighbor-list backend ('oxdna' or 'torchmdnet')."""
        if backend not in ('oxdna', 'torchmdnet'):
            raise ValueError(
                f"Unknown neighbor-list backend '{backend}'. "
                "Choose 'oxdna' or 'torchmdnet'.")
        if backend == 'torchmdnet' and not _torchmdnet_available():
            raise ImportError(
                "torchmd-net is not installed. "
                "Install with:  pip install torchmd-net")
        self.nl_backend = backend
        self._nl_pairs = None

    def compile(self, **kwargs):
        """Wrap forward() with torch.compile for kernel fusion."""
        kw = {'mode': 'reduce-overhead'}
        kw.update(kwargs)
        self.forward = torch.compile(self.forward, **kw)
        return self

    # ------------------------------------------------------------------
    # Internal: pair finding
    # ------------------------------------------------------------------

    def _get_nonbonded_pairs(
        self, positions: Tensor, box: Optional[Tensor]
    ) -> Tensor:
        """Return (possibly cached) non-bonded pair list."""
        if self.nl_skin <= 0.0:
            return find_nonbonded_pairs(
                positions, self.topology, box, self.cutoff,
                backend=self.nl_backend)

        if self._nl_pairs is not None and self._nl_positions is not None:
            disp = positions.detach() - self._nl_positions
            if box is not None:
                disp = disp - box * torch.round(disp / box)
            if disp.norm(dim=-1).max().item() < self.nl_skin * 0.5:
                return self._nl_pairs

        pairs = find_nonbonded_pairs(
            positions, self.topology, box,
            self.cutoff + self.nl_skin,
            backend=self.nl_backend)
        self._nl_pairs     = pairs
        self._nl_positions = positions.detach().clone()
        return pairs

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, state: SystemState) -> Tensor:
        """Compute total oxRNA potential energy.

        Args:
            state: SystemState with positions and quaternions

        Returns:
            scalar energy tensor (differentiable w.r.t. positions/quaternions)
        """
        positions   = state.positions
        quaternions = state.quaternions
        box         = state.box

        bp = self.bonded_pairs
        nbp = self._get_nonbonded_pairs(positions, box)

        # --- bonded terms ---
        e_fene  = rna_fene_energy(positions, quaternions, bp, box)

        e_bonded_excl = rna_bonded_excluded_volume_energy(
            positions, quaternions, bp, box)

        e_stack = rna_stacking_energy(
            positions, quaternions, bp,
            self.stacking_eps, self.stacking_shift, box)

        # --- nonbonded terms ---
        e_nonbonded_excl = rna_nonbonded_excluded_volume_energy(
            positions, quaternions, nbp, box)

        e_hbond = rna_hbond_energy(
            positions, quaternions, nbp,
            self.base_types, self.hbond_eps_table, box,
            mismatch_repulsion=self.mismatch_repulsion,
            mismatch_repulsion_strength=self.mismatch_repulsion_strength)

        e_cross = rna_cross_stacking_energy(
            positions, quaternions, nbp,
            self.base_types, self.cross_k_table, box)

        e_cxst = rna_coaxial_stacking_energy(
            positions, quaternions, nbp, box)

        return (e_fene + e_bonded_excl + e_stack
                + e_nonbonded_excl + e_hbond + e_cross + e_cxst)

    def energy_components(self, state: SystemState) -> Dict[str, float]:
        """Return a dict of named energy components (for inspection/debugging).

        Args:
            state: SystemState

        Returns:
            dict mapping component name -> float value
        """
        positions   = state.positions
        quaternions = state.quaternions
        box         = state.box

        bp  = self.bonded_pairs
        nbp = self._get_nonbonded_pairs(positions, box)

        components = {
            'fene':           rna_fene_energy(positions, quaternions, bp, box).item(),
            'bonded_excl':    rna_bonded_excluded_volume_energy(
                                  positions, quaternions, bp, box).item(),
            'stacking':       rna_stacking_energy(
                                  positions, quaternions, bp,
                                  self.stacking_eps,
                                  self.stacking_shift, box).item(),
            'nonbonded_excl': rna_nonbonded_excluded_volume_energy(
                                  positions, quaternions, nbp, box).item(),
            'hbond':          rna_hbond_energy(
                                  positions, quaternions, nbp,
                                  self.base_types,
                                  self.hbond_eps_table, box,
                                  mismatch_repulsion=False).item(),
            'mismatch_repulsion': (
                                  rna_hbond_energy(
                                      positions, quaternions, nbp,
                                      self.base_types,
                                      self.hbond_eps_table, box,
                                      mismatch_repulsion=True,
                                      mismatch_repulsion_strength=self.mismatch_repulsion_strength,
                                  ).item()
                                  - rna_hbond_energy(
                                      positions, quaternions, nbp,
                                      self.base_types,
                                      self.hbond_eps_table, box,
                                      mismatch_repulsion=False,
                                  ).item()
                                  ) if self.mismatch_repulsion else 0.0,
            'cross_stacking': rna_cross_stacking_energy(
                                  positions, quaternions, nbp,
                                  self.base_types,
                                  self.cross_k_table, box).item(),
            'coaxial_stack':  rna_coaxial_stacking_energy(
                                  positions, quaternions, nbp, box).item(),
        }
        components['total'] = sum(components.values())
        return components

    # ------------------------------------------------------------------
    # Device transfer
    # ------------------------------------------------------------------

    def _apply(self, fn):
        """Override to also move RNATopology when .to(device) is called."""
        super()._apply(fn)
        device = self.bonded_pairs.device
        if self.topology.bonded_pairs.device != device:
            self.topology = self.topology.to(device)
        self._nl_pairs = None   # invalidate cache on device transfer
        return self
