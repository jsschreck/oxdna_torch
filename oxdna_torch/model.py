"""
Top-level oxDNA energy model.

Assembles all 7 interaction terms into a single nn.Module that
computes the total potential energy of a system state.

Forces are obtained automatically via autograd:
  energy = model(state)
  energy.backward()
  forces = -state.positions.grad
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Set

from .state import SystemState
from .topology import Topology
from . import constants as C
from .params import ParameterStore
from .pairs import (
    compute_site_positions,
    compute_site_offsets,
    find_nonbonded_pairs,
    _torchmdnet_available,
)
from .interactions.fene import fene_energy
from .interactions.excluded_volume import (
    bonded_excluded_volume_energy,
    nonbonded_excluded_volume_energy,
)
from .interactions.stacking import stacking_energy
from .interactions.hbond import hydrogen_bonding_energy
from .interactions.cross_stacking import cross_stacking_energy
from .interactions.coaxial_stacking import coaxial_stacking_energy
from .interactions.electrostatics import debye_huckel_energy, debye_huckel_params


class OxDNAEnergy(nn.Module):
    """Differentiable oxDNA energy model.

    Computes the total potential energy E(positions, quaternions) of an oxDNA
    system as the sum of pairwise interaction terms. Since E is computed
    using standard PyTorch operations, forces and torques are obtained
    via automatic differentiation.

    Supports both oxDNA1 (7 interaction terms) and oxDNA2 (8 terms, adds
    Debye-Hückel electrostatics). oxDNA2 also enables major-minor groove
    backbone geometry and uses updated FENE, stacking, HB, and CXST constants.

    The model parameters (interaction strengths, geometric constants) can
    optionally be made learnable by setting them as nn.Parameters.

    Args:
        topology: Topology object defining system connectivity
        temperature: temperature in oxDNA reduced units (T_K / 3000)
        seq_dependent: whether to use sequence-dependent stacking/HB parameters
        grooving: whether to use major-minor groove backbone positions.
                  Automatically set to True when use_oxdna2=True.
        use_oxdna2: if True, use the oxDNA2 model (grooving + updated constants
                    + Debye-Hückel electrostatics). This sets grooving=True.
        salt_concentration: molar salt concentration for Debye-Hückel (oxDNA2 only).
                            Default 0.5 M (500 mM).
        dh_half_charged_ends: if True, strand-terminus nucleotides carry half
                              charge in the DH term (oxDNA2 default: True).
        learnable: optional set of parameter names to make learnable
                   (e.g. {'f4_A', 'excl_eps', 'fene_eps'}).
                   See params.PARAM_REGISTRY for available names.
                   Additional special names: 'stacking_eps', 'hbond_eps'.
    """

    def __init__(
        self,
        topology: Topology,
        temperature: float = 0.1,
        seq_dependent: bool = True,
        grooving: bool = False,
        use_oxdna2: bool = False,
        salt_concentration: float = 0.5,
        dh_half_charged_ends: bool = True,
        learnable: Optional[Set[str]] = None,
    ):
        super().__init__()

        # oxDNA2 forces grooving on
        if use_oxdna2:
            grooving = True

        self.topology = topology
        self.temperature = temperature
        self.seq_dependent = seq_dependent
        self.grooving = grooving
        self.use_oxdna2 = use_oxdna2
        self.salt_concentration = salt_concentration
        self.dh_half_charged_ends = dh_half_charged_ends

        # Learnable parameter handling
        learnable = learnable or set()
        special_learnable = {'stacking_eps', 'hbond_eps'}
        store_learnable = learnable - special_learnable

        # Create ParameterStore for smooth function parameters.
        # For oxDNA2, we patch the f4_T0[10] and f2_K[1] values after creation.
        self.param_store = ParameterStore(learnable=store_learnable if store_learnable else None)

        if use_oxdna2:
            # Override oxDNA2-specific parameter values in the param_store buffers
            with torch.no_grad():
                # CXST: different K and different theta1 T0
                self.param_store.f4_T0[C.CXST_F4_THETA1] = C.CXST_THETA1_T0_OXDNA2
                self.param_store.f2_K[C.CXST_F2] = C.CXST_K_OXDNA2
                # FENE: slightly different equilibrium backbone distance
                self.param_store.fene_r0.fill_(C.FENE_R0_OXDNA2)

        # Register topology tensors as buffers (not parameters, not differentiable)
        self.register_buffer('bonded_pairs', topology.bonded_pairs)
        self.register_buffer('base_types', topology.base_types)
        self.register_buffer('strand_ids', topology.strand_ids)

        # Precompute sequence-dependent parameters
        stacking_eps = topology.compute_stacking_eps(temperature, seq_dependent,
                                                     use_oxdna2=use_oxdna2)
        if 'stacking_eps' in learnable:
            self.stacking_eps = nn.Parameter(stacking_eps)
        else:
            self.register_buffer('stacking_eps', stacking_eps)

        hbond_eps = topology.compute_hbond_eps(seq_dependent, use_oxdna2=use_oxdna2)
        if 'hbond_eps' in learnable:
            self.hbond_eps_matrix = nn.Parameter(hbond_eps)
        else:
            self.register_buffer('hbond_eps_matrix', hbond_eps)

        # Precompute terminus mask for DH half-charged ends
        # A nucleotide is a terminus if it has no 3' neighbor OR no 5' neighbor
        if use_oxdna2:
            n3 = topology.bonded_neighbors[:, 0]  # -1 if no 3' neighbor
            n5 = topology.bonded_neighbors[:, 1]  # -1 if no 5' neighbor
            terminus = (n3 < 0) | (n5 < 0)
            self.register_buffer('terminus_mask', terminus)

            # Precompute Debye-Hückel parameters (Python scalars, not tensors)
            self._dh_params = debye_huckel_params(temperature, salt_concentration)
        else:
            self.terminus_mask = None
            self._dh_params = None

        # Track if we have any learnable params (for fast path)
        self._has_learnable = bool(learnable)

        # Cutoff for neighbor finding (extended for DH range when oxDNA2)
        dh_rc = self._dh_params['rc'] if use_oxdna2 else 0.0
        self.cutoff = C.compute_rcut(grooving, dh_rc=dh_rc)

        # Neighbor list cache: reuse pairs for `nl_skin` steps before rebuild.
        # skin adds a buffer beyond cutoff so pairs stay valid between rebuilds.
        # Set nl_skin=0 to disable caching (always rebuild).
        self.nl_skin: float = 0.0        # extra buffer distance (set via set_nl_skin)
        self._nl_pairs: Optional[torch.Tensor] = None   # cached pair list
        self._nl_positions: Optional[torch.Tensor] = None  # positions at last build
        self._nl_box: Optional[torch.Tensor] = None
        # Neighbor list backend: 'oxdna' (built-in) or 'torchmdnet'
        self.nl_backend: str = 'oxdna'

    def set_nl_backend(self, backend: str):
        """Set the neighbor-list backend.

        Args:
            backend: ``'oxdna'`` (default built-in brute-force / cell-list) or
                     ``'torchmdnet'`` (torchmd-net kernel; uses Triton on CUDA,
                     pure-PyTorch on CPU).  Requires ``pip install torchmd-net``
                     when using ``'torchmdnet'``.

        Example::

            model.set_nl_backend('torchmdnet')   # opt in
            model.set_nl_backend('oxdna')        # revert to default
        """
        if backend not in ('oxdna', 'torchmdnet'):
            raise ValueError(
                f"Unknown neighbor-list backend '{backend}'. "
                "Choose 'oxdna' or 'torchmdnet'."
            )
        if backend == 'torchmdnet' and not _torchmdnet_available():
            raise ImportError(
                "torchmd-net is not installed. "
                "Install with:  pip install torchmd-net"
            )
        self.nl_backend = backend
        self._nl_pairs = None  # invalidate cache on backend change

    def set_nl_skin(self, skin: float):
        """Enable neighbor list caching with the given skin distance.

        The neighbor list is built with cutoff + skin, then reused until any
        particle moves more than skin/2 from its position at the last build.
        Typical value: 0.05–0.10 (in oxDNA length units).  Set to 0 to disable.

        Args:
            skin: extra buffer distance beyond the interaction cutoff
        """
        self.nl_skin = skin
        self._nl_pairs = None  # force rebuild on next step

    def _get_nonbonded_pairs(self, positions: Tensor, box: Optional[Tensor]) -> Tensor:
        """Return cached neighbor list, rebuilding if positions have drifted."""
        if self.nl_skin <= 0.0:
            # Caching disabled — always build fresh
            return find_nonbonded_pairs(
                positions, self.topology, box, self.cutoff,
                backend=self.nl_backend,
            )

        # Check if cached list is still valid
        if self._nl_pairs is not None and self._nl_positions is not None:
            disp = positions.detach() - self._nl_positions
            if box is not None:
                disp = disp - box * torch.round(disp / box)
            max_drift = disp.norm(dim=-1).max().item()
            if max_drift < self.nl_skin * 0.5:
                return self._nl_pairs  # cache hit

        # Rebuild with extended cutoff
        pairs = find_nonbonded_pairs(
            positions, self.topology, box, self.cutoff + self.nl_skin,
            backend=self.nl_backend,
        )
        self._nl_pairs = pairs
        self._nl_positions = positions.detach().clone()
        self._nl_box = box
        return pairs

    def _apply(self, fn):
        """Override to also move the Topology object when .to() / .cuda() is called."""
        super()._apply(fn)
        # After super()._apply, all buffers have been moved to the new device.
        # Detect the target device from a buffer and move topology to match.
        device = self.bonded_pairs.device
        if self.topology.bonded_pairs.device != device:
            self.topology = self.topology.to(device)
        self._nl_pairs = None  # invalidate cache on device transfer
        return self

    def forward(self, state: SystemState) -> Tensor:
        """Compute total potential energy.

        Args:
            state: SystemState with positions and quaternions
                   (positions should have requires_grad=True for force computation)

        Returns:
            Scalar tensor: total potential energy
        """
        return self.total_energy(state)

    def total_energy(self, state: SystemState) -> Tensor:
        """Compute total potential energy as sum of all interaction terms.

        Args:
            state: SystemState with positions and quaternions

        Returns:
            Scalar tensor: total potential energy
        """
        components = self.energy_components(state)
        return sum(components.values())

    def energy_components(self, state: SystemState) -> Dict[str, Tensor]:
        """Compute all energy components separately.

        Useful for diagnostics and per-term analysis.

        Args:
            state: SystemState with positions and quaternions

        Returns:
            Dict mapping interaction names to scalar energy tensors
        """
        positions = state.positions
        quaternions = state.quaternions
        box = state.box

        # Get learnable params dict.
        # For oxDNA2 we always need params because the param_store holds the
        # overridden CXST K, theta1 T0, and FENE r0 values.
        params = self.param_store.as_dict() if (self._has_learnable or self.use_oxdna2) else None
        excl_eps = params['excl_eps'] if params is not None else None

        # Compute site offsets (relative to COM)
        back_offsets, stack_offsets, base_offsets = compute_site_offsets(
            quaternions, self.grooving
        )

        # Find non-bonded pairs (cached if nl_skin > 0)
        nonbonded_pairs = self._get_nonbonded_pairs(positions, box)

        components = {}

        # === Bonded interactions ===
        components['fene'] = fene_energy(
            positions, back_offsets, self.bonded_pairs, box, params=params
        )

        components['bonded_excl_vol'] = bonded_excluded_volume_energy(
            positions, back_offsets, base_offsets, self.bonded_pairs, box,
            excl_eps=excl_eps
        )

        components['stacking'] = stacking_energy(
            positions, quaternions, stack_offsets, self.bonded_pairs,
            self.stacking_eps, box, params=params
        )

        # === Non-bonded interactions ===
        components['nonbonded_excl_vol'] = nonbonded_excluded_volume_energy(
            positions, back_offsets, base_offsets, nonbonded_pairs, box,
            excl_eps=excl_eps
        )

        components['hbond'] = hydrogen_bonding_energy(
            positions, quaternions, base_offsets, nonbonded_pairs,
            self.base_types, self.hbond_eps_matrix, box, params=params
        )

        components['cross_stacking'] = cross_stacking_energy(
            positions, quaternions, base_offsets, nonbonded_pairs, box,
            params=params
        )

        components['coaxial_stacking'] = coaxial_stacking_energy(
            positions, quaternions, stack_offsets, nonbonded_pairs, box,
            params=params, oxdna2=self.use_oxdna2
        )

        # === oxDNA2 only: Debye-Hückel electrostatics ===
        if self.use_oxdna2:
            components['debye_huckel'] = debye_huckel_energy(
                positions, back_offsets, nonbonded_pairs,
                self.terminus_mask, self._dh_params, box,
                half_charged_ends=self.dh_half_charged_ends,
            )

        return components

    def compute_forces(self, state: SystemState) -> Tensor:
        """Compute forces on all nucleotides via autograd.

        Args:
            state: SystemState (positions will temporarily have requires_grad=True)

        Returns:
            (N, 3) force tensor: F = -dE/d(positions)
        """
        positions = state.positions.detach().requires_grad_(True)
        state_copy = SystemState(
            positions=positions,
            quaternions=state.quaternions,
            velocities=state.velocities,
            ang_velocities=state.ang_velocities,
            box=state.box,
        )

        energy = self.total_energy(state_copy)
        forces = -torch.autograd.grad(energy, positions, create_graph=False)[0]
        return forces

    def compile(self, **kwargs):
        """Apply torch.compile to the energy model for faster rollouts.

        Wraps the forward/energy_components path with torch.compile, which
        reduces Python interpreter overhead and enables kernel fusion.
        Typically 2-4x faster for small-to-medium systems (<1000 nucleotides).

        Note: incompatible with create_graph=True (BPTT). Disable before training.

        Args:
            **kwargs: passed directly to torch.compile
                      (e.g. mode='reduce-overhead', fullgraph=True)

        Returns:
            self (for chaining)
        """
        default_kwargs = {'mode': 'reduce-overhead'}
        default_kwargs.update(kwargs)
        self.forward = torch.compile(self.forward, **default_kwargs)
        return self

    def update_temperature(self, temperature: float):
        """Update temperature and recompute temperature-dependent parameters.

        Args:
            temperature: new temperature in oxDNA reduced units
        """
        self.temperature = temperature
        stacking_eps = self.topology.compute_stacking_eps(
            temperature, self.seq_dependent, use_oxdna2=self.use_oxdna2
        )
        self.stacking_eps.data.copy_(stacking_eps.to(self.stacking_eps.device))
        if self.use_oxdna2:
            # Recompute DH params (lambda is temperature-dependent)
            self._dh_params = debye_huckel_params(temperature, self.salt_concentration)
