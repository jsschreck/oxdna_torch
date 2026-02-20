"""
Topology representation for oxDNA systems.

The topology encodes the fixed connectivity (bonded neighbors)
and base type information. It is NOT differentiable — it defines
the graph structure over which energy terms are computed.
"""

import torch
from torch import Tensor
from typing import Optional

from . import constants as C


class Topology:
    """Stores strand connectivity and base type information.

    Attributes:
        n_nucleotides: Total number of nucleotides
        n_strands: Number of strands
        strand_ids: (N,) int tensor - which strand each nucleotide belongs to
        base_types: (N,) int tensor - base type (A=0, C=1, G=2, T=3)
        bonded_neighbors: (N, 2) int tensor - [n3_idx, n5_idx] per nucleotide
            -1 indicates no neighbor (strand end)
        seq_dep_stck_eps: (B,) float tensor - sequence-dependent stacking epsilon
            for each bonded pair, precomputed from base types and temperature
        seq_dep_hydr_eps: (N, N) or sparse - hydrogen bonding epsilon
            only nonzero for Watson-Crick pairs; looked up per pair as needed
    """

    def __init__(
        self,
        n_nucleotides: int,
        n_strands: int,
        strand_ids: Tensor,
        base_types: Tensor,
        bonded_neighbors: Tensor,
        circular: Optional[Tensor] = None,
    ):
        """
        Args:
            n_nucleotides: total number of nucleotides
            n_strands: number of strands
            strand_ids: (N,) int - strand ID for each nucleotide
            base_types: (N,) int - base type for each nucleotide
            bonded_neighbors: (N, 2) int - [n3_idx, n5_idx], -1 for no neighbor
            circular: (n_strands,) bool - whether each strand is circular
        """
        self.n_nucleotides = n_nucleotides
        self.n_strands = n_strands
        self.strand_ids = strand_ids
        self.base_types = base_types
        self.bonded_neighbors = bonded_neighbors  # (N, 2): [n3, n5]
        self.circular = circular

        # Precompute bonded pair list: (B, 2) where each row is [p, q] with q = p.n3
        self._build_bonded_pairs()

    def _build_bonded_pairs(self):
        """Build the list of bonded pairs from bonded_neighbors.

        Convention: for each pair (p, q), q is the 3' neighbor of p.
        This matches the C++ code's convention where p→n3 = q.
        """
        n3 = self.bonded_neighbors[:, 0]  # (N,) n3 neighbor indices
        has_n3 = n3 >= 0
        p_indices = torch.arange(self.n_nucleotides, device=n3.device)[has_n3]
        q_indices = n3[has_n3]

        self.bonded_pairs = torch.stack([p_indices, q_indices], dim=-1)  # (B, 2)
        self.n_bonded = self.bonded_pairs.shape[0]

        # Precompute a set of bonded pairs for fast lookup
        # Store as a flat tensor of pair hashes for O(1) membership check
        N = self.n_nucleotides
        bonded_hash = p_indices * N + q_indices
        bonded_hash_rev = q_indices * N + p_indices
        self._bonded_hash_set = set(bonded_hash.tolist() + bonded_hash_rev.tolist())

    def is_bonded(self, i: int, j: int) -> bool:
        """Check if nucleotides i and j are bonded neighbors."""
        return (i * self.n_nucleotides + j) in self._bonded_hash_set

    def get_bonded_pair_base_types(self) -> tuple:
        """Get base types for each bonded pair.

        Returns:
            p_types: (B,) base type of p (the nucleotide with n3 neighbor)
            q_types: (B,) base type of q (the n3 neighbor)
        """
        p_idx = self.bonded_pairs[:, 0]
        q_idx = self.bonded_pairs[:, 1]
        return self.base_types[p_idx], self.base_types[q_idx]

    def compute_stacking_eps(
        self, temperature: float, seq_dependent: bool = True, use_oxdna2: bool = False
    ) -> Tensor:
        """Compute sequence-dependent stacking epsilon for each bonded pair.

        In the C++ code, the stacking f1 call uses _f1(r, STCK_F1, q->type, p->type)
        which indexes F1_EPS[STCK_F1][n3_type][n5_type].

        For the bonded pair (p, q) where q = p.n3:
          - p is the 5' nucleotide
          - q is the 3' nucleotide
          - The stacking interaction is between the bases of p and q
          - In the C++ call: n3 = q->type, n5 = p->type

        For sequence-averaged:
          oxDNA1: eps = STCK_BASE_EPS_OXDNA  + STCK_FACT_EPS_OXDNA  * T
          oxDNA2: eps = STCK_BASE_EPS_OXDNA2 + STCK_FACT_EPS_OXDNA2 * T
        For sequence-dependent: eps = STCK_EPS_SEQ[n3][n5] * (1 - fact + T*9*fact)
          (same formula for both oxDNA1 and oxDNA2)

        Args:
            temperature: temperature in oxDNA reduced units
            seq_dependent: whether to use sequence-dependent parameters
            use_oxdna2: if True, use oxDNA2 average-sequence constants

        Returns:
            (B,) stacking epsilon for each bonded pair
        """
        if not seq_dependent:
            if use_oxdna2:
                eps = C.STCK_BASE_EPS_OXDNA2 + C.STCK_FACT_EPS_OXDNA2 * temperature
            else:
                eps = C.STCK_BASE_EPS_OXDNA + C.STCK_FACT_EPS_OXDNA * temperature
            return torch.full((self.n_bonded,), eps,
                          dtype=torch.float64, device=self.bonded_pairs.device)

        p_types, q_types = self.get_bonded_pair_base_types()

        # Move the constant tensor to the correct device BEFORE indexing
        device = self.bonded_pairs.device
        raw_eps = C.STCK_EPS_SEQ.to(device)[q_types, p_types]
        fact = C.STCK_FACT_EPS_SEQ
        eps = raw_eps * (1.0 - fact + temperature * 9.0 * fact)
        return eps


    def compute_stacking_shift(self, stacking_eps: Tensor) -> Tensor:
        """Compute the f1 shift for stacking.

        SHIFT = eps * (1 - exp(-(RC - R0) * A))^2
        """
        import math
        factor = (1.0 - math.exp(-(C.STCK_RC - C.STCK_R0) * C.STCK_A)) ** 2
        return stacking_eps * factor

    def compute_hbond_eps(self, seq_dependent: bool = True, use_oxdna2: bool = False) -> Tensor:
        """Get the hydrogen bonding epsilon lookup.

        Args:
            seq_dependent: whether to use sequence-dependent parameters
            use_oxdna2: if True, use HYDR_EPS_OXDNA2 for average-sequence model

        Returns:
            (4, 4) tensor of HB epsilon values indexed by [p_type, q_type]
        """
        device = self.bonded_pairs.device
        if seq_dependent:
            return C.HYDR_EPS_SEQ.clone().to(device)
        else:
            # Average model: all WC pairs get the same eps
            hydr_eps = C.HYDR_EPS_OXDNA2 if use_oxdna2 else C.HYDR_EPS_OXDNA
            eps = torch.zeros(4, 4, dtype=torch.float64, device=device)
            eps[0, 3] = eps[3, 0] = hydr_eps  # A-T
            eps[1, 2] = eps[2, 1] = hydr_eps  # C-G
            return eps

    def to(self, device: torch.device) -> 'Topology':
        """Move topology tensors to specified device."""
        return Topology(
            n_nucleotides=self.n_nucleotides,
            n_strands=self.n_strands,
            strand_ids=self.strand_ids.to(device),
            base_types=self.base_types.to(device),
            bonded_neighbors=self.bonded_neighbors.to(device),
            circular=self.circular.to(device) if self.circular is not None else None,
        )
