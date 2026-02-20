"""
Topology representation for oxRNA systems.

Mirrors Topology (topology.py) but uses A/U/G/C base encoding and
computes RNA-specific stacking and hydrogen-bonding epsilon tables.
"""

import math
import torch
from torch import Tensor
from typing import Optional

from . import rna_constants as RC


class RNATopology:
    """Stores strand connectivity and base type information for oxRNA.

    Base encoding: A=0, C=1, G=2, U=3
    Watson-Crick pairs: A-U (sum=3), G-C (sum=3)
    Wobble pair: G-U (sum=5) -- active only in sequence-dependent mode

    Attributes:
        n_nucleotides: total number of nucleotides
        n_strands: number of strands
        strand_ids: (N,) int tensor
        base_types: (N,) int tensor  (A=0, C=1, G=2, U=3)
        bonded_neighbors: (N, 2) int tensor [n3_idx, n5_idx], -1 = no neighbor
        bonded_pairs: (B, 2) int tensor of all bonded (p, q) pairs, q = p.n3
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
        self.n_nucleotides = n_nucleotides
        self.n_strands = n_strands
        self.strand_ids = strand_ids
        self.base_types = base_types          # A=0, C=1, G=2, U=3
        self.bonded_neighbors = bonded_neighbors
        self.circular = circular

        self._build_bonded_pairs()

    def _build_bonded_pairs(self):
        """Build (B, 2) bonded pair list, q = p.n3 convention."""
        n3 = self.bonded_neighbors[:, 0]
        has_n3 = n3 >= 0
        p_idx = torch.arange(self.n_nucleotides, device=n3.device)[has_n3]
        q_idx = n3[has_n3]
        self.bonded_pairs = torch.stack([p_idx, q_idx], dim=-1)
        self.n_bonded = self.bonded_pairs.shape[0]

        N = self.n_nucleotides
        bh  = p_idx * N + q_idx
        bhr = q_idx * N + p_idx
        self._bonded_hash_set = set(bh.tolist() + bhr.tolist())

    def is_bonded(self, i: int, j: int) -> bool:
        return (i * self.n_nucleotides + j) in self._bonded_hash_set

    def get_bonded_pair_base_types(self):
        """Return (p_types, q_types) for all bonded pairs."""
        p_idx = self.bonded_pairs[:, 0]
        q_idx = self.bonded_pairs[:, 1]
        return self.base_types[p_idx], self.base_types[q_idx]

    # ------------------------------------------------------------------
    # Epsilon tables
    # ------------------------------------------------------------------

    def compute_stacking_eps(
        self, temperature: float, seq_dependent: bool = True
    ) -> Tensor:
        """Compute per-bonded-pair stacking epsilon for oxRNA.

        Average-sequence: eps = RNA_STCK_BASE_EPS + RNA_STCK_FACT_EPS * T
        Sequence-dependent:
            eps(i,j) = raw_value(n3, n5) * (1 + T * RNA_STCK_FACT_EPS_SEQ)
        where i = n3 base type of q, j = n5 base type of p  (q is 3' of p).

        Args:
            temperature: in oxDNA reduced units
            seq_dependent: use sequence-dependent table

        Returns:
            (B,) stacking epsilon tensor
        """
        device = self.bonded_pairs.device
        if not seq_dependent:
            eps_val = RC.RNA_STCK_BASE_EPS + RC.RNA_STCK_FACT_EPS * temperature
            return torch.full((self.n_bonded,), eps_val,
                              dtype=torch.float64, device=device)

        p_types, q_types = self.get_bonded_pair_base_types()
        # C++ convention: n3 = q->type, n5 = p->type
        raw_eps = RC.RNA_STCK_EPS_SEQ.to(device)[q_types, p_types]
        return raw_eps * (1.0 + temperature * RC.RNA_STCK_FACT_EPS_SEQ)

    def compute_stacking_shift(self, stacking_eps: Tensor) -> Tensor:
        """Compute f1 SHIFT = eps * (1 - exp(-(RC-R0)*A))^2 for stacking."""
        factor = (1.0 - math.exp(
            -(RC.RNA_STCK_RC - RC.RNA_STCK_R0) * RC.RNA_STCK_A)) ** 2
        return stacking_eps * factor

    def compute_hbond_eps(self, seq_dependent: bool = True) -> Tensor:
        """Get the (4, 4) hydrogen-bonding epsilon table.

        Average-sequence: A-U and G-C pairs get RNA_HYDR_EPS; others zero.
        Sequence-dependent: loaded from RNA_HYDR_EPS_SEQ (includes G-U wobble).

        Returns:
            (4, 4) tensor indexed [p_type, q_type]
        """
        device = self.bonded_pairs.device
        if seq_dependent:
            return RC.RNA_HYDR_EPS_SEQ.clone().to(device)
        eps = torch.zeros(4, 4, dtype=torch.float64, device=device)
        eps[0, 3] = eps[3, 0] = RC.RNA_HYDR_EPS   # A-U / U-A
        eps[1, 2] = eps[2, 1] = RC.RNA_HYDR_EPS   # C-G / G-C
        return eps

    def compute_cross_stacking_k(self, seq_dependent: bool = True) -> Tensor:
        """Get the (4, 4) cross-stacking K scale table.

        Average-sequence: all ones (no scaling).
        Sequence-dependent: loaded from RNA_CROSS_SEQ_K.

        Returns:
            (4, 4) tensor of K scaling factors
        """
        device = self.bonded_pairs.device
        if seq_dependent:
            return RC.RNA_CROSS_SEQ_K.clone().to(device)
        return torch.ones(4, 4, dtype=torch.float64, device=device)

    def to(self, device) -> 'RNATopology':
        """Move all tensors to the specified device."""
        return RNATopology(
            n_nucleotides=self.n_nucleotides,
            n_strands=self.n_strands,
            strand_ids=self.strand_ids.to(device),
            base_types=self.base_types.to(device),
            bonded_neighbors=self.bonded_neighbors.to(device),
            circular=(self.circular.to(device)
                      if self.circular is not None else None),
        )
