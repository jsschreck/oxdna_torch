"""
RNA cross-stacking interaction.

E_CRST = f2(r_base) * f4(t1) * f4(t2) * f4(t3)
         * [f4(t7) + f4(pi-t7)] * [f4(t8) + f4(pi-t8)]

where:
  r    : distance between base sites of p and q
  t1   : acos(-a1_p . a1_q)
  t2   : acos(-a1_q . r_hat)
  t3   : acos( a1_p . r_hat)
  t7   : acos(-a3_q . r_hat)   with symmetric mirror term
  t8   : acos( a3_p . r_hat)   with symmetric mirror term

f4 type indices:
  t1 -> RNA_CRST_F4_THETA1 = 6
  t2 -> RNA_CRST_F4_THETA2 = 7
  t3 -> RNA_CRST_F4_THETA3 = 7  (same)
  t7 -> RNA_CRST_F4_THETA7 = 9  (+ mirror)
  t8 -> RNA_CRST_F4_THETA8 = 9  (+ mirror)

Reference: RNAInteraction::_cross_stacking
"""

import math
import torch
from torch import Tensor
from typing import Optional

from .. import rna_constants as RC
from ..smooth import f2
from ..utils import safe_acos
from ..pairs import min_image_displacement


def _rna_f4(theta: Tensor, f4_type: int) -> Tensor:
    """RNA f4 angular modulation — reads from RNA_F4_THETA_* (16 entries)."""
    A  = RC.RNA_F4_THETA_A[f4_type].item()
    B  = RC.RNA_F4_THETA_B[f4_type].item()
    T0 = RC.RNA_F4_THETA_T0[f4_type].item()
    TS = RC.RNA_F4_THETA_TS[f4_type].item()
    TC = RC.RNA_F4_THETA_TC[f4_type].item()

    t = torch.abs(theta - T0)
    parabola = 1.0 - A * t * t
    smooth   = B * (TC - t) ** 2
    zero     = torch.zeros_like(theta)
    return torch.where(t < TC,
                       torch.where(t > TS, smooth, parabola),
                       zero)


def rna_cross_stacking_energy(
    positions: Tensor,
    quaternions: Tensor,
    nonbonded_pairs: Tensor,
    base_types: Tensor,
    cross_k_table: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Compute RNA cross-stacking energy for all non-bonded pairs.

    Args:
        positions:       (N, 3)
        quaternions:     (N, 4)
        nonbonded_pairs: (P, 2)
        base_types:      (N,)  A=0, C=1, G=2, U=3
        cross_k_table:   (4, 4) K-scale factor per base-pair combination
                         (all ones for average-sequence mode)
        box:             (3,) periodic box dimensions, or None

    Returns:
        scalar total cross-stacking energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    from ..quaternion import quat_to_rotmat
    R = quat_to_rotmat(quaternions)
    a1 = R[:, :, 0]
    a3 = R[:, :, 2]

    base_pos = positions + RC.RNA_POS_BASE * a1   # (N, 3)

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    r_vec = min_image_displacement(
        base_pos[q_idx] - base_pos[p_idx], box)  # (P, 3)
    r_mod = r_vec.norm(dim=-1)
    r_hat = r_vec / r_mod.clamp(min=1e-12).unsqueeze(-1)

    # Sequence-dependent K scaling
    p_types = base_types[p_idx]
    q_types = base_types[q_idx]
    k_scale = cross_k_table[p_types, q_types]   # (P,)

    # f2 radial (type 0 = CRST), scaled by k_scale
    f2_val = f2(r_mod, RC.RNA_CRST_F2) * k_scale

    # Angular factors
    a1_p = a1[p_idx]
    a1_q = a1[q_idx]
    a3_p = a3[p_idx]
    a3_q = a3[q_idx]

    t1 = safe_acos(-(a1_p * a1_q).sum(-1))
    t2 = safe_acos(-(a1_q * r_hat).sum(-1))
    t3 = safe_acos( (a1_p * r_hat).sum(-1))
    t7 = safe_acos(-(a3_q * r_hat).sum(-1))
    t8 = safe_acos( (a3_p * r_hat).sum(-1))

    f4_t1 = _rna_f4(t1, RC.RNA_CRST_F4_THETA1)
    f4_t2 = _rna_f4(t2, RC.RNA_CRST_F4_THETA2)
    f4_t3 = _rna_f4(t3, RC.RNA_CRST_F4_THETA3)
    # Mirror symmetry for t7 and t8 (C++ adds f4(t) + f4(pi-t))
    f4_t7 = (_rna_f4(t7, RC.RNA_CRST_F4_THETA7)
             + _rna_f4(math.pi - t7, RC.RNA_CRST_F4_THETA7))
    f4_t8 = (_rna_f4(t8, RC.RNA_CRST_F4_THETA8)
             + _rna_f4(math.pi - t8, RC.RNA_CRST_F4_THETA8))

    energy_per_pair = f2_val * f4_t1 * f4_t2 * f4_t3 * f4_t7 * f4_t8
    return energy_per_pair.sum()
