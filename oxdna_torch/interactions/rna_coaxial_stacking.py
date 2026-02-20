"""
RNA coaxial-stacking interaction.

E_CXST = f2(r_stack) * [f4(t1)+f4(2pi-t1)] * f4(t4)
         * [f4(t5)+f4(pi-t5)] * [f4(t6)+f4(pi-t6)]
         * f5(phi3) * f5(phi4)

Key difference from oxDNA: **two** phi angles (phi3 and phi4) computed
from the backbone vectors of p and q respectively.

Angles:
  r    : distance between STACK sites of p and q
  t1   : acos(-a1_p . a1_q)           [+ mirror: 2pi-t1]
  t4   : acos( a3_p . a3_q)
  t5   : acos( a3_p . r_hat)          [+ mirror: pi-t5]
  t6   : acos(-a3_q . r_hat)          [+ mirror: pi-t6]
  phi3 : cos of dihedral = r_hat . (rback_hat x a1_p)   via f5 type 2
  phi4 : cos of dihedral = r_hat . (rback_hat x a1_q)   via f5 type 3

where rback_hat = unit vector from back_p to back_q (same backbone as FENE).

Reference: RNAInteraction::_coaxial_stacking
"""

import math
import torch
from torch import Tensor
from typing import Optional

from .. import rna_constants as RC
from ..smooth import f2, f5
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


def rna_coaxial_stacking_energy(
    positions: Tensor,
    quaternions: Tensor,
    nonbonded_pairs: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Compute RNA coaxial-stacking energy for all non-bonded pairs.

    Args:
        positions:       (N, 3)
        quaternions:     (N, 4)
        nonbonded_pairs: (P, 2)
        box:             (3,) periodic box dimensions, or None

    Returns:
        scalar total coaxial-stacking energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    from ..quaternion import quat_to_rotmat
    R = quat_to_rotmat(quaternions)   # (N, 3, 3)
    a1 = R[:, :, 0]
    a3 = R[:, :, 2]

    # STACK site (simple scalar along a1, matches C++ RNANucleotide::STACK)
    stack_pos = positions + RC.RNA_POS_STACK * a1   # (N, 3)

    # Backbone site (3D offset)
    back_pos = (positions
                + RC.RNA_POS_BACK_a1 * R[:, :, 0]
                + RC.RNA_POS_BACK_a2 * R[:, :, 1]
                + RC.RNA_POS_BACK_a3 * R[:, :, 2])  # (N, 3)

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # --- stacking vector (with MIC) ---
    r_vec = min_image_displacement(
        stack_pos[q_idx] - stack_pos[p_idx], box)  # (P, 3)
    r_mod = r_vec.norm(dim=-1)
    r_hat = r_vec / r_mod.clamp(min=1e-12).unsqueeze(-1)

    # --- backbone vector (with MIC) ---
    rback_vec = min_image_displacement(
        back_pos[q_idx] - back_pos[p_idx], box)
    rback_mod = rback_vec.norm(dim=-1).clamp(min=1e-12)
    rback_hat = rback_vec / rback_mod.unsqueeze(-1)

    # --- f2 radial (type 1 = CXST) ---
    f2_val = f2(r_mod, RC.RNA_CXST_F2)

    # --- particle axes ---
    a1_p = a1[p_idx]
    a1_q = a1[q_idx]
    a3_p = a3[p_idx]
    a3_q = a3[q_idx]

    # --- angular factors ---
    t1 = safe_acos(-(a1_p * a1_q).sum(-1))
    t4 = safe_acos( (a3_p * a3_q).sum(-1))
    t5 = safe_acos( (a3_p * r_hat).sum(-1))
    t6 = safe_acos(-(a3_q * r_hat).sum(-1))

    # t1: symmetric about 2pi (C++ adds f4(t1) + f4(2pi-t1))
    f4_t1 = (_rna_f4(t1, RC.RNA_CXST_F4_THETA1)
             + _rna_f4(2.0 * math.pi - t1, RC.RNA_CXST_F4_THETA1))
    f4_t4 = _rna_f4(t4, RC.RNA_CXST_F4_THETA4)
    # t5, t6: symmetric about pi
    f4_t5 = (_rna_f4(t5, RC.RNA_CXST_F4_THETA5)
             + _rna_f4(math.pi - t5, RC.RNA_CXST_F4_THETA5))
    f4_t6 = (_rna_f4(t6, RC.RNA_CXST_F4_THETA6)
             + _rna_f4(math.pi - t6, RC.RNA_CXST_F4_THETA6))

    # --- phi3 and phi4 (dihedral-like) ---
    # cosphi3 = r_hat . (rback_hat x a1_p)
    cross_p = torch.linalg.cross(rback_hat, a1_p)
    cos_phi3 = (r_hat * cross_p).sum(-1)

    # cosphi4 = r_hat . (rback_hat x a1_q)
    cross_q = torch.linalg.cross(rback_hat, a1_q)
    cos_phi4 = (r_hat * cross_q).sum(-1)

    f5_phi3 = f5(cos_phi3, RC.RNA_CXST_F5_PHI3)
    f5_phi4 = f5(cos_phi4, RC.RNA_CXST_F5_PHI4)

    energy_per_pair = (f2_val * f4_t1 * f4_t4 * f4_t5 * f4_t6
                       * f5_phi3 * f5_phi4)
    return energy_per_pair.sum()
