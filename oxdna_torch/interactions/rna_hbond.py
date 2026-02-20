"""
RNA hydrogen-bonding interaction.

E_HB = f1(r_base) * f4(t1) * f4(t2) * f4(t3) * f4(t4) * f4(t7) * f4(t8)

Only between Watson-Crick pairs (A-U, G-C) and, in sequence-dependent mode,
G-U wobble pairs.

Angles (all via acos, following RNAInteraction::_hydrogen_bonding):
  r    : distance between base sites of p and q
  t1   : acos(-a1_p . a1_q)
  t2   : acos(-a1_q . r_hat)
  t3   : acos( a1_p . r_hat)
  t4   : acos( a3_p . a3_q)
  t7   : acos(-a3_q . r_hat)
  t8   : acos( a3_p . r_hat)

f4 type indices (from rna_constants):
  t1 -> RNA_HYDR_F4_THETA1 = 2
  t2 -> RNA_HYDR_F4_THETA2 = 3
  t3 -> RNA_HYDR_F4_THETA3 = 3  (same params as t2)
  t4 -> RNA_HYDR_F4_THETA4 = 4
  t7 -> RNA_HYDR_F4_THETA7 = 5
  t8 -> RNA_HYDR_F4_THETA8 = 5  (same params as t7)
"""

import torch
from torch import Tensor
from typing import Optional

from .. import rna_constants as RC
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


def rna_hbond_energy(
    positions: Tensor,
    quaternions: Tensor,
    nonbonded_pairs: Tensor,
    base_types: Tensor,
    hbond_eps_table: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Compute RNA hydrogen-bonding energy for all non-bonded pairs.

    Args:
        positions:       (N, 3)
        quaternions:     (N, 4)
        nonbonded_pairs: (P, 2)
        base_types:      (N,)  A=0, C=1, G=2, U=3
        hbond_eps_table: (4, 4) epsilon indexed [p_type, q_type];
                         zero for non-pairing combinations
        box:             (3,) periodic box dimensions, or None

    Returns:
        scalar total HB energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    from ..quaternion import quat_to_rotmat
    R = quat_to_rotmat(quaternions)   # (N, 3, 3)
    a1 = R[:, :, 0]
    a3 = R[:, :, 2]

    base_pos = positions + RC.RNA_POS_BASE * a1   # (N, 3)

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # Per-pair epsilon from table
    p_types = base_types[p_idx]
    q_types = base_types[q_idx]
    eps = hbond_eps_table[p_types, q_types]   # (P,)

    # Skip pairs with zero epsilon early (non-pairing combinations)
    active = eps > 0.0
    if not active.any():
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    # --- radial part ---
    r_vec = min_image_displacement(
        base_pos[q_idx] - base_pos[p_idx], box)  # (P, 3)
    r_mod = r_vec.norm(dim=-1)                    # (P,)
    r_hat = r_vec / r_mod.clamp(min=1e-12).unsqueeze(-1)

    # f1 (Morse-like, type 0 = HYDR)
    A      = RC.RNA_F1_A[RC.RNA_HYDR_F1].item()
    R0     = RC.RNA_F1_R0[RC.RNA_HYDR_F1].item()
    RC_val = RC.RNA_F1_RC[RC.RNA_HYDR_F1].item()
    BLOW   = RC.RNA_F1_BLOW[RC.RNA_HYDR_F1].item()
    BHIGH  = RC.RNA_F1_BHIGH[RC.RNA_HYDR_F1].item()
    RLOW   = RC.RNA_F1_RLOW[RC.RNA_HYDR_F1].item()
    RHIGH  = RC.RNA_F1_RHIGH[RC.RNA_HYDR_F1].item()
    RCLOW  = RC.RNA_F1_RCLOW[RC.RNA_HYDR_F1].item()
    RCHIGH = RC.RNA_F1_RCHIGH[RC.RNA_HYDR_F1].item()

    shift = eps * (1.0 - torch.exp(torch.tensor(
        -(RC_val - R0) * A, dtype=positions.dtype, device=positions.device))) ** 2

    r = r_mod
    tmp = 1.0 - torch.exp(-(r - R0) * A)
    morse    = eps * tmp * tmp - shift
    onset    = eps * BLOW  * (r - RCLOW)  ** 2
    cutoff_r = eps * BHIGH * (r - RCHIGH) ** 2
    zero     = torch.zeros_like(r)

    f1_val = torch.where(r < RCHIGH,
                 torch.where(r > RHIGH, cutoff_r,
                     torch.where(r > RLOW, morse,
                         torch.where(r > RCLOW, onset, zero))),
                 zero)

    # --- angular part ---
    a1_p = a1[p_idx]
    a1_q = a1[q_idx]
    a3_p = a3[p_idx]
    a3_q = a3[q_idx]

    # t1 = acos(-a1_p . a1_q)
    t1 = safe_acos(-(a1_p * a1_q).sum(-1))
    # t2 = acos(-a1_q . r_hat)
    t2 = safe_acos(-(a1_q * r_hat).sum(-1))
    # t3 = acos(a1_p . r_hat)
    t3 = safe_acos( (a1_p * r_hat).sum(-1))
    # t4 = acos(a3_p . a3_q)
    t4 = safe_acos( (a3_p * a3_q).sum(-1))
    # t7 = acos(-a3_q . r_hat)
    t7 = safe_acos(-(a3_q * r_hat).sum(-1))
    # t8 = acos(a3_p . r_hat)
    t8 = safe_acos( (a3_p * r_hat).sum(-1))

    f4_t1 = _rna_f4(t1, RC.RNA_HYDR_F4_THETA1)
    f4_t2 = _rna_f4(t2, RC.RNA_HYDR_F4_THETA2)
    f4_t3 = _rna_f4(t3, RC.RNA_HYDR_F4_THETA3)
    f4_t4 = _rna_f4(t4, RC.RNA_HYDR_F4_THETA4)
    f4_t7 = _rna_f4(t7, RC.RNA_HYDR_F4_THETA7)
    f4_t8 = _rna_f4(t8, RC.RNA_HYDR_F4_THETA8)

    energy_per_pair = f1_val * f4_t1 * f4_t2 * f4_t3 * f4_t4 * f4_t7 * f4_t8
    return energy_per_pair.sum()
