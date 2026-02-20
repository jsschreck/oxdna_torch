"""
RNA stacking interaction.

E_stack = f1(r_stack) * f4(t4) * f4(PI-t5) * f4(t6) * f5(phi1) * f5(phi2)
          * f4(tB1) * f4(tB2)

Reference: RNAInteraction::_stacking  (RNAInteraction.cpp lines ~530-660)

Key details matching the C++ implementation exactly:
 - p = 5'-side nucleotide (bonded_pairs[:, 0])
 - q = 3'-side nucleotide (bonded_pairs[:, 1] = p.n3)
 - rstack = STACK_5(q) - STACK_3(p)         (5' -> 3' direction)
 - rback  = BACK(q) - BACK(p)               (same direction)
 - t5  = acos(a3_p . rstackdir);  f4(PI - t5, THETA5)  [note: PI-t5 !]
 - t6  = acos(-a3_q . rstackdir); f4(t6, THETA6)
 - tB1 = acos(-rback_dir . BBVECTOR_3_p);  f4(tB1, THETAB1)
 - tB2 = acos(-rback_dir . BBVECTOR_5_q);  f4(tB2, THETAB2)
 - phi1 = a2_p . rback_dir;  f5(phi1, PHI1)
 - phi2 = a2_q . rback_dir;  f5(phi2, PHI2)
"""

import math
import torch
from torch import Tensor
from typing import Optional

from .. import rna_constants as RC
from ..smooth import f5
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


def _rna_stacking_sites(positions: Tensor, quaternions: Tensor):
    """Compute STACK_3 and STACK_5 site positions for all nucleotides.

    STACK_3 = COM + a1 * RNA_POS_STACK_3_a1 + a2 * RNA_POS_STACK_3_a2
    STACK_5 = COM + a1 * RNA_POS_STACK_5_a1 + a2 * RNA_POS_STACK_5_a2

    Also returns backbone vectors BBVECTOR_3 and BBVECTOR_5 in the lab frame,
    and the a2 axis (needed for phi1, phi2).

    Returns:
        stack3_pos: (N, 3)  STACK_3 site of each nucleotide
        stack5_pos: (N, 3)  STACK_5 site of each nucleotide
        bbvec3_lab: (N, 3)  BBVECTOR_3 in lab frame
        bbvec5_lab: (N, 3)  BBVECTOR_5 in lab frame
        a1:         (N, 3)
        a2:         (N, 3)
        a3:         (N, 3)
    """
    from ..quaternion import quat_to_rotmat
    R = quat_to_rotmat(quaternions)   # (N, 3, 3)
    a1 = R[:, :, 0]
    a2 = R[:, :, 1]
    a3 = R[:, :, 2]

    stack3_pos = (positions
                  + RC.RNA_POS_STACK_3_a1 * a1
                  + RC.RNA_POS_STACK_3_a2 * a2)

    stack5_pos = (positions
                  + RC.RNA_POS_STACK_5_a1 * a1
                  + RC.RNA_POS_STACK_5_a2 * a2)

    # BBVECTOR_3 = p3_body rotated to lab frame; p3_body = (p3_x, p3_y, p3_z)
    p3_body = torch.tensor(
        [RC.RNA_P3_x, RC.RNA_P3_y, RC.RNA_P3_z],
        dtype=positions.dtype, device=positions.device)
    p5_body = torch.tensor(
        [RC.RNA_P5_x, RC.RNA_P5_y, RC.RNA_P5_z],
        dtype=positions.dtype, device=positions.device)

    bbvec3_lab = (R @ p3_body.unsqueeze(-1)).squeeze(-1)   # (N, 3)
    bbvec5_lab = (R @ p5_body.unsqueeze(-1)).squeeze(-1)   # (N, 3)

    return stack3_pos, stack5_pos, bbvec3_lab, bbvec5_lab, a1, a2, a3


def rna_stacking_energy(
    positions: Tensor,
    quaternions: Tensor,
    bonded_pairs: Tensor,
    stacking_eps: Tensor,
    stacking_shift: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Compute the RNA stacking energy for all bonded pairs.

    Follows RNAInteraction::_stacking exactly:

      E = f1(r) * f4(PI-t5) * f4(t6) * f5(phi1) * f5(phi2) * f4(tB1) * f4(tB2)

    where p = 5'-side nucleotide, q = 3'-side nucleotide:
      rstack  = STACK_5(q) - STACK_3(p)       [5'->3' direction]
      rback   = BACK(q) - BACK(p)             [5'->3' direction]
      t5      = acos( a3_p  . rstackdir)      → use f4(PI - t5)
      t6      = acos(-a3_q  . rstackdir)      → use f4(t6)
      tB1     = acos(-rback_dir . bbvec3_p)   → use f4(tB1)
      tB2     = acos(-rback_dir . bbvec5_q)   → use f4(tB2)
      phi1    = a2_p . rback_dir              → use f5(phi1)
      phi2    = a2_q . rback_dir              → use f5(phi2)

    Args:
        positions:      (N, 3)
        quaternions:    (N, 4)
        bonded_pairs:   (B, 2)  [p, q] where p = 5'-side, q = p.n3 = 3'-side
        stacking_eps:   (B,)   epsilon per bonded pair
        stacking_shift: (B,)   f1 SHIFT per bonded pair
        box:            (3,) periodic box dimensions, or None

    Returns:
        scalar total stacking energy
    """
    if bonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    stack3, stack5, bbvec3_lab, bbvec5_lab, a1, a2, a3 = _rna_stacking_sites(
        positions, quaternions)

    p_idx = bonded_pairs[:, 0]   # 5' nucleotide
    q_idx = bonded_pairs[:, 1]   # 3' nucleotide  (q = p.n3)

    # Stacking vector: STACK_5(q=3'-side) - STACK_3(p=5'-side), with MIC
    # This is the 5'->3' direction following C++ convention.
    r_vec = min_image_displacement(stack5[q_idx] - stack3[p_idx], box)  # (B, 3)
    r_mod = r_vec.norm(dim=-1)              # (B,)
    r_hat = r_vec / r_mod.clamp(min=1e-12).unsqueeze(-1)

    # Backbone vector: BACK(q) - BACK(p), with MIC
    back_pos = (positions
                + RC.RNA_POS_BACK_a1 * a1
                + RC.RNA_POS_BACK_a2 * a2
                + RC.RNA_POS_BACK_a3 * a3)
    rback_vec = min_image_displacement(back_pos[q_idx] - back_pos[p_idx], box)
    rback_mod = rback_vec.norm(dim=-1).clamp(min=1e-12)
    rback_hat = rback_vec / rback_mod.unsqueeze(-1)

    # Particle axes
    a2_p = a2[p_idx]
    a2_q = a2[q_idx]
    a3_p = a3[p_idx]
    a3_q = a3[q_idx]

    # --- f1 radial (Morse-like, sequence-dependent eps) ---
    A      = RC.RNA_F1_A[RC.RNA_STCK_F1].item()
    R0     = RC.RNA_F1_R0[RC.RNA_STCK_F1].item()
    BLOW   = RC.RNA_F1_BLOW[RC.RNA_STCK_F1].item()
    BHIGH  = RC.RNA_F1_BHIGH[RC.RNA_STCK_F1].item()
    RLOW   = RC.RNA_F1_RLOW[RC.RNA_STCK_F1].item()
    RHIGH  = RC.RNA_F1_RHIGH[RC.RNA_STCK_F1].item()
    RCLOW  = RC.RNA_F1_RCLOW[RC.RNA_STCK_F1].item()
    RCHIGH = RC.RNA_F1_RCHIGH[RC.RNA_STCK_F1].item()

    r = r_mod
    eps   = stacking_eps    # (B,)
    shift = stacking_shift  # (B,)

    tmp        = 1.0 - torch.exp(-(r - R0) * A)
    morse      = eps * tmp * tmp - shift
    onset      = eps * BLOW  * (r - RCLOW)  ** 2
    cutoff_reg = eps * BHIGH * (r - RCHIGH) ** 2
    zero       = torch.zeros_like(r)

    f1_val = torch.where(r < RCHIGH,
                 torch.where(r > RHIGH, cutoff_reg,
                     torch.where(r > RLOW, morse,
                         torch.where(r > RCLOW, onset, zero))),
                 zero)

    # --- f4 angular ---

    # t5 = acos(a3_p . r_hat);  C++ applies f4(PI - t5, THETA5)
    cos_t5 = (a3_p * r_hat).sum(-1)
    t5 = safe_acos(cos_t5)
    f4_t5 = _rna_f4(math.pi - t5, RC.RNA_STCK_F4_THETA5)

    # t6 = acos(-a3_q . r_hat)
    cos_t6 = -(a3_q * r_hat).sum(-1)
    t6 = safe_acos(cos_t6)
    f4_t6 = _rna_f4(t6, RC.RNA_STCK_F4_THETA6)

    # tB1 = acos(-rback_dir . BBVECTOR_3_p)
    bbvec3_p = bbvec3_lab[p_idx]
    cos_tB1 = -(rback_hat * bbvec3_p).sum(-1)
    tB1 = safe_acos(cos_tB1)
    f4_tB1 = _rna_f4(tB1, RC.RNA_STCK_F4_THETAB1)

    # tB2 = acos(-rback_dir . BBVECTOR_5_q)
    bbvec5_q = bbvec5_lab[q_idx]
    cos_tB2 = -(rback_hat * bbvec5_q).sum(-1)
    tB2 = safe_acos(cos_tB2)
    f4_tB2 = _rna_f4(tB2, RC.RNA_STCK_F4_THETAB2)

    # --- f5 dihedral angles ---
    # phi1 = a2_p . rback_dir  (C++: cosphi1 = a2 * rback / rbackmod)
    cos_phi1 = (a2_p * rback_hat).sum(-1)
    f5_phi1 = f5(cos_phi1, RC.RNA_STCK_F5_PHI1)

    # phi2 = a2_q . rback_dir  (C++: cosphi2 = b2 * rback / rbackmod)
    cos_phi2 = (a2_q * rback_hat).sum(-1)
    f5_phi2 = f5(cos_phi2, RC.RNA_STCK_F5_PHI2)

    energy_per_pair = (f1_val * f4_t5 * f4_t6
                       * f5_phi1 * f5_phi2 * f4_tB1 * f4_tB2)
    return energy_per_pair.sum()
