"""
RNA excluded-volume interactions.

Both bonded (nearest-neighbour) and non-bonded pairs use the same
repulsive-LJ form as oxDNA, but with RNA-specific S/R/B/RC values.

Four pair types (matching RNAInteraction.cpp):
  Type 1: back-back   (S1, R1, B1, RC1)
  Type 2: base-base   (S2, R2, B2, RC2)
  Type 3: p_base vs q_back  (S3, R3, B3, RC3)
  Type 4: p_back vs q_base  (S4, R4, B4, RC4)

The bonded excluded volume omits back-back (that is handled by FENE).
The nonbonded excluded volume uses all four types.
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from .. import rna_constants as RC
from ..smooth import repulsive_lj
from ..pairs import min_image_displacement


def _compute_rna_sites(
    positions: Tensor,
    quaternions: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Return backbone and base site positions for all nucleotides.

    back_pos = pos + R @ (back_a1, back_a2, back_a3)
    base_pos = pos + R @ (RNA_POS_BASE, 0, 0)  i.e. along a1

    Returns:
        back_pos: (N, 3)
        base_pos: (N, 3)
    """
    from ..quaternion import quat_to_rotmat
    R = quat_to_rotmat(quaternions)   # (N, 3, 3)
    a1 = R[:, :, 0]
    a2 = R[:, :, 1]
    a3 = R[:, :, 2]

    back_pos = (positions
                + RC.RNA_POS_BACK_a1 * a1
                + RC.RNA_POS_BACK_a2 * a2
                + RC.RNA_POS_BACK_a3 * a3)
    base_pos = positions + RC.RNA_POS_BASE * a1
    return back_pos, base_pos


def rna_bonded_excluded_volume_energy(
    positions: Tensor,
    quaternions: Tensor,
    bonded_pairs: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Bonded excluded-volume energy (between consecutive nucleotides).

    Only base-base (type 2), p_base/q_back (type 3), p_back/q_base (type 4).
    (Back-back is covered by FENE.)

    Args:
        positions:    (N, 3)
        quaternions:  (N, 4)
        bonded_pairs: (B, 2)
        box:          (3,) periodic box dimensions, or None

    Returns:
        scalar energy
    """
    if bonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    back_pos, base_pos = _compute_rna_sites(positions, quaternions)

    p_idx = bonded_pairs[:, 0]
    q_idx = bonded_pairs[:, 1]

    # base-base
    r_bb = min_image_displacement(base_pos[q_idx] - base_pos[p_idx], box)
    e = repulsive_lj(r_bb.pow(2).sum(-1),
                     RC.RNA_EXCL_S2, RC.RNA_EXCL_R2,
                     RC.RNA_EXCL_B2, RC.RNA_EXCL_RC2)

    # p_base vs q_back
    r_bk = min_image_displacement(back_pos[q_idx] - base_pos[p_idx], box)
    e = e + repulsive_lj(r_bk.pow(2).sum(-1),
                         RC.RNA_EXCL_S3, RC.RNA_EXCL_R3,
                         RC.RNA_EXCL_B3, RC.RNA_EXCL_RC3)

    # p_back vs q_base
    r_kb = min_image_displacement(base_pos[q_idx] - back_pos[p_idx], box)
    e = e + repulsive_lj(r_kb.pow(2).sum(-1),
                         RC.RNA_EXCL_S4, RC.RNA_EXCL_R4,
                         RC.RNA_EXCL_B4, RC.RNA_EXCL_RC4)

    return e.sum()


def rna_nonbonded_excluded_volume_energy(
    positions: Tensor,
    quaternions: Tensor,
    nonbonded_pairs: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Non-bonded excluded-volume energy for all non-bonded pairs.

    All four types: back-back, base-base, p_base/q_back, p_back/q_base.

    Args:
        positions:       (N, 3)
        quaternions:     (N, 4)
        nonbonded_pairs: (P, 2)
        box:             (3,) periodic box dimensions, or None

    Returns:
        scalar energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype,
                           device=positions.device).squeeze()

    back_pos, base_pos = _compute_rna_sites(positions, quaternions)

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # back-back
    r_kk = min_image_displacement(back_pos[q_idx] - back_pos[p_idx], box)
    e = repulsive_lj(r_kk.pow(2).sum(-1),
                     RC.RNA_EXCL_S1, RC.RNA_EXCL_R1,
                     RC.RNA_EXCL_B1, RC.RNA_EXCL_RC1)

    # base-base
    r_bb = min_image_displacement(base_pos[q_idx] - base_pos[p_idx], box)
    e = e + repulsive_lj(r_bb.pow(2).sum(-1),
                         RC.RNA_EXCL_S2, RC.RNA_EXCL_R2,
                         RC.RNA_EXCL_B2, RC.RNA_EXCL_RC2)

    # p_base vs q_back
    r_bk = min_image_displacement(back_pos[q_idx] - base_pos[p_idx], box)
    e = e + repulsive_lj(r_bk.pow(2).sum(-1),
                         RC.RNA_EXCL_S3, RC.RNA_EXCL_R3,
                         RC.RNA_EXCL_B3, RC.RNA_EXCL_RC3)

    # p_back vs q_base
    r_kb = min_image_displacement(base_pos[q_idx] - back_pos[p_idx], box)
    e = e + repulsive_lj(r_kb.pow(2).sum(-1),
                         RC.RNA_EXCL_S4, RC.RNA_EXCL_R4,
                         RC.RNA_EXCL_B4, RC.RNA_EXCL_RC4)

    return e.sum()
