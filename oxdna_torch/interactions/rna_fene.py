"""
RNA FENE backbone potential.

Energy = -RNA_FENE_EPS/2 * log(1 - (r - RNA_FENE_R0)^2 / RNA_FENE_DELTA2)

Applied to the distance between backbone sites of bonded neighbours.
The backbone site for nucleotide i is:
    pos_back_i = pos_i + R_i @ [RNA_POS_BACK_a1, RNA_POS_BACK_a2, RNA_POS_BACK_a3]
"""

import torch
from torch import Tensor
from typing import Optional

from .. import rna_constants as RC
from ..pairs import min_image_displacement


def rna_fene_energy(
    positions: Tensor,
    quaternions: Tensor,
    bonded_pairs: Tensor,
    box: Optional[Tensor] = None,
) -> Tensor:
    """Compute the FENE backbone energy for all bonded pairs.

    Args:
        positions:    (N, 3) centre-of-mass positions
        quaternions:  (N, 4) unit quaternions
        bonded_pairs: (B, 2) bonded pair indices, q = p.n3
        box:          (3,) periodic box dimensions, or None

    Returns:
        scalar total FENE energy
    """
    from ..quaternion import quat_to_rotmat

    if bonded_pairs.shape[0] == 0:
        return torch.zeros(1, dtype=positions.dtype, device=positions.device).squeeze()

    R = quat_to_rotmat(quaternions)          # (N, 3, 3)
    a1 = R[:, :, 0]                           # (N, 3)
    a2 = R[:, :, 1]
    a3 = R[:, :, 2]

    # Backbone offset in lab frame: a1*back_a1 + a2*back_a2 + a3*back_a3
    back_offset = (RC.RNA_POS_BACK_a1 * a1
                   + RC.RNA_POS_BACK_a2 * a2
                   + RC.RNA_POS_BACK_a3 * a3)            # (N, 3)
    back_pos = positions + back_offset                   # (N, 3)

    p_idx = bonded_pairs[:, 0]
    q_idx = bonded_pairs[:, 1]

    # Vector from p's backbone site to q's backbone site (with MIC)
    r_vec = back_pos[q_idx] - back_pos[p_idx]           # (B, 3)
    r_vec = min_image_displacement(r_vec, box)
    r = r_vec.norm(dim=-1)                               # (B,)

    dr = r - RC.RNA_FENE_R0
    dr2 = dr * dr

    # Clamp to avoid log(0) or negative argument; blows up naturally outside delta
    arg = 1.0 - dr2 / RC.RNA_FENE_DELTA2
    arg = arg.clamp(min=1e-12)
    energy_per_pair = -RC.RNA_FENE_EPS * 0.5 * torch.log(arg)

    return energy_per_pair.sum()
