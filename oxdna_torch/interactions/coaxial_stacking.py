"""
Coaxial stacking interaction between non-bonded nucleotides.

E_cxst = f2(r_stack) * f4_cxst_t1(theta1) * f4(theta4)
         * f4_sym(theta5) * f4_sym(theta6) * f5(phi3)^2

where:
  f4_cxst_t1(theta1) = f4(theta1) + f4(2*pi - theta1)
  f4_sym(theta) = f4(theta) + f4(-theta) for theta5, theta6

The phi3 angle uses a cross product construction:
  cosphi3 = r_stack_dir . (r_backref_dir x a1)
"""

import torch
from torch import Tensor
from typing import Optional, Dict

from .. import constants as C
from ..smooth import f2, f4_of_cos, f4_of_cos_cxst_t1, f5
from ..utils import safe_norm, dot, cross
from ..pairs import min_image_displacement


def coaxial_stacking_energy(
    positions: Tensor,
    quaternions: Tensor,
    stack_offsets: Tensor,
    nonbonded_pairs: Tensor,
    box: Optional[Tensor] = None,
    params: Optional[Dict[str, Tensor]] = None,
    oxdna2: bool = False,
) -> Tensor:
    """Compute coaxial stacking energy for all non-bonded pairs.

    Args:
        positions: (N, 3) COM positions
        quaternions: (N, 4) unit quaternions
        stack_offsets: (N, 3) stacking site offsets from COM
        nonbonded_pairs: (P, 2) non-bonded pair indices
        box: (3,) periodic box dimensions, or None
        params: optional dict from ParameterStore.as_dict() for learnable params
        oxdna2: if True, use oxDNA2 coaxial stacking (modified K and theta1 function)

    Returns:
        Scalar total coaxial stacking energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    from ..quaternion import quat_to_rotmat

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # COM displacement
    r_com = positions[q_idx] - positions[p_idx]
    r_com = min_image_displacement(r_com, box)

    # Stack-to-stack displacement
    r_stack = r_com + stack_offsets[q_idx] - stack_offsets[p_idx]
    r_stack_mod = safe_norm(r_stack, dim=-1)

    # Distance filter
    in_range = (r_stack_mod > C.CXST_RCLOW) & (r_stack_mod < C.CXST_RCHIGH)

    if not in_range.any():
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    r_stack_dir = r_stack / r_stack_mod.unsqueeze(-1)

    # Get orientation vectors
    R = quat_to_rotmat(quaternions)
    a1_all = R[:, :, 0]
    a2_all = R[:, :, 1]
    a3_all = R[:, :, 2]

    pa1 = a1_all[p_idx]
    pa2 = a2_all[p_idx]
    pa3 = a3_all[p_idx]
    qa1 = a1_all[q_idx]  # b1
    qa3 = a3_all[q_idx]  # b3

    # Reference backbone vector (for phi3 calculation)
    r_backref = r_com + qa1 * C.POS_BACK - pa1 * C.POS_BACK
    r_backref_mod = safe_norm(r_backref, dim=-1)
    r_backref_dir = r_backref / r_backref_mod.unsqueeze(-1)

    # Compute cosines of angles
    cost1 = -dot(pa1, qa1, dim=-1)
    cost4 = dot(pa3, qa3, dim=-1)
    cost5 = dot(pa3, r_stack_dir, dim=-1)
    cost6 = -dot(qa3, r_stack_dir, dim=-1)

    # phi3: cosphi3 = r_stack_dir . (r_backref_dir x a1)
    r_backref_cross_a1 = cross(r_backref_dir, pa1)  # (P, 3)
    cosphi3 = dot(r_stack_dir, r_backref_cross_a1, dim=-1)

    # f2 radial (coaxial stacking type = 1)
    # oxDNA2 uses CXST_K_OXDNA2 = 58.5 instead of CXST_K_OXDNA = 46.0
    if oxdna2 and params is None:
        # Build a minimal params override just for the K value
        import copy
        _params2 = {
            'f2_K': C.F2_K.clone().to(positions.device),
            'f2_R0': C.F2_R0.clone().to(positions.device),
            'f2_RC': C.F2_RC.clone().to(positions.device),
            'f2_BLOW': C.F2_BLOW.clone().to(positions.device),
            'f2_BHIGH': C.F2_BHIGH.clone().to(positions.device),
            'f2_RLOW': C.F2_RLOW.clone().to(positions.device),
            'f2_RHIGH': C.F2_RHIGH.clone().to(positions.device),
            'f2_RCLOW': C.F2_RCLOW.clone().to(positions.device),
            'f2_RCHIGH': C.F2_RCHIGH.clone().to(positions.device),
        }
        _params2['f2_K'] = _params2['f2_K'].clone()
        _params2['f2_K'][C.CXST_F2] = C.CXST_K_OXDNA2
        val_f2 = f2(r_stack_mod, C.CXST_F2, params=_params2)
    else:
        val_f2 = f2(r_stack_mod, C.CXST_F2, params=params)

    # Angular modulations
    # oxDNA1 theta1: f4(acos(t)) + f4(2pi - acos(t))
    # oxDNA2 theta1: f4(acos(t)) + f4_pure_harmonic(acos(t))  [T0 changed to pi-0.25]
    val_f4t1 = f4_of_cos_cxst_t1(cost1, C.CXST_F4_THETA1, params=params, oxdna2=oxdna2)

    val_f4t4 = f4_of_cos(cost4, C.CXST_F4_THETA4, params=params)

    # Symmetric f4 for theta5, theta6: f4(cos) + f4(-cos)
    val_f4t5 = f4_of_cos(cost5, C.CXST_F4_THETA5, params=params) + f4_of_cos(-cost5, C.CXST_F4_THETA5, params=params)
    val_f4t6 = f4_of_cos(cost6, C.CXST_F4_THETA6, params=params) + f4_of_cos(-cost6, C.CXST_F4_THETA6, params=params)

    # f5 azimuthal (squared)
    val_f5phi3 = f5(cosphi3, C.CXST_F5_PHI3, params=params)

    # Total energy per pair: f2 * f4t1 * f4t4 * f4t5 * f4t6 * f5^2
    energy_per_pair = val_f2 * val_f4t1 * val_f4t4 * val_f4t5 * val_f4t6 * val_f5phi3 ** 2

    # Zero out pairs outside distance range
    energy_per_pair = torch.where(in_range, energy_per_pair, torch.zeros_like(energy_per_pair))

    return energy_per_pair.sum()
