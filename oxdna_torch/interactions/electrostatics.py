"""
Debye-Hückel electrostatic interaction (oxDNA2 only).

E_DH = cut_factor * Q * exp(-r_back / lambda) / r_back    for r_back < RHIGH
     = cut_factor * B * (r_back - RC)^2                   for RHIGH <= r_back < RC
     = 0                                                    for r_back >= RC

where:
  r_back = distance between backbone sites of p and q
  lambda = lambda_0 * sqrt(T / T_ref) / sqrt(salt)    (Debye screening length)
  RHIGH  = 3 * lambda                                  (onset of smooth cutoff)
  RC     = x*(Q*x + 3*Q*lambda) / (Q*(x+lambda))      (smooth cutoff endpoint)
  B      = -exp(-x/lambda)*Q^2*(x+lambda)^2 / (4*x^3*lambda^2*Q)  (smoothing coeff)
  cut_factor = 0.5 if terminus, else 1.0  (when half_charged_ends=True)

Source: DNA2Interaction::_debye_huckel, DNA2Interaction::init
"""

import math
import torch
from torch import Tensor
from typing import Optional

from .. import constants as C
from ..pairs import min_image_displacement
from ..utils import safe_norm


def debye_huckel_params(
    temperature: float,
    salt_concentration: float = 0.5,
    prefactor: float = C.DH_PREFACTOR,
    lambda_factor: float = C.DH_LAMBDAFACTOR,
) -> dict:
    """Compute Debye-Hückel derived parameters.

    Args:
        temperature: temperature in oxDNA reduced units (T_K / 3000)
        salt_concentration: molar salt concentration (e.g. 0.5 for 500 mM NaCl)
        prefactor: Q, overall strength prefactor (default 0.0543)
        lambda_factor: lambda_0 at T_ref=300K, I=1M (default 0.3616455)

    Returns:
        dict with keys: lambda_, minus_kappa, rhigh, rc, b
    """
    # Debye length: lambda = lambda_0 * sqrt(T/T_ref) / sqrt(salt)
    lambda_ = lambda_factor * math.sqrt(temperature / C.DH_T_REF) / math.sqrt(salt_concentration)
    minus_kappa = -1.0 / lambda_

    # Smooth cutoff onset
    rhigh = 3.0 * lambda_

    # Smooth cutoff endpoint (analytical derivation from C++ code)
    x = rhigh
    q = prefactor
    l = lambda_
    rc = x * (q * x + 3.0 * q * l) / (q * (x + l))

    # Smoothing coefficient B (ensures continuity of E and dE/dr at rhigh)
    b = -(math.exp(-x / l) * q * q * (x + l) * (x + l)) / (-4.0 * x * x * x * l * l * q)

    return {
        'lambda_': lambda_,
        'minus_kappa': minus_kappa,
        'rhigh': rhigh,
        'rc': rc,
        'b': b,
        'prefactor': prefactor,
    }


def debye_huckel_energy(
    positions: Tensor,
    back_offsets: Tensor,
    nonbonded_pairs: Tensor,
    terminus_mask: Tensor,
    dh_params: dict,
    box: Optional[Tensor] = None,
    half_charged_ends: bool = True,
) -> Tensor:
    """Compute Debye-Hückel electrostatic energy for all non-bonded pairs.

    Args:
        positions: (N, 3) COM positions
        back_offsets: (N, 3) backbone site offsets from COM
        nonbonded_pairs: (P, 2) non-bonded pair indices
        terminus_mask: (N,) bool tensor, True if nucleotide is at a strand terminus
        dh_params: dict from debye_huckel_params()
        box: (3,) periodic box dimensions, or None
        half_charged_ends: if True, terminus nucleotides contribute half charge

    Returns:
        Scalar total Debye-Hückel energy
    """
    if nonbonded_pairs.shape[0] == 0:
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    rhigh = dh_params['rhigh']
    rc = dh_params['rc']
    b = dh_params['b']
    minus_kappa = dh_params['minus_kappa']
    q = dh_params['prefactor']

    p_idx = nonbonded_pairs[:, 0]
    q_idx = nonbonded_pairs[:, 1]

    # Backbone-to-backbone displacement
    r_com = positions[q_idx] - positions[p_idx]
    r_com = min_image_displacement(r_com, box)
    r_back = r_com + back_offsets[q_idx] - back_offsets[p_idx]
    r_back_mod = safe_norm(r_back, dim=-1)

    # Only evaluate within the hard cutoff
    in_range = r_back_mod < rc

    if not in_range.any():
        return torch.tensor(0.0, dtype=positions.dtype, device=positions.device)

    # Half-charged-ends: cut_factor = product of 0.5 for each terminus partner
    if half_charged_ends:
        pterm = terminus_mask[p_idx].float()  # 1.0 if terminus
        qterm = terminus_mask[q_idx].float()
        cut_factor = (1.0 - 0.5 * pterm) * (1.0 - 0.5 * qterm)
    else:
        cut_factor = torch.ones(nonbonded_pairs.shape[0],
                                dtype=positions.dtype, device=positions.device)

    # Energy: Yukawa (r < rhigh) or quadratic cutoff (rhigh <= r < rc)
    r = r_back_mod.clamp(min=1e-9)

    yukawa = torch.exp(minus_kappa * r) * (q / r)
    quadratic = b * (r - rc) ** 2

    energy_per_pair = torch.where(r < rhigh, yukawa, quadratic)
    energy_per_pair = energy_per_pair * cut_factor

    # Zero outside range
    energy_per_pair = torch.where(in_range, energy_per_pair,
                                  torch.zeros_like(energy_per_pair))

    return energy_per_pair.sum()
