"""
Smooth potential functions f1, f2, f4, f5 for the oxDNA model.

All functions are implemented as pure PyTorch operations using torch.where
for piecewise regions. This ensures:
1. Differentiability everywhere (autograd handles all derivatives)
2. GPU-compatible vectorized computation
3. No need for mesh interpolation tables

Reference: DNAInteraction.cpp _f1, _f2, _f4, _f5 and their derivatives.

Learnable parameters:
    All functions accept an optional `params` dict (from ParameterStore.as_dict()).
    When params is None, constants are extracted via .item() (Python float, no gradient).
    When params is provided, tensor indexing preserves the autograd graph.
"""

import torch
from torch import Tensor
from typing import Optional, Dict
from . import constants as C


def f1(r: Tensor, f1_type: int, eps: Tensor, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Morse-like radial potential with smooth boundaries.

    f1 has 4 regions:
      r < RCLOW:  0
      RCLOW < r < RLOW:   eps * BLOW * (r - RCLOW)^2       (smooth onset)
      RLOW < r < RHIGH:   eps * (1 - exp(-(r-R0)*A))^2 - SHIFT  (Morse well)
      RHIGH < r < RCHIGH: eps * BHIGH * (r - RCHIGH)^2     (smooth cutoff)
      r > RCHIGH: 0

    Args:
        r: (...) distances
        f1_type: 0 for HYDR, 1 for STCK (selects parameter set)
        eps: (...) epsilon values (may be sequence-dependent)
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        (...) energy values
    """
    if params is not None:
        A = params['f1_A'][f1_type]
        RC = params['f1_RC'][f1_type]
        R0 = params['f1_R0'][f1_type]
        BLOW = params['f1_BLOW'][f1_type]
        BHIGH = params['f1_BHIGH'][f1_type]
        RLOW = params['f1_RLOW'][f1_type]
        RHIGH = params['f1_RHIGH'][f1_type]
        RCLOW = params['f1_RCLOW'][f1_type]
        RCHIGH = params['f1_RCHIGH'][f1_type]
        # Tensor expression preserves gradient through A, RC, R0
        shift = eps * (1.0 - torch.exp(-(RC - R0) * A)) ** 2
    else:
        A = C.F1_A[f1_type].item()
        RC = C.F1_RC[f1_type].item()
        R0 = C.F1_R0[f1_type].item()
        BLOW = C.F1_BLOW[f1_type].item()
        BHIGH = C.F1_BHIGH[f1_type].item()
        RLOW = C.F1_RLOW[f1_type].item()
        RHIGH = C.F1_RHIGH[f1_type].item()
        RCLOW = C.F1_RCLOW[f1_type].item()
        RCHIGH = C.F1_RCHIGH[f1_type].item()
        # Compute SHIFT = eps * (1 - exp(-(RC - R0)*A))^2
        shift = eps * (1.0 - torch.exp(torch.tensor(-(RC - R0) * A, dtype=r.dtype, device=r.device))) ** 2

    # Region 2: Morse potential
    tmp = 1.0 - torch.exp(-(r - R0) * A)
    morse = eps * tmp * tmp - shift

    # Region 1: smooth onset
    onset = eps * BLOW * (r - RCLOW) ** 2

    # Region 3: smooth cutoff
    cutoff = eps * BHIGH * (r - RCHIGH) ** 2

    zero = torch.zeros_like(r)

    # Build piecewise: innermost conditions first, then override
    val = torch.where(r < RCHIGH,
                      torch.where(r > RHIGH, cutoff,
                                  torch.where(r > RLOW, morse,
                                              torch.where(r > RCLOW, onset, zero))),
                      zero)
    return val


def f2(r: Tensor, f2_type: int, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Harmonic radial potential with smooth boundaries.

    f2 has 4 regions:
      r < RCLOW:  0
      RCLOW < r < RLOW:   K * BLOW * (r - RCLOW)^2         (smooth onset)
      RLOW < r < RHIGH:   (K/2) * ((r - R0)^2 - (RC - R0)^2)  (harmonic well)
      RHIGH < r < RCHIGH: K * BHIGH * (r - RCHIGH)^2       (smooth cutoff)
      r > RCHIGH: 0

    Args:
        r: (...) distances
        f2_type: 0 for CRST, 1 for CXST
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        (...) energy values
    """
    if params is not None:
        K = params['f2_K'][f2_type]
        RC = params['f2_RC'][f2_type]
        R0 = params['f2_R0'][f2_type]
        BLOW = params['f2_BLOW'][f2_type]
        BHIGH = params['f2_BHIGH'][f2_type]
        RLOW = params['f2_RLOW'][f2_type]
        RHIGH = params['f2_RHIGH'][f2_type]
        RCLOW = params['f2_RCLOW'][f2_type]
        RCHIGH = params['f2_RCHIGH'][f2_type]
    else:
        K = C.F2_K[f2_type].item()
        RC = C.F2_RC[f2_type].item()
        R0 = C.F2_R0[f2_type].item()
        BLOW = C.F2_BLOW[f2_type].item()
        BHIGH = C.F2_BHIGH[f2_type].item()
        RLOW = C.F2_RLOW[f2_type].item()
        RHIGH = C.F2_RHIGH[f2_type].item()
        RCLOW = C.F2_RCLOW[f2_type].item()
        RCHIGH = C.F2_RCHIGH[f2_type].item()

    # Region 2: harmonic
    harmonic = (K / 2.0) * ((r - R0) ** 2 - (RC - R0) ** 2)

    # Region 1: smooth onset
    onset = K * BLOW * (r - RCLOW) ** 2

    # Region 3: smooth cutoff
    cutoff = K * BHIGH * (r - RCHIGH) ** 2

    zero = torch.zeros_like(r)

    val = torch.where(r < RCHIGH,
                      torch.where(r > RHIGH, cutoff,
                                  torch.where(r > RLOW, harmonic,
                                              torch.where(r > RCLOW, onset, zero))),
                      zero)
    return val


def f4(theta: Tensor, f4_type: int, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Angular modulation function.

    Parabolic modulation centered at T0 with smooth cutoff:
      |t| = |theta - T0|
      |t| > TC:           0
      TS < |t| < TC:      B * (TC - |t|)^2        (smooth cutoff)
      |t| < TS:           1 - A * |t|^2           (parabolic well)

    The C++ code evaluates this on cos(theta) via mesh interpolation,
    but here we evaluate it directly on theta (and let autograd handle
    the chain rule through the arccos).

    Args:
        theta: (...) angles in radians
        f4_type: index 0-12 selecting parameter set
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        (...) modulation values in [0, 1]
    """
    if params is not None:
        A = params['f4_A'][f4_type]
        B = params['f4_B'][f4_type]
        T0 = params['f4_T0'][f4_type]
        TS = params['f4_TS'][f4_type]
        TC = params['f4_TC'][f4_type]
    else:
        A = C.F4_THETA_A[f4_type].item()
        B = C.F4_THETA_B[f4_type].item()
        T0 = C.F4_THETA_T0[f4_type].item()
        TS = C.F4_THETA_TS[f4_type].item()
        TC = C.F4_THETA_TC[f4_type].item()

    t = torch.abs(theta - T0)

    parabola = 1.0 - A * t * t
    smooth = B * (TC - t) ** 2
    zero = torch.zeros_like(theta)

    val = torch.where(t < TC,
                      torch.where(t > TS, smooth, parabola),
                      zero)
    return val


def f4_of_cos(cos_theta: Tensor, f4_type: int, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """f4 evaluated on cos(theta), matching the C++ mesh-based evaluation.

    In oxDNA, the f4 angular meshes are built as functions of cos(theta),
    i.e., the mesh stores f4(acos(cos_theta)).

    For most interaction terms, the code calls _custom_f4(cos_val, type)
    which queries the mesh built from _fakef4 = f4(acos(t)).

    We implement this directly by computing theta = acos(cos_theta)
    then calling f4. This is differentiable through autograd.

    Args:
        cos_theta: (...) cosine of angle
        f4_type: index 0-12
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        (...) modulation values
    """
    from .utils import safe_acos
    theta = safe_acos(cos_theta)
    return f4(theta, f4_type, params=params)


def f4_pure_harmonic(theta: Tensor, sa: float, sb: float) -> Tensor:
    """Pure-harmonic correction term used by oxDNA2 CXST theta1.

    Computes:  SA * max(theta - SB, 0)^2

    This adds a one-sided harmonic ramp above SB that smoothly extends
    the f4(theta1) well to larger angles without disrupting the potential.
    Source: DNA2Interaction::_f4_pure_harmonic

    Args:
        theta: (...) angle in radians
        sa: harmonic prefactor  (CXST_THETA1_SA = 20.0)
        sb: onset angle in radians  (CXST_THETA1_SB = pi - 0.025)

    Returns:
        (...) correction values (>= 0)
    """
    tt0 = theta - sb
    return torch.where(tt0 > 0.0, sa * tt0 * tt0, torch.zeros_like(theta))


def f4_of_cos_cxst_t1(
    cos_theta: Tensor,
    f4_type: int,
    params: Optional[Dict[str, Tensor]] = None,
    oxdna2: bool = False,
) -> Tensor:
    """Special f4 for coaxial stacking theta1.

    oxDNA1:
      _fakef4_cxst_t1(t) = f4(acos(t)) + f4(2*pi - acos(t))

    oxDNA2:
      _fakef4_cxst_t1(t) = f4(acos(t)) + f4_pure_harmonic(acos(t))

    In oxDNA2 the T0 for index 10 is changed to pi-0.25 (set at model level)
    and the second term is the pure-harmonic correction, not the mirror image.

    Args:
        cos_theta: (...) cosine of angle
        f4_type: should be CXST_F4_THETA1 = 10
        params: optional dict from ParameterStore.as_dict() for learnable params
        oxdna2: if True, use oxDNA2 formula (f4 + pure_harmonic)

    Returns:
        (...) modulation values
    """
    import math
    from .utils import safe_acos
    from . import constants as C
    theta = safe_acos(cos_theta)
    if oxdna2:
        return (f4(theta, f4_type, params=params)
                + f4_pure_harmonic(theta, C.CXST_THETA1_SA, C.CXST_THETA1_SB))
    return f4(theta, f4_type, params=params) + f4(2.0 * math.pi - theta, f4_type, params=params)


def f5(cos_phi: Tensor, f5_type: int, params: Optional[Dict[str, Tensor]] = None) -> Tensor:
    """Azimuthal modulation function.

    Depends on cos(phi) rather than phi directly:
      cos_phi < XC:         0
      XC < cos_phi < XS:    B * (XC - cos_phi)^2   (smooth onset)
      XS < cos_phi < 0:     1 - A * cos_phi^2       (parabolic)
      cos_phi >= 0:          1                        (flat)

    Args:
        cos_phi: (...) cosine of azimuthal angle
        f5_type: index 0-3 selecting parameter set
        params: optional dict from ParameterStore.as_dict() for learnable params

    Returns:
        (...) modulation values in [0, 1]
    """
    if params is not None:
        A = params['f5_A'][f5_type]
        B = params['f5_B'][f5_type]
        XC = params['f5_XC'][f5_type]
        XS = params['f5_XS'][f5_type]
    else:
        A = C.F5_PHI_A[f5_type].item()
        B = C.F5_PHI_B[f5_type].item()
        XC = C.F5_PHI_XC[f5_type].item()
        XS = C.F5_PHI_XS[f5_type].item()

    parabola = 1.0 - A * cos_phi * cos_phi
    smooth = B * (XC - cos_phi) ** 2
    one = torch.ones_like(cos_phi)
    zero = torch.zeros_like(cos_phi)

    val = torch.where(cos_phi > XC,
                      torch.where(cos_phi < XS, smooth,
                                  torch.where(cos_phi < 0.0, parabola, one)),
                      zero)
    return val


def repulsive_lj(r_sq: Tensor, sigma: float, rstar: float, b: float, rc: float,
                 excl_eps: Optional[Tensor] = None) -> Tensor:
    """Repulsive Lennard-Jones potential with smooth cutoff.

    Two regions:
      r > rc:               0
      rstar < r < rc:       EXCL_EPS * b * (r - rc)^2    (quadratic smoothing)
      r < rstar:            4 * EXCL_EPS * (sigma^6/r^6)^2 - sigma^6/r^6)  (LJ)

    Args:
        r_sq: (...) squared distances between interaction sites
        sigma: LJ sigma parameter
        rstar: transition distance from LJ to quadratic
        b: quadratic smoothing coefficient
        rc: cutoff distance
        excl_eps: optional learnable epsilon (scalar tensor). If None, uses C.EXCL_EPS.

    Returns:
        (...) energy values (always >= 0)
    """
    eps = excl_eps if excl_eps is not None else C.EXCL_EPS

    rc_sq = rc * rc
    rstar_sq = rstar * rstar
    zero = torch.zeros_like(r_sq)

    # LJ region
    sigma_sq = sigma * sigma
    lj_part = (sigma_sq / r_sq.clamp(min=1e-18)) ** 3  # (sigma/r)^6
    lj_energy = 4.0 * eps * (lj_part * lj_part - lj_part)

    # Quadratic region
    r_mod = torch.sqrt(r_sq.clamp(min=1e-18))
    quad_energy = eps * b * (r_mod - rc) ** 2

    val = torch.where(r_sq < rc_sq,
                      torch.where(r_sq > rstar_sq, quad_energy, lj_energy),
                      zero)
    return val
