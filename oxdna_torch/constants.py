"""
All oxDNA1 model constants.

Transcribed directly from oxDNA src/model.h and src/Interactions/DNAInteraction.cpp.
All values are in oxDNA reduced units:
  - Length unit: ~8.518 Angstroms (roughly the distance between consecutive phosphates)
  - Energy unit: kT at 3000 K  (so T_reduced = T_kelvin / 3000)
  - Time unit: derived from length and energy units

Base type encoding: A=0, C=1, G=2, T=3  (dummy=4)
Watson-Crick pairs: A-T (0+3=3), C-G (1+2=3)
"""

import torch
import math

# ============================================================
# Geometry: positions of interaction centers along the a1 axis
# ============================================================
POS_BACK = -0.4       # Backbone center offset (without major-minor grooving)
POS_MM_BACK1 = -0.34  # Backbone offset with grooving (component along a1)
POS_MM_BACK2 = 0.3408 # Backbone offset with grooving (component along a2)
POS_STACK = 0.34      # Stacking center offset
POS_BASE = 0.4        # Base center offset
GAMMA = 0.74          # POS_STACK - POS_BACK

# ============================================================
# FENE backbone potential
# ============================================================
FENE_EPS = 2.0
FENE_R0_OXDNA = 0.7525
FENE_R0_OXDNA2 = 0.7564
FENE_DELTA = 0.25
FENE_DELTA2 = 0.0625  # FENE_DELTA^2

# ============================================================
# Excluded volume (repulsive Lennard-Jones)
# ============================================================
EXCL_EPS = 2.0

# Back-Back
EXCL_S1 = 0.70
EXCL_R1 = 0.675
EXCL_B1 = 892.016223343
EXCL_RC1 = 0.711879214356

# Base-Base
EXCL_S2 = 0.33
EXCL_R2 = 0.32
EXCL_B2 = 4119.70450017
EXCL_RC2 = 0.335388426126

# Base-Back (p_base - q_back)
EXCL_S3 = 0.515
EXCL_R3 = 0.50
EXCL_B3 = 1707.30627298
EXCL_RC3 = 0.52329943261

# Back-Base (p_back - q_base)
EXCL_S4 = 0.515
EXCL_R4 = 0.50
EXCL_B4 = 1707.30627298
EXCL_RC4 = 0.52329943261

# ============================================================
# Hydrogen bonding (f1 radial, type=0)
# ============================================================
HYDR_F1 = 0  # f1 type index for hydrogen bonding
HYDR_EPS_OXDNA = 1.077
HYDR_EPS_OXDNA2 = 1.0678
HYDR_A = 8.0
HYDR_RC = 0.75
HYDR_R0 = 0.4
HYDR_BLOW = -126.243
HYDR_BHIGH = -7.87708
HYDR_RLOW = 0.34
HYDR_RHIGH = 0.7
HYDR_RCLOW = 0.276908
HYDR_RCHIGH = 0.783775

# f4 angular indices for hydrogen bonding
HYDR_F4_THETA1 = 2
HYDR_F4_THETA2 = 3
HYDR_F4_THETA3 = 3  # same mesh as theta2
HYDR_F4_THETA4 = 4
HYDR_F4_THETA7 = 5
HYDR_F4_THETA8 = 5  # same mesh as theta7

# ============================================================
# Stacking (f1 radial, type=1)
# ============================================================
STCK_F1 = 1  # f1 type index for stacking
STCK_BASE_EPS_OXDNA = 1.3448
STCK_BASE_EPS_OXDNA2 = 1.3523
STCK_FACT_EPS_OXDNA = 2.6568
STCK_FACT_EPS_OXDNA2 = 2.6717
STCK_A = 6.0
STCK_RC = 0.9
STCK_R0 = 0.4
STCK_BLOW = -68.1857
STCK_BHIGH = -3.12992
STCK_RLOW = 0.32
STCK_RHIGH = 0.75
STCK_RCLOW = 0.23239
STCK_RCHIGH = 0.956

# f4 angular indices for stacking
STCK_F4_THETA4 = 0
STCK_F4_THETA5 = 1
STCK_F4_THETA6 = 1  # same mesh as theta5

# f5 azimuthal indices for stacking
STCK_F5_PHI1 = 0
STCK_F5_PHI2 = 1

# ============================================================
# Cross stacking (f2 radial, type=0)
# ============================================================
CRST_F2 = 0  # f2 type index
CRST_R0 = 0.575
CRST_RC = 0.675
CRST_K = 47.5
CRST_BLOW = -0.888889
CRST_RLOW = 0.495
CRST_RCLOW = 0.45
CRST_BHIGH = -0.888889
CRST_RHIGH = 0.655
CRST_RCHIGH = 0.7

# f4 angular indices for cross stacking
CRST_F4_THETA1 = 6
CRST_F4_THETA2 = 7
CRST_F4_THETA3 = 7  # same mesh as theta2
CRST_F4_THETA4 = 8
CRST_F4_THETA7 = 9
CRST_F4_THETA8 = 9  # same mesh as theta7

# ============================================================
# Coaxial stacking (f2 radial, type=1)
# ============================================================
CXST_F2 = 1  # f2 type index
CXST_R0 = 0.400
CXST_RC = 0.6
CXST_K_OXDNA = 46.0
CXST_K_OXDNA2 = 58.5
CXST_BLOW = -2.13158
CXST_RLOW = 0.22
CXST_RCLOW = 0.177778
CXST_BHIGH = -2.13158
CXST_RHIGH = 0.58
CXST_RCHIGH = 0.6222222

# oxDNA2 coaxial stacking theta1 parameters (pure-harmonic correction)
# Source: oxDNA model.h CXST_THETA1_T0_OXDNA2, CXST_THETA1_SA, CXST_THETA1_SB
CXST_THETA1_T0_OXDNA2 = math.pi - 0.25
CXST_THETA1_SA = 20.0
CXST_THETA1_SB = math.pi - 0.1 * (math.pi - (math.pi - 0.25))  # pi - 0.025

# f4 angular indices for coaxial stacking
CXST_F4_THETA1 = 10
CXST_F4_THETA4 = 11
CXST_F4_THETA5 = 12
CXST_F4_THETA6 = 12  # same mesh as theta5

# f5 azimuthal indices for coaxial stacking
CXST_F5_PHI3 = 2
CXST_F5_PHI4 = 3

# ============================================================
# Debye-Hückel electrostatics (oxDNA2 only)
# ============================================================
DH_PREFACTOR = 0.0543          # Q: overall strength prefactor
DH_LAMBDAFACTOR = 0.3616455    # lambda_0 at T=300K, I=1M (in oxDNA length units)
DH_T_REF = 0.1                 # reference temperature (300K in oxDNA units = 300/3000)
# Default salt: 0.5 M (set at runtime via OxDNA2Energy salt_concentration arg)

# ============================================================
# f4 angular function parameters (13 parameter sets, indexed 0-12)
# Each set has: A, B, T0, TS, TC
# ============================================================
F4_THETA_A = torch.tensor([
    # 0: STCK_THETA4
    1.3,
    # 1: STCK_THETA5 (and THETA6)
    0.9,
    # 2: HYDR_THETA1
    1.5,
    # 3: HYDR_THETA2 (and THETA3)
    1.5,
    # 4: HYDR_THETA4
    0.46,
    # 5: HYDR_THETA7 (and THETA8)
    4.0,
    # 6: CRST_THETA1
    2.25,
    # 7: CRST_THETA2 (and THETA3)
    1.70,
    # 8: CRST_THETA4
    1.50,
    # 9: CRST_THETA7 (and THETA8)
    1.70,
    # 10: CXST_THETA1
    2.0,
    # 11: CXST_THETA4
    1.3,
    # 12: CXST_THETA5 (and THETA6)
    0.9,
], dtype=torch.float64)

F4_THETA_B = torch.tensor([
    6.4381,      # 0: STCK_THETA4
    3.89361,     # 1: STCK_THETA5
    4.16038,     # 2: HYDR_THETA1
    4.16038,     # 3: HYDR_THETA2
    0.133855,    # 4: HYDR_THETA4
    17.0526,     # 5: HYDR_THETA7
    7.00545,     # 6: CRST_THETA1
    6.2469,      # 7: CRST_THETA2
    2.59556,     # 8: CRST_THETA4
    6.2469,      # 9: CRST_THETA7
    10.9032,     # 10: CXST_THETA1
    6.4381,      # 11: CXST_THETA4
    3.89361,     # 12: CXST_THETA5
], dtype=torch.float64)

F4_THETA_T0 = torch.tensor([
    0.0,                    # 0: STCK_THETA4
    0.0,                    # 1: STCK_THETA5
    0.0,                    # 2: HYDR_THETA1
    0.0,                    # 3: HYDR_THETA2
    math.pi,                # 4: HYDR_THETA4
    math.pi * 0.5,          # 5: HYDR_THETA7
    math.pi - 2.35,         # 6: CRST_THETA1
    1.0,                    # 7: CRST_THETA2
    0.0,                    # 8: CRST_THETA4
    0.875,                  # 9: CRST_THETA7
    math.pi - 0.60,         # 10: CXST_THETA1 (oxDNA1 value)
    0.0,                    # 11: CXST_THETA4
    0.0,                    # 12: CXST_THETA5
], dtype=torch.float64)

F4_THETA_TS = torch.tensor([
    0.8,     # 0: STCK_THETA4
    0.95,    # 1: STCK_THETA5
    0.7,     # 2: HYDR_THETA1
    0.7,     # 3: HYDR_THETA2
    0.7,     # 4: HYDR_THETA4
    0.45,    # 5: HYDR_THETA7
    0.58,    # 6: CRST_THETA1
    0.68,    # 7: CRST_THETA2
    0.65,    # 8: CRST_THETA4
    0.68,    # 9: CRST_THETA7
    0.65,    # 10: CXST_THETA1
    0.8,     # 11: CXST_THETA4
    0.95,    # 12: CXST_THETA5
], dtype=torch.float64)

F4_THETA_TC = torch.tensor([
    0.961538,   # 0: STCK_THETA4
    1.16959,    # 1: STCK_THETA5
    0.952381,   # 2: HYDR_THETA1
    0.952381,   # 3: HYDR_THETA2
    3.10559,    # 4: HYDR_THETA4
    0.555556,   # 5: HYDR_THETA7
    0.766284,   # 6: CRST_THETA1
    0.865052,   # 7: CRST_THETA2
    1.02564,    # 8: CRST_THETA4
    0.865052,   # 9: CRST_THETA7
    0.769231,   # 10: CXST_THETA1
    0.961538,   # 11: CXST_THETA4
    1.16959,    # 12: CXST_THETA5
], dtype=torch.float64)

# ============================================================
# f5 azimuthal function parameters (4 parameter sets, indexed 0-3)
# Each set has: A, B, XC, XS
# ============================================================
F5_PHI_A = torch.tensor([
    2.0,    # 0: STCK_PHI1
    2.0,    # 1: STCK_PHI2
    2.0,    # 2: CXST_PHI3
    2.0,    # 3: CXST_PHI4
], dtype=torch.float64)

F5_PHI_B = torch.tensor([
    10.9032,   # 0: STCK_PHI1
    10.9032,   # 1: STCK_PHI2
    10.9032,   # 2: CXST_PHI3
    10.9032,   # 3: CXST_PHI4 (note: C++ code uses PHI3_B for index 3)
], dtype=torch.float64)

F5_PHI_XC = torch.tensor([
    -0.769231,  # 0: STCK_PHI1
    -0.769231,  # 1: STCK_PHI2
    -0.769231,  # 2: CXST_PHI3
    -0.769231,  # 3: CXST_PHI4
], dtype=torch.float64)

F5_PHI_XS = torch.tensor([
    -0.65,   # 0: STCK_PHI1
    -0.65,   # 1: STCK_PHI2
    -0.65,   # 2: CXST_PHI3
    -0.65,   # 3: CXST_PHI4
], dtype=torch.float64)

# ============================================================
# f1 radial function parameters (2 types: HYDR=0, STCK=1)
# ============================================================
F1_A = torch.tensor([HYDR_A, STCK_A], dtype=torch.float64)
F1_RC = torch.tensor([HYDR_RC, STCK_RC], dtype=torch.float64)
F1_R0 = torch.tensor([HYDR_R0, STCK_R0], dtype=torch.float64)
F1_BLOW = torch.tensor([HYDR_BLOW, STCK_BLOW], dtype=torch.float64)
F1_BHIGH = torch.tensor([HYDR_BHIGH, STCK_BHIGH], dtype=torch.float64)
F1_RLOW = torch.tensor([HYDR_RLOW, STCK_RLOW], dtype=torch.float64)
F1_RHIGH = torch.tensor([HYDR_RHIGH, STCK_RHIGH], dtype=torch.float64)
F1_RCLOW = torch.tensor([HYDR_RCLOW, STCK_RCLOW], dtype=torch.float64)
F1_RCHIGH = torch.tensor([HYDR_RCHIGH, STCK_RCHIGH], dtype=torch.float64)

# ============================================================
# f2 radial function parameters (2 types: CRST=0, CXST=1)
# ============================================================
F2_K = torch.tensor([CRST_K, CXST_K_OXDNA], dtype=torch.float64)
F2_RC = torch.tensor([CRST_RC, CXST_RC], dtype=torch.float64)
F2_R0 = torch.tensor([CRST_R0, CXST_R0], dtype=torch.float64)
F2_BLOW = torch.tensor([CRST_BLOW, CXST_BLOW], dtype=torch.float64)
F2_BHIGH = torch.tensor([CRST_BHIGH, CXST_BHIGH], dtype=torch.float64)
F2_RLOW = torch.tensor([CRST_RLOW, CXST_RLOW], dtype=torch.float64)
F2_RHIGH = torch.tensor([CRST_RHIGH, CXST_RHIGH], dtype=torch.float64)
F2_RCLOW = torch.tensor([CRST_RCLOW, CXST_RCLOW], dtype=torch.float64)
F2_RCHIGH = torch.tensor([CRST_RCHIGH, CXST_RCHIGH], dtype=torch.float64)

# ============================================================
# Sequence-dependent stacking parameters
# Format: STCK_EPS_SEQ[i][j] for 3'→5' stacking where i=3' base, j=5' base
# Base encoding: A=0, C=1, G=2, T=3
# These are the "raw" epsilon values from oxDNA1_sequence_dependent_parameters.txt
# The temperature-dependent epsilon is: eps * (1 - STCK_FACT_EPS_SEQ + T_reduced * 9 * STCK_FACT_EPS_SEQ)
# where STCK_FACT_EPS_SEQ = 0.18
# ============================================================
STCK_FACT_EPS_SEQ = 0.18

# [n3_type][n5_type] -> epsilon
# Rows: A=0, C=1, G=2, T=3 (3' base)
# Cols: A=0, C=1, G=2, T=3 (5' base)
STCK_EPS_SEQ = torch.tensor([
    # n5:  A       C       G       T
    [1.709, 1.671, 1.610, 1.553],  # n3=A: AA, AC, AG, AT
    [1.654, 1.604, 1.737, 1.610],  # n3=C: CA, CC, CG, CT
    [1.590, 1.684, 1.604, 1.671],  # n3=G: GA, GC, GG, GT
    [1.634, 1.590, 1.654, 1.709],  # n3=T: TA, TC, TG, TT
], dtype=torch.float64)

# Sequence-dependent hydrogen bonding parameters
# Only AT and GC pairs matter; symmetric
HYDR_EPS_SEQ = torch.tensor([
    #       A      C      G      T
    [0.000, 0.000, 0.000, 0.893],  # A: only AT
    [0.000, 0.000, 1.243, 0.000],  # C: only CG
    [0.000, 1.243, 0.000, 0.000],  # G: only GC
    [0.893, 0.000, 0.000, 0.000],  # T: only TA
], dtype=torch.float64)

# ============================================================
# Base type encoding
# ============================================================
BASE_A = 0
BASE_C = 1
BASE_G = 2
BASE_T = 3

BASE_CHAR_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
BASE_INT_TO_CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

# ============================================================
# Cutoff distance
# ============================================================
def compute_rcut(grooving: bool = False, dh_rc: float = 0.0) -> float:
    """Compute the interaction cutoff distance.

    Args:
        grooving: whether to use major-minor groove backbone positions
        dh_rc: Debye-Hückel cutoff radius (0 = no DH term). When > 0,
               the cutoff is extended to cover backbone–backbone DH interactions.
    """
    if grooving:
        rcutback = 2 * math.sqrt(POS_MM_BACK1**2 + POS_MM_BACK2**2) + EXCL_RC1
    else:
        rcutback = 2 * abs(POS_BACK) + EXCL_RC1
    rcutbase = 2 * abs(POS_BASE) + HYDR_RCHIGH
    rcut = max(rcutback, rcutbase)

    if dh_rc > 0.0:
        if grooving:
            debyecut = 2 * math.sqrt(POS_MM_BACK1**2 + POS_MM_BACK2**2) + dh_rc
        else:
            debyecut = 2 * abs(POS_BACK) + dh_rc
        rcut = max(rcut, debyecut)

    return rcut
