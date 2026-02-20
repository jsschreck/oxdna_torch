"""
All oxRNA model constants.

Transcribed directly from oxDNA/src/Interactions/rna_model.h (struct Model constructor)
and RNAInteraction.cpp (init()).
All values are in oxDNA reduced units.

Base type encoding: A=0, C=1, G=2, U=3  (U plays the role T plays in DNA)
Watson-Crick pairs: A-U (0+3=3), G-C (1+2=3)
Wobble pair: G-U (2+3=5)  -- allowed in sequence-dependent mode
"""

import math
import torch

# ============================================================
# Geometry: interaction site offsets from centre-of-mass
# Using body-frame axes a1 (principal), a2, a3 (base normal)
# ============================================================

# Backbone site:  BACK = a1*(-0.4) + a2*(0.0) + a3*(0.2)
RNA_POS_BACK_a1 = -0.4
RNA_POS_BACK_a2 =  0.0
RNA_POS_BACK_a3 =  0.2

# Simple scalar backbone distance (for cutoff calculation)
RNA_POS_BACK = -0.4

# Base / stacking scalar offsets (along a1)
RNA_POS_STACK = 0.34
RNA_POS_BASE  = 0.4
RNA_GAMMA     = 0.74   # RNA_POS_STACK - RNA_POS_BACK

# Asymmetric stacking sites:
#   STACK_3: site used when nucleotide stacks onto its 3' neighbour
RNA_POS_STACK_3_a1 =  0.4
RNA_POS_STACK_3_a2 =  0.1
#   STACK_5: site used when nucleotide stacks onto its 5' neighbour
RNA_POS_STACK_5_a1 =  0.124906078525
RNA_POS_STACK_5_a2 = -0.00866274917473

# Backbone-vector body-frame unit vectors (for thetaB1, thetaB2 in stacking)
#   p3 = vector pointing toward 3' backbone neighbour (body frame)
RNA_P3_x = -0.462510
RNA_P3_y = -0.528218
RNA_P3_z =  0.712089
#   p5 = vector pointing toward 5' backbone neighbour (body frame)
RNA_P5_x = -0.104402
RNA_P5_y = -0.841783
RNA_P5_z =  0.529624

# ============================================================
# FENE backbone potential
# ============================================================
RNA_FENE_EPS    = 2.0
RNA_FENE_R0     = 0.761070781051
RNA_FENE_DELTA  = 0.25
RNA_FENE_DELTA2 = 0.0625   # DELTA^2

# ============================================================
# Excluded volume (repulsive LJ) – 4 interaction types
# ============================================================
RNA_EXCL_EPS  = 2.0

# Back-Back (type 1)
RNA_EXCL_S1  = 0.70
RNA_EXCL_R1  = 0.675
RNA_EXCL_B1  = 892.016223343
RNA_EXCL_RC1 = 0.711879214356

# Base-Base (type 2)
RNA_EXCL_S2  = 0.33
RNA_EXCL_R2  = 0.32
RNA_EXCL_B2  = 4119.70450017
RNA_EXCL_RC2 = 0.335388426126

# Base-Back  p_base vs q_back  (type 3)
RNA_EXCL_S3  = 0.515
RNA_EXCL_R3  = 0.50
RNA_EXCL_B3  = 1707.30627298
RNA_EXCL_RC3 = 0.52329943261

# Back-Base  p_back vs q_base  (type 4)
RNA_EXCL_S4  = 0.515
RNA_EXCL_R4  = 0.50
RNA_EXCL_B4  = 1707.30627298
RNA_EXCL_RC4 = 0.52329943261

# ============================================================
# Hydrogen bonding (f1 radial, index 0)
# ============================================================
RNA_HYDR_F1   = 0
RNA_HYDR_EPS  = 0.870439
RNA_HYDR_A    = 8.0
RNA_HYDR_RC   = 0.75
RNA_HYDR_R0   = 0.4
RNA_HYDR_BLOW   = -126.243
RNA_HYDR_BHIGH  = -7.87708
RNA_HYDR_RLOW   = 0.34
RNA_HYDR_RHIGH  = 0.7
RNA_HYDR_RCLOW  = 0.276908
RNA_HYDR_RCHIGH = 0.783775

# ============================================================
# Stacking (f1 radial, index 1)
# ============================================================
RNA_STCK_F1         = 1
RNA_STCK_BASE_EPS   = 1.40206
RNA_STCK_FACT_EPS   = 2.77
RNA_STCK_A          = 6.0
RNA_STCK_RC         = 0.93
RNA_STCK_R0         = 0.43
RNA_STCK_BLOW       = -68.1857
RNA_STCK_BHIGH      = -3.12992
RNA_STCK_RLOW       = 0.35
RNA_STCK_RHIGH      = 0.78
RNA_STCK_RCLOW      = 0.26239
RNA_STCK_RCHIGH     = 0.986

# ============================================================
# Cross stacking (f2 radial, index 0)
# ============================================================
RNA_CRST_F2     = 0
RNA_CRST_R0     = 0.5
RNA_CRST_RC     = 0.6
RNA_CRST_K      = 59.9626
RNA_CRST_BLOW   = -0.888889
RNA_CRST_BHIGH  = -0.888889
RNA_CRST_RLOW   = 0.42
RNA_CRST_RHIGH  = 0.58
RNA_CRST_RCLOW  = 0.375
RNA_CRST_RCHIGH = 0.625

# ============================================================
# Coaxial stacking (f2 radial, index 1)
# ============================================================
RNA_CXST_F2     = 1
RNA_CXST_R0     = 0.5
RNA_CXST_RC     = 0.6
RNA_CXST_K      = 80.0
RNA_CXST_BLOW   = -0.888889
RNA_CXST_BHIGH  = -0.888889
RNA_CXST_RLOW   = 0.42
RNA_CXST_RHIGH  = 0.58
RNA_CXST_RCLOW  = 0.375
RNA_CXST_RCHIGH = 0.625

# ============================================================
# f4 angular function parameter sets  (16 sets, indexed 0-15)
#
# Index mapping (from RNAInteraction.cpp init()):
#   0  = RNA_STCK_F4_THETA4   (A=0 => disabled)
#   1  = RNA_STCK_F4_THETA5
#   2  = RNA_HYDR_F4_THETA1
#   3  = RNA_HYDR_F4_THETA2 (=THETA3)
#   4  = RNA_HYDR_F4_THETA4
#   5  = RNA_HYDR_F4_THETA7 (=THETA8)
#   6  = RNA_CRST_F4_THETA1
#   7  = RNA_CRST_F4_THETA2 (=THETA3)
#   8  = RNA_CRST_F4_THETA4
#   9  = RNA_CRST_F4_THETA7 (=THETA8)
#  10  = RNA_CXST_F4_THETA1
#  11  = RNA_CXST_F4_THETA4
#  12  = RNA_CXST_F4_THETA5 (=THETA6)
#  13  = RNA_STCK_F4_THETA6
#  14  = RNA_STCK_F4_THETAB1
#  15  = RNA_STCK_F4_THETAB2
# ============================================================

# Index aliases (mirroring the C++ #defines in rna_model.h)
RNA_STCK_F4_THETA4  = 0
RNA_STCK_F4_THETA5  = 1
RNA_HYDR_F4_THETA1  = 2
RNA_HYDR_F4_THETA2  = 3
RNA_HYDR_F4_THETA3  = 3   # same params as THETA2
RNA_HYDR_F4_THETA4  = 4
RNA_HYDR_F4_THETA7  = 5
RNA_HYDR_F4_THETA8  = 5   # same params as THETA7
RNA_CRST_F4_THETA1  = 6
RNA_CRST_F4_THETA2  = 7
RNA_CRST_F4_THETA3  = 7   # same params as THETA2
RNA_CRST_F4_THETA4  = 8
RNA_CRST_F4_THETA7  = 9
RNA_CRST_F4_THETA8  = 9   # same params as THETA7
RNA_CXST_F4_THETA1  = 10
RNA_CXST_F4_THETA4  = 11
RNA_CXST_F4_THETA5  = 12
RNA_CXST_F4_THETA6  = 12  # same params as THETA5
RNA_STCK_F4_THETA6  = 13
RNA_STCK_F4_THETAB1 = 14
RNA_STCK_F4_THETAB2 = 15

# f5 index aliases
RNA_STCK_F5_PHI1 = 0
RNA_STCK_F5_PHI2 = 1
RNA_CXST_F5_PHI3 = 2
RNA_CXST_F5_PHI4 = 3

# T0 values (scattered in rna_model.h constructor, assembled here)
RNA_CRST_THETA1_T0 = 0.505
RNA_CRST_THETA2_T0 = 1.266
RNA_CRST_THETA3_T0 = 1.266
RNA_CRST_THETA4_T0 = 0.0
RNA_CRST_THETA7_T0 = 0.309
RNA_CRST_THETA8_T0 = 0.309

RNA_CXST_THETA1_T0 = 2.592
RNA_CXST_THETA4_T0 = 0.151
RNA_CXST_THETA5_T0 = 0.685
RNA_CXST_THETA6_T0 = 0.685

# 16-entry tensors  (indices 0-15)
RNA_F4_THETA_A = torch.tensor([
    0.0,      #  0: STCK_T4 (disabled – A=0)
    0.9,      #  1: STCK_T5
    1.5,      #  2: HYDR_T1
    1.5,      #  3: HYDR_T2/T3
    0.46,     #  4: HYDR_T4
    4.0,      #  5: HYDR_T7/T8
    2.25,     #  6: CRST_T1
    1.70,     #  7: CRST_T2/T3
    1.50,     #  8: CRST_T4
    1.70,     #  9: CRST_T7/T8
    2.0,      # 10: CXST_T1
    1.3,      # 11: CXST_T4
    0.9,      # 12: CXST_T5/T6
    0.9,      # 13: STCK_T6
    1.3,      # 14: STCK_TB1
    1.3,      # 15: STCK_TB2
], dtype=torch.float64)

RNA_F4_THETA_B = torch.tensor([
    0.0,        #  0: STCK_T4 (disabled)
    3.89361,    #  1: STCK_T5
    4.16038,    #  2: HYDR_T1
    4.16038,    #  3: HYDR_T2
    0.133855,   #  4: HYDR_T4
    17.0526,    #  5: HYDR_T7
    7.00545,    #  6: CRST_T1
    6.2469,     #  7: CRST_T2
    2.59556,    #  8: CRST_T4
    6.2469,     #  9: CRST_T7
    10.9032,    # 10: CXST_T1
    6.4381,     # 11: CXST_T4
    3.89361,    # 12: CXST_T5
    3.89361,    # 13: STCK_T6
    6.4381,     # 14: STCK_TB1
    6.4381,     # 15: STCK_TB2
], dtype=torch.float64)

RNA_F4_THETA_T0 = torch.tensor([
    0.0,                   #  0: STCK_T4
    0.0,                   #  1: STCK_T5
    0.0,                   #  2: HYDR_T1
    0.0,                   #  3: HYDR_T2
    math.pi,               #  4: HYDR_T4
    math.pi * 0.5,         #  5: HYDR_T7
    RNA_CRST_THETA1_T0,    #  6: CRST_T1
    RNA_CRST_THETA2_T0,    #  7: CRST_T2
    RNA_CRST_THETA4_T0,    #  8: CRST_T4
    RNA_CRST_THETA7_T0,    #  9: CRST_T7
    RNA_CXST_THETA1_T0,    # 10: CXST_T1
    RNA_CXST_THETA4_T0,    # 11: CXST_T4
    RNA_CXST_THETA5_T0,    # 12: CXST_T5
    0.0,                   # 13: STCK_T6
    0.0,                   # 14: STCK_TB1
    0.0,                   # 15: STCK_TB2
], dtype=torch.float64)

RNA_F4_THETA_TS = torch.tensor([
    0.8,     #  0: STCK_T4
    0.95,    #  1: STCK_T5
    0.7,     #  2: HYDR_T1
    0.7,     #  3: HYDR_T2
    0.7,     #  4: HYDR_T4
    0.45,    #  5: HYDR_T7
    0.58,    #  6: CRST_T1
    0.68,    #  7: CRST_T2
    0.65,    #  8: CRST_T4
    0.68,    #  9: CRST_T7
    0.65,    # 10: CXST_T1
    0.8,     # 11: CXST_T4
    0.95,    # 12: CXST_T5
    0.95,    # 13: STCK_T6
    0.8,     # 14: STCK_TB1
    0.8,     # 15: STCK_TB2
], dtype=torch.float64)

RNA_F4_THETA_TC = torch.tensor([
    0.961538,   #  0: STCK_T4
    1.16959,    #  1: STCK_T5
    0.952381,   #  2: HYDR_T1
    0.952381,   #  3: HYDR_T2
    3.10559,    #  4: HYDR_T4
    0.555556,   #  5: HYDR_T7
    0.766284,   #  6: CRST_T1
    0.865052,   #  7: CRST_T2
    1.02564,    #  8: CRST_T4
    0.865052,   #  9: CRST_T7
    0.769231,   # 10: CXST_T1
    0.961538,   # 11: CXST_T4
    1.16959,    # 12: CXST_T5
    1.16959,    # 13: STCK_T6
    0.961538,   # 14: STCK_TB1
    0.961538,   # 15: STCK_TB2
], dtype=torch.float64)

# ============================================================
# f5 azimuthal function parameters (4 sets, indexed 0-3)
# ============================================================
RNA_F5_PHI_A = torch.tensor([
    2.0,   # 0: STCK_PHI1
    2.0,   # 1: STCK_PHI2
    2.0,   # 2: CXST_PHI3
    2.0,   # 3: CXST_PHI4
], dtype=torch.float64)

RNA_F5_PHI_B = torch.tensor([
    10.9032,   # 0: STCK_PHI1
    10.9032,   # 1: STCK_PHI2
    10.9032,   # 2: CXST_PHI3
    10.9032,   # 3: CXST_PHI4  (C++ accidentally uses PHI3_B for index 3 too)
], dtype=torch.float64)

RNA_F5_PHI_XC = torch.tensor([
    -0.769231,   # 0: STCK_PHI1
    -0.769231,   # 1: STCK_PHI2
    -0.769231,   # 2: CXST_PHI3
    -0.769231,   # 3: CXST_PHI4
], dtype=torch.float64)

RNA_F5_PHI_XS = torch.tensor([
    -0.65,   # 0: STCK_PHI1
    -0.65,   # 1: STCK_PHI2
    -0.65,   # 2: CXST_PHI3
    -0.65,   # 3: CXST_PHI4
], dtype=torch.float64)

# ============================================================
# f1 radial function tensors (index 0=HYDR, 1=STCK)
# ============================================================
RNA_F1_A      = torch.tensor([RNA_HYDR_A,      RNA_STCK_A],      dtype=torch.float64)
RNA_F1_RC     = torch.tensor([RNA_HYDR_RC,     RNA_STCK_RC],     dtype=torch.float64)
RNA_F1_R0     = torch.tensor([RNA_HYDR_R0,     RNA_STCK_R0],     dtype=torch.float64)
RNA_F1_BLOW   = torch.tensor([RNA_HYDR_BLOW,   RNA_STCK_BLOW],   dtype=torch.float64)
RNA_F1_BHIGH  = torch.tensor([RNA_HYDR_BHIGH,  RNA_STCK_BHIGH],  dtype=torch.float64)
RNA_F1_RLOW   = torch.tensor([RNA_HYDR_RLOW,   RNA_STCK_RLOW],   dtype=torch.float64)
RNA_F1_RHIGH  = torch.tensor([RNA_HYDR_RHIGH,  RNA_STCK_RHIGH],  dtype=torch.float64)
RNA_F1_RCLOW  = torch.tensor([RNA_HYDR_RCLOW,  RNA_STCK_RCLOW],  dtype=torch.float64)
RNA_F1_RCHIGH = torch.tensor([RNA_HYDR_RCHIGH, RNA_STCK_RCHIGH], dtype=torch.float64)

# ============================================================
# f2 radial function tensors (index 0=CRST, 1=CXST)
# ============================================================
RNA_F2_K      = torch.tensor([RNA_CRST_K,      RNA_CXST_K],      dtype=torch.float64)
RNA_F2_RC     = torch.tensor([RNA_CRST_RC,     RNA_CXST_RC],     dtype=torch.float64)
RNA_F2_R0     = torch.tensor([RNA_CRST_R0,     RNA_CXST_R0],     dtype=torch.float64)
RNA_F2_BLOW   = torch.tensor([RNA_CRST_BLOW,   RNA_CXST_BLOW],   dtype=torch.float64)
RNA_F2_BHIGH  = torch.tensor([RNA_CRST_BHIGH,  RNA_CXST_BHIGH],  dtype=torch.float64)
RNA_F2_RLOW   = torch.tensor([RNA_CRST_RLOW,   RNA_CXST_RLOW],   dtype=torch.float64)
RNA_F2_RHIGH  = torch.tensor([RNA_CRST_RHIGH,  RNA_CXST_RHIGH],  dtype=torch.float64)
RNA_F2_RCLOW  = torch.tensor([RNA_CRST_RCLOW,  RNA_CXST_RCLOW],  dtype=torch.float64)
RNA_F2_RCHIGH = torch.tensor([RNA_CRST_RCHIGH, RNA_CXST_RCHIGH], dtype=torch.float64)

# ============================================================
# Base type encoding  (A=0, C=1, G=2, U=3)
# U takes the role of T in DNA; Watson-Crick: A-U (sum=3), G-C (sum=3)
# ============================================================
RNA_BASE_A = 0
RNA_BASE_C = 1
RNA_BASE_G = 2
RNA_BASE_U = 3

RNA_BASE_CHAR_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
RNA_BASE_INT_TO_CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# ============================================================
# Sequence-dependent stacking parameters
# Source: oxRNA_sequence_dependent_parameters.txt
# eps(i,j) = raw_value * (1 + T * ST_T_DEP)
# where i = 3' base type, j = 5' base type
# ST_T_DEP is the temperature-scaling factor from the parameter file
# For average-sequence mode: eps = RNA_STCK_BASE_EPS + RNA_STCK_FACT_EPS * T
# ============================================================
RNA_STCK_FACT_EPS_SEQ = 0.18   # ST_T_DEP from oxRNA seq-dep file

# [n3][n5] stacking epsilons  (A=0, C=1, G=2, U=3)
RNA_STCK_EPS_SEQ = torch.tensor([
    # n5:     A       C       G       U
    [1.709,  1.671,  1.610,  1.553],   # n3=A
    [1.654,  1.604,  1.737,  1.610],   # n3=C
    [1.590,  1.684,  1.604,  1.671],   # n3=G
    [1.634,  1.590,  1.654,  1.709],   # n3=U
], dtype=torch.float64)

# Hydrogen-bonding epsilons  (Watson-Crick + G-U wobble)
# Only non-zero for valid pairs; G-U wobble uses same HB eps as default
RNA_HYDR_EPS_SEQ = torch.tensor([
    #        A      C      G      U
    [0.000, 0.000, 0.000, 0.893],   # A: A-U
    [0.000, 0.000, 1.243, 0.000],   # C: C-G
    [0.000, 1.243, 0.000, 0.870],   # G: G-C and G-U wobble
    [0.893, 0.000, 0.870, 0.000],   # U: U-A and U-G wobble
], dtype=torch.float64)

# Cross-stacking sequence-dependent K scaling (4x4, ratio vs RNA_CRST_K)
# Default = 1.0 (all pairs scale equally); loaded from seq-dep file if available
RNA_CROSS_SEQ_K = torch.ones(4, 4, dtype=torch.float64)

# ============================================================
# Cutoff distance calculation for RNA
# ============================================================
def compute_rna_rcut() -> float:
    """Compute the global interaction cutoff for oxRNA.

    Matches RNAInteraction::init():
        rcutback = 2*|back_vec| + RNA_EXCL_RC1
        rcutbaseA = 2*RNA_POS_BASE + RNA_HYDR_RCHIGH
        rcutbaseB = 2*RNA_POS_BASE + RNA_CRST_RCHIGH
        rcut = max(rcutback, rcutbaseA, rcutbaseB)
    """
    back_mag = math.sqrt(RNA_POS_BACK_a1**2 + RNA_POS_BACK_a2**2 + RNA_POS_BACK_a3**2)
    rcutback  = 2.0 * back_mag + RNA_EXCL_RC1
    rcutbaseA = 2.0 * abs(RNA_POS_BASE) + RNA_HYDR_RCHIGH
    rcutbaseB = 2.0 * abs(RNA_POS_BASE) + RNA_CRST_RCHIGH
    return max(rcutback, rcutbaseA, rcutbaseB)
