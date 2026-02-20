"""
oxdna_torch: Differentiable oxDNA/oxRNA potential in PyTorch.

A PyTorch reimplementation of the oxDNA and oxRNA coarse-grained nucleic-acid
models that supports backpropagation through time for:
  - Parameter learning
  - Hybrid neural network + physics models
  - Inverse sequence design

References:
  Ouldridge et al., J. Chem. Phys. 134, 085101 (2011)          [oxDNA]
  Sulc et al., J. Chem. Phys. 137, 135101 (2012)               [oxDNA seq-dep]
  Sulc et al., J. Chem. Phys. 140, 235102 (2014)               [oxRNA]
"""

from .state import SystemState
from .topology import Topology
from .io import load_system, read_topology, read_configuration, write_configuration
from .model import OxDNAEnergy
from .params import ParameterStore

# oxRNA
from .rna_topology import RNATopology
from .rna_model import OxRNAEnergy
from .rna_io import read_rna_topology, load_rna_system

__all__ = [
    # DNA
    'SystemState',
    'Topology',
    'OxDNAEnergy',
    'ParameterStore',
    'load_system',
    'read_topology',
    'read_configuration',
    'write_configuration',
    # RNA
    'RNATopology',
    'OxRNAEnergy',
    'read_rna_topology',
    'load_rna_system',
]
