"""
I/O helpers for oxRNA topology and configuration files.

oxRNA uses the same .conf format as oxDNA; the topology file may use either
letter-code bases (A/C/G/U) or the integer 'btype' encoding from native
oxDNA RNA topology files (e.g. 13, 14, -17, ...).
"""

from pathlib import Path
from typing import Union, Optional, Tuple

import torch

from .rna_topology import RNATopology
from .state import SystemState
from . import rna_constants as RC


def _decode_base_type(token: str) -> int:
    """Decode an oxDNA RNA base type token to A=0, C=1, G=2, U=3.

    The topology file may use:
      - Single letter codes: A, C, G, U (and T which maps to U for RNA)
      - Integer 'btype' codes (positive or negative), following the C++ rule:
            type = btype % 4               if btype >= 0
            type = 3 - ((3 - btype) % 4)  if btype < 0
        This matches RNAInteraction.cpp:
            p->type = (p->btype < 0) ? 3 - ((3-p->btype) % 4) : p->btype % 4;
    """
    # Try letter first (A, C, G, U or T for RNA)
    if token.upper() in RC.RNA_BASE_CHAR_TO_INT:
        return RC.RNA_BASE_CHAR_TO_INT[token.upper()]
    # Try numeric btype
    try:
        btype = int(token)
    except ValueError:
        raise ValueError(f"Unknown RNA base token: '{token}'. "
                         "Expected A/C/G/U or an integer btype.")
    if btype >= 0:
        return btype % 4
    else:
        return 3 - ((3 - btype) % 4)


def read_rna_topology(filepath: Union[str, Path]) -> RNATopology:
    """Read an oxDNA-format topology file for an RNA system.

    Handles both letter-code base types (A/C/G/U) and the integer 'btype'
    encoding used by native oxDNA RNA topology files (e.g. 13, -14, ...).

    The integer encoding follows the C++ convention in RNAInteraction.cpp:
        type = btype % 4           (btype >= 0)
        type = 3 - ((3-btype)%4)  (btype < 0)

    Args:
        filepath: path to .top file

    Returns:
        RNATopology object (base_types: A=0, C=1, G=2, U=3)
    """
    filepath = Path(filepath)

    with open(filepath, 'r') as f:
        lines = f.readlines()

    header = lines[0].strip().split()
    n_nucleotides = int(header[0])
    n_strands     = int(header[1])

    strand_ids       = torch.zeros(n_nucleotides, dtype=torch.long)
    base_types       = torch.zeros(n_nucleotides, dtype=torch.long)
    bonded_neighbors = torch.full((n_nucleotides, 2), -1, dtype=torch.long)

    for i in range(n_nucleotides):
        parts = lines[i + 1].strip().split()
        strand_id   = int(parts[0]) - 1   # 0-indexed
        base_token  = parts[1]
        n3 = int(parts[2])
        n5 = int(parts[3])

        strand_ids[i]          = strand_id
        base_types[i]          = _decode_base_type(base_token)
        bonded_neighbors[i, 0] = n3
        bonded_neighbors[i, 1] = n5

    return RNATopology(
        n_nucleotides=n_nucleotides,
        n_strands=n_strands,
        strand_ids=strand_ids,
        base_types=base_types,
        bonded_neighbors=bonded_neighbors,
    )


def load_rna_system(
    topology_file: Union[str, Path],
    conf_file: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[RNATopology, SystemState]:
    """Load an oxRNA system from topology and configuration files.

    The .conf format is identical to oxDNA — positions and orientations
    are stored as COM position + a1 + a3 vectors.

    Args:
        topology_file: path to .top file (letter or integer base codes)
        conf_file:     path to .conf / .dat file
        device:        torch device (default: cpu)

    Returns:
        (RNATopology, SystemState)
    """
    from .io import read_configuration   # reuse the conf reader unchanged

    topology = read_rna_topology(topology_file)
    # read_configuration works on any Topology-like object; pass None to skip
    # the nucleotide count assertion when topology is RNATopology
    state = read_configuration(conf_file, topology=None)

    if device is not None:
        topology = topology.to(device)
        state = state.to(device)

    return topology, state
