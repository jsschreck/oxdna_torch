"""
Neighbor/pair finding for oxDNA interactions.

Provides functions to find:
1. Bonded pairs (from topology)
2. Non-bonded pairs within cutoff distance

Two backends are available for step 2:
  - 'oxdna'      : built-in brute-force / cell-list (default, no extra deps)
  - 'torchmdnet' : torchmd-net kernel (brute on CPU, Triton-fused on GPU)
                   Requires: pip install torchmd-net
"""

import torch
from torch import Tensor
from typing import Optional, Tuple

from .topology import Topology
from . import constants as C


# ---------------------------------------------------------------------------
# Optional torchmd-net neighbor kernel
# ---------------------------------------------------------------------------

def _torchmdnet_available() -> bool:
    try:
        from torchmdnet.extensions.ops import get_neighbor_pairs_kernel  # noqa: F401
        return True
    except ImportError:
        return False


def find_nonbonded_pairs_torchmdnet(
    positions: Tensor,
    topology: Topology,
    box: Optional[Tensor] = None,
    cutoff: Optional[float] = None,
    strategy: str = 'brute',
) -> Tensor:
    """Find non-bonded pairs using the torchmd-net neighbor kernel.

    Uses ``get_neighbor_pairs_kernel`` from torchmd-net, which dispatches to
    a Triton-fused kernel on CUDA and a pure-PyTorch fallback on CPU.  The
    output is converted to the same (P, 2) upper-triangular format as the
    built-in backend so the rest of the code is unaffected.

    Args:
        positions: (N, 3) center-of-mass positions
        topology:  Topology object (bonded pairs are excluded from the result)
        box:       (3,) periodic box side lengths, or None
        cutoff:    interaction cutoff distance (default: from constants)
        strategy:  ``'brute'`` or ``'cell'`` — passed to the kernel.
                   ``'brute'`` works on both CPU and GPU; ``'cell'`` requires
                   CUDA and a periodic box.

    Returns:
        pairs: (P, 2) int64 tensor of non-bonded pair indices (i < j)

    Raises:
        ImportError: if torchmd-net is not installed
    """
    try:
        from torchmdnet.extensions.ops import get_neighbor_pairs_kernel
    except ImportError as exc:
        raise ImportError(
            "torchmd-net is required for the 'torchmdnet' neighbor backend. "
            "Install with:  pip install torchmd-net"
        ) from exc

    if cutoff is None:
        cutoff = C.compute_rcut()

    N = positions.shape[0]
    device = positions.device

    # torchmd-net wants float32; we work in float64 — cast, compute, cast back.
    pos_f32 = positions.detach().float()

    # batch tensor: all zeros (single molecule)
    batch = torch.zeros(N, dtype=torch.long, device=device)

    # box_vectors: torchmd-net expects a (3, 3) diagonal matrix (or None-like).
    # We support only orthogonal boxes (diagonal), matching our PBC convention.
    use_periodic = box is not None
    if use_periodic:
        box_mat = torch.diag(box.float()).unsqueeze(0)  # (1, 3, 3)
    else:
        box_mat = torch.zeros(1, 3, 3, dtype=torch.float32, device=device)

    # Conservative upper bound on number of pairs: N*(N-1)/2
    max_num_pairs = max(1, N * (N - 1) // 2)

    neighbors, _dist_vecs, _dists, num_pairs = get_neighbor_pairs_kernel(
        strategy=strategy,
        positions=pos_f32,
        batch=batch,
        box_vectors=box_mat,
        use_periodic=use_periodic,
        cutoff_lower=0.0,
        cutoff_upper=float(cutoff),
        max_num_pairs=max_num_pairs,
        loop=False,           # no self-pairs
        include_transpose=False,  # upper-triangular only (i < j after sort)
        num_cells=0,          # auto
    )
    # neighbors: (2, max_num_pairs), padded with -1
    # Trim to valid pairs
    n_valid = int(num_pairs.item())
    pairs_raw = neighbors[:, :n_valid].t()  # (P_raw, 2)

    # Enforce i < j (torchmd-net returns lower triangular: i > j)
    i = pairs_raw[:, 0]
    j = pairs_raw[:, 1]
    pairs = torch.stack([torch.minimum(i, j), torch.maximum(i, j)], dim=1)

    # Exclude bonded pairs
    pairs = _exclude_bonded(pairs, topology, N, device)
    return pairs


def compute_site_positions(
    positions: Tensor,
    quaternions: Tensor,
    grooving: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute backbone, stacking, and base site positions for all nucleotides.

    Each nucleotide has three interaction sites offset from the center of mass
    along the a1 axis (and a2 for grooving):
      - BACK: backbone site at POS_BACK * a1 (or POS_MM_BACK1*a1 + POS_MM_BACK2*a2 with grooving)
      - STACK: stacking site at POS_STACK * a1
      - BASE: base site at POS_BASE * a1

    Args:
        positions: (N, 3) center of mass positions
        quaternions: (N, 4) unit quaternions
        grooving: whether to use major-minor groove backbone positions

    Returns:
        back_sites: (N, 3) backbone site positions (absolute)
        stack_sites: (N, 3) stacking site positions (absolute)
        base_sites: (N, 3) base site positions (absolute)
    """
    from .quaternion import quat_to_rotmat

    R = quat_to_rotmat(quaternions)  # (N, 3, 3)

    a1 = R[:, :, 0]  # (N, 3) principal axis
    a2 = R[:, :, 1]  # (N, 3)

    if grooving:
        back_offset = C.POS_MM_BACK1 * a1 + C.POS_MM_BACK2 * a2
    else:
        back_offset = C.POS_BACK * a1

    stack_offset = C.POS_STACK * a1
    base_offset = C.POS_BASE * a1

    back_sites = positions + back_offset
    stack_sites = positions + stack_offset
    base_sites = positions + base_offset

    return back_sites, stack_sites, base_sites


def compute_site_offsets(
    quaternions: Tensor,
    grooving: bool = False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute site offsets (relative to COM) for all nucleotides.

    Same as compute_site_positions but returns offsets instead of absolute positions.
    Useful for torque calculations.

    Args:
        quaternions: (N, 4) unit quaternions
        grooving: whether to use major-minor groove backbone positions

    Returns:
        back_offsets: (N, 3) backbone offsets from COM
        stack_offsets: (N, 3) stacking offsets from COM
        base_offsets: (N, 3) base offsets from COM
    """
    from .quaternion import quat_to_rotmat

    R = quat_to_rotmat(quaternions)  # (N, 3, 3)

    a1 = R[:, :, 0]  # (N, 3)
    a2 = R[:, :, 1]  # (N, 3)

    if grooving:
        back_offset = C.POS_MM_BACK1 * a1 + C.POS_MM_BACK2 * a2
    else:
        back_offset = C.POS_BACK * a1

    stack_offset = C.POS_STACK * a1
    base_offset = C.POS_BASE * a1

    return back_offset, stack_offset, base_offset


def find_nonbonded_pairs(
    positions: Tensor,
    topology: Topology,
    box: Optional[Tensor] = None,
    cutoff: Optional[float] = None,
    method: str = 'auto',
    backend: str = 'oxdna',
) -> Tensor:
    """Find non-bonded pairs within the interaction cutoff.

    Args:
        positions: (N, 3) center-of-mass positions
        topology: Topology object (to exclude bonded pairs)
        box: (3,) periodic box dimensions, or None for non-periodic
        cutoff: interaction cutoff distance (default: computed from constants)
        method: 'auto', 'brute_force', or 'cell_list'  (ignored when
                backend='torchmdnet')
        backend: neighbor-list backend to use:
                 - ``'oxdna'``      : built-in implementation (default)
                 - ``'torchmdnet'`` : torchmd-net kernel (requires
                                     ``pip install torchmd-net``); uses Triton
                                     on CUDA, pure-PyTorch on CPU

    Returns:
        pairs: (P, 2) int tensor of non-bonded pair indices, where
               pairs[:, 0] < pairs[:, 1] (upper triangular)
    """
    if backend == 'torchmdnet':
        return find_nonbonded_pairs_torchmdnet(
            positions, topology, box, cutoff, strategy='brute'
        )

    if cutoff is None:
        cutoff = C.compute_rcut()

    N = positions.shape[0]

    use_cell_list = False
    if method == 'cell_list':
        assert box is not None, "Cell list requires periodic box"
        use_cell_list = True
    elif method == 'auto':
        use_cell_list = (N > 500 and box is not None)

    if use_cell_list:
        return _find_nonbonded_pairs_cell_list(positions, topology, box, cutoff)

    return _find_nonbonded_pairs_brute_force(positions, topology, box, cutoff)


def _find_nonbonded_pairs_brute_force(
    positions: Tensor,
    topology: Topology,
    box: Optional[Tensor],
    cutoff: float,
) -> Tensor:
    """Brute-force O(N^2) pair finding. Efficient for N < ~5000."""
    N = positions.shape[0]

    # Compute all pairwise displacement vectors
    r = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)

    # Apply minimum image convention for periodic boundaries
    if box is not None:
        r = r - box * torch.round(r / box)

    # Compute squared distances
    dist_sq = (r * r).sum(dim=-1)  # (N, N)

    # Create mask: within cutoff, upper triangular, and not self
    cutoff_sq = cutoff * cutoff
    mask = (dist_sq < cutoff_sq) & (dist_sq > 0)

    # Only upper triangular to avoid double counting
    triu_mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=positions.device), diagonal=1)
    mask = mask & triu_mask

    # Exclude bonded pairs
    bonded_p = topology.bonded_pairs[:, 0]
    bonded_q = topology.bonded_pairs[:, 1]
    bp_min = torch.minimum(bonded_p, bonded_q)
    bp_max = torch.maximum(bonded_p, bonded_q)
    mask[bp_min, bp_max] = False

    # Extract pair indices
    pairs = torch.nonzero(mask, as_tuple=False)  # (P, 2)
    return pairs


def _find_nonbonded_pairs_cell_list(
    positions: Tensor,
    topology: Topology,
    box: Tensor,
    cutoff: float,
) -> Tensor:
    """Cell list O(N) pair finding for large periodic systems."""
    N = positions.shape[0]
    device = positions.device
    cutoff_sq = cutoff * cutoff

    # 1. Compute cell grid
    n_cells = torch.floor(box / cutoff).to(torch.long)
    n_cells = torch.clamp(n_cells, min=3)
    cell_size = box / n_cells.to(box.dtype)
    nc = n_cells  # shorthand

    # If box is too small for cell list, fall back to brute force
    if (cell_size < cutoff).any() and N <= 5000:
        return _find_nonbonded_pairs_brute_force(positions, topology, box, cutoff)

    n_total_cells = nc[0].item() * nc[1].item() * nc[2].item()

    # 2. Assign particles to cells
    wrapped = positions - box * torch.floor(positions / box)
    cell_coords = torch.floor(wrapped / cell_size).to(torch.long) % nc  # (N, 3)
    cell_idx = (cell_coords[:, 0] * nc[1] * nc[2]
                + cell_coords[:, 1] * nc[2]
                + cell_coords[:, 2])  # (N,)

    # 3. Sort particles by cell index
    sorted_order = torch.argsort(cell_idx)
    sorted_cell_idx = cell_idx[sorted_order]

    # 4. Find cell boundaries
    unique_cells, counts = torch.unique_consecutive(sorted_cell_idx, return_counts=True)
    cell_starts = torch.zeros(n_total_cells, dtype=torch.long, device=device)
    cell_ends = torch.zeros(n_total_cells, dtype=torch.long, device=device)
    cumsum = torch.cumsum(counts, dim=0)
    cell_starts[unique_cells] = cumsum - counts
    cell_ends[unique_cells] = cumsum

    max_per_cell = counts.max().item() if counts.numel() > 0 else 0
    if max_per_cell == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    # 5. Build cell_members lookup: (n_total_cells, max_per_cell)
    cell_members = torch.full((n_total_cells, max_per_cell), -1,
                              dtype=torch.long, device=device)
    offsets_in_cell = torch.arange(N, device=device) - cell_starts[sorted_cell_idx]
    cell_members[sorted_cell_idx, offsets_in_cell] = sorted_order

    # 6. Build 27-neighbor stencil
    d = torch.tensor([-1, 0, 1], device=device)
    dx, dy, dz = torch.meshgrid(d, d, d, indexing='ij')
    neighbor_offsets_3d = torch.stack([dx.reshape(-1), dy.reshape(-1),
                                       dz.reshape(-1)], dim=-1)  # (27, 3)

    # 7. For each particle, find neighbor cell indices
    particle_cell_3d = torch.stack([
        cell_idx // (nc[1] * nc[2]),
        (cell_idx // nc[2]) % nc[1],
        cell_idx % nc[2],
    ], dim=-1)  # (N, 3)

    neighbor_cells_3d = (particle_cell_3d.unsqueeze(1)
                         + neighbor_offsets_3d.unsqueeze(0)) % nc  # (N, 27, 3)
    neighbor_flat = (neighbor_cells_3d[..., 0] * nc[1] * nc[2]
                     + neighbor_cells_3d[..., 1] * nc[2]
                     + neighbor_cells_3d[..., 2])  # (N, 27)

    # 8. Gather candidates — process in chunks if needed to avoid OOM
    total_candidates = N * 27 * max_per_cell
    MAX_ELEMENTS = 50_000_000

    if total_candidates <= MAX_ELEMENTS:
        pairs = _cell_list_gather_all(
            positions, box, cutoff_sq, N, cell_members,
            neighbor_flat, topology, device)
    else:
        # Chunked processing for very large systems
        chunk_size = max(1, MAX_ELEMENTS // (27 * max_per_cell))
        all_pairs = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_pairs = _cell_list_gather_chunk(
                positions, box, cutoff_sq, start, end, cell_members,
                neighbor_flat, topology, N, device)
            if chunk_pairs.shape[0] > 0:
                all_pairs.append(chunk_pairs)
        if all_pairs:
            pairs = torch.cat(all_pairs, dim=0)
        else:
            pairs = torch.zeros((0, 2), dtype=torch.long, device=device)

    return pairs


def _cell_list_gather_all(
    positions: Tensor, box: Tensor, cutoff_sq: float,
    N: int, cell_members: Tensor, neighbor_flat: Tensor,
    topology: Topology, device: torch.device,
) -> Tensor:
    """Gather and filter all pairs at once (fits in memory)."""
    candidates = cell_members[neighbor_flat]  # (N, 27, max_per_cell)
    candidates = candidates.reshape(N, -1)    # (N, 27 * max_per_cell)

    i_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(candidates)
    valid = (candidates >= 0) & (candidates > i_idx)

    valid_i = i_idx[valid]
    valid_j = candidates[valid]

    r = positions[valid_j] - positions[valid_i]
    r = r - box * torch.round(r / box)
    dist_sq = (r * r).sum(dim=-1)

    within_cutoff = dist_sq < cutoff_sq
    pair_i = valid_i[within_cutoff]
    pair_j = valid_j[within_cutoff]

    pairs = torch.stack([pair_i, pair_j], dim=-1)

    # Exclude bonded pairs
    return _exclude_bonded(pairs, topology, N, device)


def _cell_list_gather_chunk(
    positions: Tensor, box: Tensor, cutoff_sq: float,
    start: int, end: int, cell_members: Tensor,
    neighbor_flat: Tensor, topology: Topology,
    N: int, device: torch.device,
) -> Tensor:
    """Gather and filter pairs for a chunk of particles."""
    chunk_neighbor_flat = neighbor_flat[start:end]  # (chunk, 27)
    candidates = cell_members[chunk_neighbor_flat]   # (chunk, 27, max_per_cell)
    chunk_size = end - start
    candidates = candidates.reshape(chunk_size, -1)  # (chunk, 27 * max_per_cell)

    i_idx = torch.arange(start, end, device=device).unsqueeze(1).expand_as(candidates)
    valid = (candidates >= 0) & (candidates > i_idx)

    valid_i = i_idx[valid]
    valid_j = candidates[valid]

    r = positions[valid_j] - positions[valid_i]
    r = r - box * torch.round(r / box)
    dist_sq = (r * r).sum(dim=-1)

    within_cutoff = dist_sq < cutoff_sq
    pair_i = valid_i[within_cutoff]
    pair_j = valid_j[within_cutoff]

    if pair_i.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=device)

    pairs = torch.stack([pair_i, pair_j], dim=-1)
    return _exclude_bonded(pairs, topology, N, device)


def _exclude_bonded(
    pairs: Tensor, topology: Topology, N: int, device: torch.device,
) -> Tensor:
    """Exclude bonded pairs from a pair list using hash-based lookup."""
    if pairs.shape[0] == 0:
        return pairs

    bp = topology.bonded_pairs
    bp_min = torch.minimum(bp[:, 0], bp[:, 1])
    bp_max = torch.maximum(bp[:, 0], bp[:, 1])
    bonded_hash = bp_min * N + bp_max

    pair_hash = pairs[:, 0] * N + pairs[:, 1]
    is_bonded = torch.isin(pair_hash, bonded_hash)
    return pairs[~is_bonded]


def min_image_displacement(
    r: Tensor,
    box: Optional[Tensor],
) -> Tensor:
    """Apply minimum image convention to displacement vectors.

    Args:
        r: (..., 3) displacement vectors
        box: (3,) box dimensions, or None

    Returns:
        (..., 3) minimum-image displacements
    """
    if box is None:
        return r
    return r - box * torch.round(r / box)
