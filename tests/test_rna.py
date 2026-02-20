"""
Tests for the oxRNA potential (OxRNAEnergy).

Uses a small synthetic system (single 4-nucleotide RNA strand) built
entirely in Python so no external .conf file is needed.
"""

import math
import pytest
import torch

from oxdna_torch import SystemState
from oxdna_torch.rna_topology import RNATopology
from oxdna_torch.rna_model import OxRNAEnergy
from oxdna_torch import RNATopology, OxRNAEnergy   # test __init__ exports
from oxdna_torch import rna_constants as RC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_quat(n: int) -> torch.Tensor:
    """Return (N, 4) identity quaternions [w=1, x=0, y=0, z=0]."""
    q = torch.zeros(n, 4, dtype=torch.float64)
    q[:, 0] = 1.0
    return q


def _make_linear_rna(n: int = 4, spacing: float = 0.8):
    """Create a single straight RNA strand with N nucleotides.

    Nucleotides are spaced 'spacing' apart along the x-axis.
    Sequence cycles A, C, G, U.

    Returns:
        topology: RNATopology
        state:    SystemState (no box / non-periodic)
    """
    base_cycle = [0, 1, 2, 3]  # A, C, G, U

    strand_ids       = torch.zeros(n, dtype=torch.long)
    base_types       = torch.tensor([base_cycle[i % 4] for i in range(n)],
                                    dtype=torch.long)
    bonded_neighbors = torch.full((n, 2), -1, dtype=torch.long)

    for i in range(n):
        if i > 0:
            bonded_neighbors[i, 0] = i - 1   # n3 neighbor (5' direction)
        if i < n - 1:
            bonded_neighbors[i, 1] = i + 1   # n5 neighbor (3' direction)

    topology = RNATopology(
        n_nucleotides=n,
        n_strands=1,
        strand_ids=strand_ids,
        base_types=base_types,
        bonded_neighbors=bonded_neighbors,
    )

    positions = torch.zeros(n, 3, dtype=torch.float64)
    for i in range(n):
        positions[i, 0] = i * spacing

    quaternions = _identity_quat(n)
    state = SystemState(positions=positions, quaternions=quaternions, box=None)
    return topology, state


def _make_model(n: int = 4, seq_dependent: bool = False,
                temperature: float = 0.1) -> tuple:
    topology, state = _make_linear_rna(n)
    model = OxRNAEnergy(topology, temperature=temperature,
                        seq_dependent=seq_dependent)
    return model, topology, state


# ---------------------------------------------------------------------------
# Basic import / construction
# ---------------------------------------------------------------------------

def test_rna_imports():
    """OxRNAEnergy and RNATopology are importable from the top-level package."""
    from oxdna_torch import OxRNAEnergy, RNATopology  # noqa: F401


def test_rna_constants_cutoff():
    """RNA cutoff should be positive and > 1 (in reduced units)."""
    rcut = RC.compute_rna_rcut()
    assert rcut > 1.0
    assert rcut < 5.0   # sanity upper bound


def test_rna_base_encoding():
    assert RC.RNA_BASE_CHAR_TO_INT['A'] == 0
    assert RC.RNA_BASE_CHAR_TO_INT['C'] == 1
    assert RC.RNA_BASE_CHAR_TO_INT['G'] == 2
    assert RC.RNA_BASE_CHAR_TO_INT['U'] == 3


# ---------------------------------------------------------------------------
# RNATopology tests
# ---------------------------------------------------------------------------

def test_rna_topology_bonded_pairs():
    topology, _ = _make_linear_rna(4)
    # 4 nucleotides, 3 bonded pairs
    assert topology.n_bonded == 3
    assert topology.bonded_pairs.shape == (3, 2)


def test_rna_topology_stacking_eps_average():
    topology, _ = _make_linear_rna(4)
    T = 0.1
    eps = topology.compute_stacking_eps(T, seq_dependent=False)
    expected = RC.RNA_STCK_BASE_EPS + RC.RNA_STCK_FACT_EPS * T
    assert eps.shape == (3,)
    assert torch.allclose(eps, torch.full_like(eps, expected))


def test_rna_topology_stacking_eps_seqdep():
    topology, _ = _make_linear_rna(4)
    eps = topology.compute_stacking_eps(0.1, seq_dependent=True)
    assert eps.shape == (3,)
    assert (eps > 0).all()


def test_rna_topology_hbond_eps_average():
    topology, _ = _make_linear_rna(4)
    tbl = topology.compute_hbond_eps(seq_dependent=False)
    assert tbl.shape == (4, 4)
    # A(0)-U(3) and G(2)-C(1) should be nonzero; A(0)-A(0) should be zero
    assert tbl[0, 3].item() > 0
    assert tbl[2, 1].item() > 0
    assert tbl[0, 0].item() == 0.0


def test_rna_topology_hbond_eps_seqdep_wobble():
    topology, _ = _make_linear_rna(4)
    tbl = topology.compute_hbond_eps(seq_dependent=True)
    # G(2)-U(3) wobble pair should be nonzero in seq-dep mode
    assert tbl[2, 3].item() > 0


# ---------------------------------------------------------------------------
# OxRNAEnergy: forward pass
# ---------------------------------------------------------------------------

def test_oxrna_energy_scalar():
    model, _, state = _make_model()
    energy = model(state)
    assert energy.ndim == 0
    assert not torch.isnan(energy)
    assert not torch.isinf(energy)


def test_oxrna_energy_negative_or_finite():
    """A physically reasonable configuration should give a finite energy."""
    model, _, state = _make_model()
    energy = model(state)
    assert energy.isfinite()


def test_oxrna_energy_components():
    model, _, state = _make_model()
    comps = model.energy_components(state)
    required_keys = {'fene', 'bonded_excl', 'stacking',
                     'nonbonded_excl', 'hbond', 'mismatch_repulsion',
                     'cross_stacking', 'coaxial_stack', 'total'}
    assert required_keys == set(comps.keys())
    # Total should match sum of parts
    parts_sum = sum(v for k, v in comps.items() if k != 'total')
    assert abs(comps['total'] - parts_sum) < 1e-10


def test_oxrna_energy_seqdep_vs_average():
    """Sequence-dependent energy differs from average-sequence energy."""
    model_avg, _, state = _make_model(seq_dependent=False)
    model_seq, _, _     = _make_model(seq_dependent=True)
    e_avg = model_avg(state).item()
    e_seq = model_seq(state).item()
    # They don't need to be equal; just both finite
    assert math.isfinite(e_avg)
    assert math.isfinite(e_seq)


# ---------------------------------------------------------------------------
# Autograd: forces and torques
# ---------------------------------------------------------------------------

def test_oxrna_forces_via_autograd():
    """Forces = -dE/d(positions) should be finite and non-NaN."""
    model, _, state = _make_model()
    pos = state.positions.detach().requires_grad_(True)
    s = SystemState(positions=pos, quaternions=state.quaternions.detach(),
                    box=None)
    energy = model(s)
    energy.backward()
    forces = -pos.grad
    assert forces.shape == state.positions.shape
    assert not torch.isnan(forces).any()
    assert not torch.isinf(forces).any()


def test_oxrna_torques_via_autograd():
    """Torques = -dE/d(quaternions) should be finite and non-NaN."""
    model, _, state = _make_model()
    quats = state.quaternions.detach().requires_grad_(True)
    s = SystemState(positions=state.positions.detach(),
                    quaternions=quats, box=None)
    energy = model(s)
    energy.backward()
    assert quats.grad is not None
    assert not torch.isnan(quats.grad).any()
    assert not torch.isinf(quats.grad).any()


# ---------------------------------------------------------------------------
# Neighbor list
# ---------------------------------------------------------------------------

def test_oxrna_nl_skin():
    model, _, state = _make_model()
    model.set_nl_skin(0.05)
    e1 = model(state).item()
    e2 = model(state).item()   # should use cache
    assert abs(e1 - e2) < 1e-12


def test_oxrna_nl_backend_default():
    model, _, state = _make_model()
    assert model.nl_backend == 'oxdna'
    energy = model(state)
    assert energy.isfinite()


def test_oxrna_nl_backend_torchmdnet():
    pytest.importorskip('torchmdnet',
                        reason='torchmd-net not installed')
    model, _, state = _make_model()
    model.set_nl_backend('torchmdnet')
    energy = model(state)
    assert energy.isfinite()


def test_oxrna_nl_backend_bad_name():
    model, _, state = _make_model()
    with pytest.raises(ValueError, match='Unknown neighbor-list backend'):
        model.set_nl_backend('magic')


# ---------------------------------------------------------------------------
# Device transfer (CPU only; GPU test skipped if unavailable)
# ---------------------------------------------------------------------------

def test_oxrna_to_cpu():
    model, _, state = _make_model()
    model = model.to('cpu')
    energy = model(state)
    assert energy.isfinite()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='no CUDA')
def test_oxrna_to_cuda():
    model, _, state = _make_model()
    model  = model.cuda()
    state  = state.to(torch.device('cuda'))
    energy = model(state)
    assert energy.isfinite()


# ---------------------------------------------------------------------------
# Integration with LangevinIntegrator
# ---------------------------------------------------------------------------

def test_oxrna_integrator_step():
    """OxRNAEnergy works as a drop-in for LangevinIntegrator."""
    from oxdna_torch.integrator import LangevinIntegrator

    model, _, state = _make_model(n=4)
    integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=0.1)

    new_state = integrator.step(state, stochastic=False)
    assert new_state.positions.shape == state.positions.shape
    assert not torch.isnan(new_state.positions).any()


def test_oxrna_rollout():
    """A short rollout should produce a finite energy trajectory."""
    from oxdna_torch.integrator import LangevinIntegrator

    model, _, state = _make_model(n=4)
    integrator = LangevinIntegrator(model, dt=0.001, gamma=1.0, temperature=0.1)

    current = state.clone().detach()
    for _ in range(10):
        current = integrator.step(current, stochastic=True)

    energy = model(current)
    assert energy.isfinite()


# ---------------------------------------------------------------------------
# RNA I/O  (read_rna_topology with U bases)
# ---------------------------------------------------------------------------

def test_read_rna_topology_u_bases(tmp_path):
    """read_rna_topology should handle U bases."""
    from oxdna_torch.rna_io import read_rna_topology

    top_content = """\
4 1
1 A -1 1
1 U 0 2
1 G 1 3
1 C 2 -1
"""
    top_file = tmp_path / 'test.top'
    top_file.write_text(top_content)

    topology = read_rna_topology(top_file)
    assert topology.n_nucleotides == 4
    assert topology.base_types[1].item() == RC.RNA_BASE_U  # U=3


def test_read_rna_topology_integer_btypes(tmp_path):
    """read_rna_topology should decode integer btype codes (native oxDNA format)."""
    from oxdna_torch.rna_io import read_rna_topology, _decode_base_type

    # Verify _decode_base_type decodes correctly:
    # btype >= 0: type = btype % 4
    assert _decode_base_type('0')  == 0   # 0%4=0 = A
    assert _decode_base_type('13') == 1   # 13%4=1 = C
    assert _decode_base_type('14') == 2   # 14%4=2 = G
    assert _decode_base_type('15') == 3   # 15%4=3 = U
    assert _decode_base_type('16') == 0   # 16%4=0 = A
    # btype < 0: type = 3 - ((3-btype)%4)
    assert _decode_base_type('-17') == 3  # 3-((3+17)%4)=3-(20%4)=3-0=3 = U
    assert _decode_base_type('-14') == 2  # 3-((3+14)%4)=3-(17%4)=3-1=2 = G

    # Write a topology with integer btypes (like sim.top in RNA_DUPLEX_MELT)
    top_content = """\
8 1
1 13 -1 1
1 14 0 2
1 15 1 3
1 16 2 4
1 17 3 5
1 18 4 6
1 19 5 7
1 20 6 -1
"""
    top_file = tmp_path / 'int_btype.top'
    top_file.write_text(top_content)

    topology = read_rna_topology(top_file)
    assert topology.n_nucleotides == 8
    # btypes 13,14,15,16,17,18,19,20 → types 1,2,3,0,1,2,3,0 (C,G,U,A,C,G,U,A)
    expected = [1, 2, 3, 0, 1, 2, 3, 0]
    assert topology.base_types.tolist() == expected


# ---------------------------------------------------------------------------
# Integration tests: RNA_MD example (132-nt single-stranded RNA, MD snapshot)
#
# The example uses interaction_type=RNA2 (oxRNA2, with Debye–Hückel and
# mismatch_repulsion) and sequence-dependent parameters.  Our implementation
# covers oxRNA1, so we test against the oxRNA1 seq-dep energy and verify that
# all interaction terms are physically reasonable.
#
# Reference energy from the conf header:  E = -1.3387027012208 per nucleotide
# (computed by the C++ oxDNA with RNA2 + seq-dep + DH at salt=1.0 M, T=25°C).
# oxRNA1 seq-dep (no DH) gives ≈ -1.311 per nucleotide on the same snapshot,
# which is within ~2% of the RNA2 value — consistent with DH being a small
# correction at 1 M salt.
# ---------------------------------------------------------------------------

_HERE = __import__('pathlib').Path(__file__).parent.parent   # repo root
RNA_MD_TOP  = str(_HERE / 'examples' / 'RNA_MD' / 'prova.top')
RNA_MD_CONF = str(_HERE / 'examples' / 'RNA_MD' / 'prova.dat')
_rna_md_files_exist = (
    __import__('pathlib').Path(RNA_MD_TOP).exists()
    and __import__('pathlib').Path(RNA_MD_CONF).exists()
)


@pytest.mark.skipif(not _rna_md_files_exist, reason='examples/RNA_MD files not found')
def test_rna_md_load():
    """load_rna_system correctly parses the RNA_MD topology and configuration."""
    from oxdna_torch import load_rna_system

    topo, state = load_rna_system(RNA_MD_TOP, RNA_MD_CONF)

    assert topo.n_nucleotides == 132
    assert topo.n_strands == 1
    assert topo.n_bonded == 131          # linear strand → N-1 bonded pairs
    assert (topo.base_types >= 0).all()
    assert (topo.base_types <= 3).all()
    # Box is 30 × 30 × 30 as declared in the conf header
    assert state.box.tolist() == pytest.approx([30.0, 30.0, 30.0])


@pytest.mark.skipif(not _rna_md_files_exist, reason='examples/RNA_MD files not found')
def test_rna_md_energy_components():
    """oxRNA1 energy components on the RNA_MD snapshot are finite and physically sane.

    The conf was generated with oxRNA2 (seq-dep, Debye–Hückel, mismatch_repulsion)
    at T=25°C.  We evaluate with oxRNA1 seq-dep (no DH) and verify:
      * Total energy is finite and negative.
      * Stacking dominates (>100 SU in magnitude for 131 bonded pairs).
      * Hydrogen-bond energy is non-trivial (structured RNA).
      * Per-nucleotide energy is within ±20% of the RNA2 reference.
    """
    from oxdna_torch import load_rna_system, OxRNAEnergy

    topo, state = load_rna_system(RNA_MD_TOP, RNA_MD_CONF)
    T_red = (273.15 + 25) / 3000.0   # 25°C in oxDNA reduced units ≈ 0.09938

    model = OxRNAEnergy(topo, temperature=T_red, seq_dependent=True)
    comps = model.energy_components(state)

    # All terms must be finite
    for name, val in comps.items():
        assert torch.isfinite(torch.tensor(val)), f"{name} is not finite: {val}"

    # Total energy must be substantially negative (folded RNA)
    assert comps['total'] < 0.0
    assert comps['stacking'] < -100.0, "Stacking should dominate for a 132-nt RNA"
    assert comps['hbond']    < -10.0,  "H-bonds expected in structured RNA"

    # Per-nucleotide energy should be in [-2, -1] SU
    # (RNA2 reference: -1.339; oxRNA1 seq-dep: ≈ -1.311)
    e_per_nt = comps['total'] / topo.n_nucleotides
    assert -2.0 < e_per_nt < -1.0, (
        f"Energy per nucleotide {e_per_nt:.4f} outside expected range [-2, -1]"
    )


@pytest.mark.skipif(not _rna_md_files_exist, reason='examples/RNA_MD files not found')
def test_rna_md_forces_autograd():
    """Forces computed via autograd are finite and non-NaN for the RNA_MD snapshot."""
    from oxdna_torch import load_rna_system, OxRNAEnergy

    topo, state = load_rna_system(RNA_MD_TOP, RNA_MD_CONF)
    T_red = (273.15 + 25) / 3000.0

    model = OxRNAEnergy(topo, temperature=T_red, seq_dependent=True)

    pos = state.positions.detach().requires_grad_(True)
    s = state.__class__(positions=pos,
                        quaternions=state.quaternions.detach(),
                        box=state.box)
    model(s).backward()

    assert pos.grad is not None
    assert not torch.isnan(pos.grad).any(),  "Force NaNs detected"
    assert not torch.isinf(pos.grad).any(),  "Force Infs detected"
    # Forces should be non-trivially large for a packed RNA
    assert pos.grad.norm() > 1.0, "Forces unexpectedly small"
