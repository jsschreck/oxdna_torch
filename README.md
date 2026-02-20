# oxdna_torch

> **Warning:** This codebase is experimental and under active development. There are likely bugs. Use with caution and validate results against the reference C++ oxDNA implementation before drawing scientific conclusions.

A fully differentiable PyTorch reimplementation of the **oxDNA** and **oxRNA** coarse-grained nucleic acid models. Forces and torques are obtained via automatic differentiation, so gradients flow through the physics — enabling backpropagation through molecular dynamics trajectories and gradient-based parameter optimisation.

> **BPTT scope:** The integrator is stateful (not functionally pure), so full backpropagation through long trajectories is memory-intensive. The intended use case is physics-informed neural networks — short or targeted rollouts supply the oxDNA potential as a differentiable energy term that a downstream model can learn from, rather than differentiating through entire MD runs.

## Models

| Model | Class | Interaction terms |
|-------|-------|-------------------|
| **oxDNA1** | `OxDNAEnergy` | FENE, excl. vol., stacking, H-bond, cross-stacking, coaxial stacking |
| **oxDNA2** | `OxDNAEnergy(use_oxdna2=True)` | All oxDNA1 terms + Debye–Hückel electrostatics + major/minor-groove geometry |
| **oxRNA1** | `OxRNAEnergy` | RNA-specific FENE, excl. vol., stacking (asymmetric STACK_3/STACK_5 sites), H-bond, cross-stacking, coaxial stacking. Optional oxRNA2 mismatch repulsion via `mismatch_repulsion=True`. Debye–Hückel electrostatics (full oxRNA2) not yet implemented. |

All three models support:
- **Sequence-averaged** or **sequence-dependent** stacking / H-bond parameters
- **Periodic boundary conditions** with minimum-image convention
- **GPU acceleration** — move the model and state with `.to('cuda')`
- **Autograd forces and torques** through every interaction term
- **Learnable parameters** — promote any physical constant to `nn.Parameter`

## Features

- **Complete potentials** — every interaction term from the published oxDNA1, oxDNA2, and oxRNA1 models, validated against the reference C++ implementation
- **Inertial Langevin integrator** — velocity-Verlet + Langevin thermostat matching the `MD_CPUBackend` in reference oxDNA C++, with correct torque derivation from quaternion gradients
- **Backprop through time** — `create_graph=True` keeps the autograd graph across integration steps; `save_every=N` + `create_graph=True` builds the graph only at saved steps (burn-in runs under `no_grad`) for decorrelated, memory-efficient sampling; optional gradient checkpointing trades compute for memory on long trajectories
- **Neighbour-list caching** — `set_nl_skin(skin)` reuses the pair list across steps until any particle drifts more than `skin/2`, reducing NL rebuilds during dynamics
- **TorchMD-Net NL backend** — drop-in replacement neighbour kernel (`set_nl_backend('torchmdnet')`) that dispatches to a Triton-fused GPU kernel when CUDA is available, or pure-PyTorch on CPU
- **`torch.compile` support** — call `model.compile()` to apply `torch.compile` to the energy model for faster rollouts
- **Standard file I/O** — reads and writes oxDNA `.top` / `.conf` files directly, including native RNA topology files with integer base-type codes

## Installation

```bash
git clone https://github.com/jsschreck/oxdna_torch.git
cd oxdna_torch
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

**Optional — TorchMD-Net neighbour kernel:**
```bash
pip install torchmd-net
```

## Quick Start — DNA

```python
import torch
from oxdna_torch import load_system, OxDNAEnergy
from oxdna_torch.integrator import LangevinIntegrator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load topology and initial configuration
topology, state = load_system("examples/HAIRPIN/initial.top",
                              "examples/HAIRPIN/initial.conf",
                              device=device)

# Build the energy model (oxDNA1, sequence-dependent)
# Temperature in oxDNA reduced units: T_reduced = T_kelvin / 3000
model = OxDNAEnergy(
    topology,
    temperature=0.1113,   # ≈ 334 K
    seq_dependent=True,
).to(device)

# Evaluate energy and per-term breakdown
energy = model(state)
components = model.energy_components(state)
for name, val in components.items():
    print(f"  {name:25s}: {val:+.4f}")   # values are floats

# Compute forces via autograd
forces = model.compute_forces(state)   # (N, 3)

# Run Langevin dynamics — returns a List[SystemState]
integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0, temperature=0.1113)
trajectory = integrator.rollout(state, n_steps=1000)
final_state = trajectory[-1]
```

## Quick Start — oxDNA2

```python
from oxdna_torch import load_system, OxDNAEnergy

topology, state = load_system("topology.top", "conf.dat")

model = OxDNAEnergy(
    topology,
    temperature=0.1,
    seq_dependent=True,
    use_oxdna2=True,            # enables Debye–Hückel + major/minor groove
    salt_concentration=0.5,     # molar salt for DH screening length
    dh_half_charged_ends=True,  # reduced charge at strand termini
)
```

## Quick Start — oxRNA

```python
import torch
from oxdna_torch import load_rna_system, OxRNAEnergy, SystemState
from oxdna_torch.integrator import LangevinIntegrator

# Load an RNA system (supports letter codes A/C/G/U and native integer btype codes)
topology, state = load_rna_system("examples/RNA_MD/prova.top",
                                  "examples/RNA_MD/prova.dat")

# T = 25 °C in oxDNA reduced units (T_kelvin / 3000)
T_red = (273.15 + 25) / 3000.0

model = OxRNAEnergy(
    topology,
    temperature=T_red,
    seq_dependent=True,              # use published RNA sequence-dependent parameters
    mismatch_repulsion=True,         # repulsive bump for non-Watson-Crick pairs (oxRNA2)
    mismatch_repulsion_strength=1.0, # strength of mismatch repulsion
)

# Per-term energy breakdown — values are floats
components = model.energy_components(state)
for name, val in components.items():
    print(f"  {name:20s}: {val:+.4f}")

# Forces via autograd
pos = state.positions.detach().requires_grad_(True)
s = SystemState(positions=pos, quaternions=state.quaternions.detach(),
                box=state.box)
model(s).backward()
forces = -pos.grad   # (N, 3)

# Langevin MD — returns a List[SystemState]
integrator = LangevinIntegrator(model, dt=0.003, gamma=1.0, temperature=T_red)
trajectory = integrator.rollout(state, n_steps=500)
final_state = trajectory[-1]
```

The `examples/RNA_MD/` directory contains a 132-nucleotide single-stranded RNA snapshot (`prova.top` / `prova.dat`) generated with oxRNA2 MD at 25 °C, suitable for validation.

## Neighbour-List Backends

Two backends are available for building the non-bonded pair list:

| Backend | Select | Notes |
|---------|--------|-------|
| **`'oxdna'`** (default) | `model.set_nl_backend('oxdna')` | Built-in brute-force / cell-list; no extra dependencies |
| **`'torchmdnet'`** | `model.set_nl_backend('torchmdnet')` | TorchMD-Net kernel; Triton-fused on CUDA, pure-PyTorch on CPU. Requires `pip install torchmd-net` |

```python
model.set_nl_backend('torchmdnet')   # opt in to faster GPU kernel
model.set_nl_backend('oxdna')        # revert to default
```

### Neighbour-List Caching

Enable NL caching to avoid rebuilding the pair list every step:

```python
model.set_nl_skin(0.05)   # reuse NL until any particle drifts > 0.025 reduced units
```

The pair list is rebuilt automatically when needed. Set `skin=0.0` (default) to rebuild every step.

## `torch.compile`

```python
model.compile()   # wraps the energy model with torch.compile
```

## Integrator

`LangevinIntegrator` implements inertial Langevin dynamics (velocity-Verlet + Langevin thermostat) matching the `MD_CPUBackend` algorithm in reference oxDNA:

1. Half-kick velocities and angular momenta from forces / torques
2. Thermostat noise kick (first half)
3. Drift positions; update orientations from angular velocity via Rodrigues rotation
4. Recompute forces / torques at new positions
5. Second half-kick and thermostat noise kick

Torques are derived from the quaternion gradient `dE/dq` using the rotation generator `dq/dθᵢ = 0.5 · [0, eᵢ] ⊗ q`. Rotational diffusion uses `D_rot = 3 · D_trans` per the oxDNA convention.

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dt` | Timestep (reduced units) | `0.003` |
| `gamma` | Translational friction | `1.0` |
| `temperature` | Temperature (`T_kelvin / 3000`) | `0.1` |
| `mass` | Particle mass | `1.0` |
| `inertia` | Moment of inertia | `1.0` |

### Backprop Through Time (BPTT)

Pass `create_graph=True` to `step()` or `rollout()` to keep the autograd graph across integration steps:

```python
trajectory = integrator.rollout(state, n_steps=100, create_graph=True)
loss = some_loss(trajectory)
loss.backward()   # gradients flow through every step
```

For memory-efficient BPTT on long trajectories, use `checkpoint_every=N` to recompute activations during the backward pass instead of storing them:

```python
trajectory = integrator.rollout(state, n_steps=1000,
                                create_graph=True, checkpoint_every=50)
```

### Decorrelated Sampling with `save_every`

MD configurations are highly correlated between adjacent steps. For gradient-based learning it is often better to collect a set of *decorrelated* snapshots rather than every consecutive step.

Set `save_every=N` to save one state every *N* steps. Only the saved states appear in the returned trajectory:

```python
# Save 10 decorrelated snapshots from 10 000 steps
trajectory = integrator.rollout(state, n_steps=10_000, save_every=1000)
# len(trajectory) == 11  (initial + 10 saved states)
```

**Memory optimisation when combined with `create_graph=True`:**
When `save_every > 1` and `create_graph=True`, the integrator automatically runs the intermediate *burn-in* steps as pure inference (`torch.no_grad()`). The autograd graph is only built for the single step leading to each saved state. The state is detached before each graph step so gradients do not flow through the no-grad burn-in steps. This avoids storing the full computational graph for every intermediate step:

```python
# Collect 10 decorrelated, gradient-connected snapshots.
# Only 10 graph steps are materialised (one per saved state),
# not 10 000, so memory scales with save_every rather than n_steps.
trajectory = integrator.rollout(
    state, n_steps=10_000,
    save_every=1000,
    create_graph=True,
)
loss = sum(some_loss(s) for s in trajectory[1:])
loss.backward()   # gradients flow through each saved step only
```

With `save_every=1` (default) the classic full-graph BPTT behaviour is preserved — every step carries the autograd graph.

### File Output (trajectory, lastconf, energy)

`rollout()` supports the same output files as the reference C++ oxDNA, controlled by keyword arguments:

| Argument | oxDNA equivalent | Description |
|----------|-----------------|-------------|
| `trajectory_file` | `trajectory_file` | Append a conf frame every `print_conf_interval` steps |
| `lastconf_file` | `lastconf_file` | Write final configuration at end of run |
| `energy_file` | `energy_file` | Append energy record every `print_energy_every` steps |
| `print_conf_interval` | `print_conf_interval` | Frame write interval (0 = off) |
| `print_energy_every` | `print_energy_every` | Energy write interval (0 = off) |
| `start_step` | `restart_step_counter` | Step counter offset for resumed runs |

```python
trajectory = integrator.rollout(
    state, n_steps=10_000_000,
    trajectory_file="trajectory.dat",   # conf written every 100 000 steps
    lastconf_file="last_conf.dat",       # final conf always written
    energy_file="energy.dat",            # energy written every 10 000 steps
    print_conf_interval=100_000,
    print_energy_every=10_000,
)
```

The trajectory and energy files can be read directly by oxDNA analysis tools. To resume a run, pass `start_step=N` so timestep headers are continuous.

```python
# Resume from step 10 000 000
topology, state = load_system("initial.top", "last_conf.dat")
trajectory = integrator.rollout(
    state, n_steps=10_000_000,
    trajectory_file="trajectory.dat",
    lastconf_file="last_conf.dat",
    energy_file="energy.dat",
    print_conf_interval=100_000,
    print_energy_every=10_000,
    start_step=10_000_000,
)
```

## Learnable Parameters

Any physical constant in the potential can be made a gradient-tracked `nn.Parameter`:

```python
model = OxDNAEnergy(
    topology,
    temperature=0.1113,
    learnable={'stacking_eps', 'hbond_eps', 'fene_eps'},
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

Available learnable parameter names are listed in `oxdna_torch/params.py` (`PARAM_REGISTRY`). The special names `stacking_eps` and `hbond_eps` make the per-pair / per-type epsilon matrices learnable.

## Repository Structure

```
oxdna_torch/
├── model.py             # OxDNAEnergy — oxDNA1 and oxDNA2
├── rna_model.py         # OxRNAEnergy — oxRNA1
├── integrator.py        # LangevinIntegrator, VelocityVerletIntegrator
├── state.py             # SystemState dataclass (positions, quaternions, box, …)
├── topology.py          # Topology: DNA connectivity, base types, bonded pairs
├── rna_topology.py      # RNATopology: RNA connectivity and sequence-dep. tables
├── io.py                # load_system, read/write_configuration (.top / .conf)
├── rna_io.py            # load_rna_system; handles letter and integer btype codes
├── quaternion.py        # Quaternion arithmetic and rotation utilities
├── pairs.py             # Neighbour finding ('oxdna' and 'torchmdnet' backends)
├── params.py            # ParameterStore for learnable vs. frozen constants
├── constants.py         # All oxDNA1/2 numerical constants
├── rna_constants.py     # All oxRNA1 numerical constants
└── interactions/
    ├── fene.py                  # DNA FENE backbone
    ├── excluded_volume.py       # DNA excluded volume
    ├── stacking.py              # DNA stacking
    ├── hbond.py                 # DNA hydrogen bonding
    ├── cross_stacking.py        # DNA cross-stacking
    ├── coaxial_stacking.py      # DNA coaxial stacking
    ├── electrostatics.py        # Debye–Hückel (oxDNA2)
    ├── rna_fene.py              # RNA FENE backbone
    ├── rna_excl_vol.py          # RNA excluded volume
    ├── rna_stacking.py          # RNA stacking (asymmetric STACK_3/STACK_5)
    ├── rna_hbond.py             # RNA hydrogen bonding
    ├── rna_cross_stacking.py    # RNA cross-stacking
    └── rna_coaxial_stacking.py  # RNA coaxial stacking
examples/
├── pytorch_demo.py      # End-to-end DNA demo (energy, forces, dynamics, BPTT)
├── HAIRPIN/             # 18-nt DNA hairpin — topology, conf, input files
└── RNA_MD/              # 132-nt single-stranded RNA — topology, conf, input files
                         #   (generated with oxRNA2 MD at 25 °C, seq-dependent)
```

## Units

oxDNA uses its own reduced unit system:

| Quantity | Reduced unit |
|----------|-------------|
| Length | ~8.518 Å (≈ phosphate–phosphate distance) |
| Energy | kT at 3000 K → `T_reduced = T_kelvin / 3000` |
| Time | derived from length and energy units |

## References

- Ouldridge, Louis, Doye, *J. Chem. Phys.* **134**, 085101 (2011) — oxDNA1
- Šulc, Romano, Ouldridge, Rovigatti, Doye, Louis, *J. Chem. Phys.* **137**, 135101 (2012) — oxDNA1 sequence-dependent parameters
- Snodin, Randisi, Mosayebi, Šulc, Schreck, Romano, Ouldridge, Tsukanov, Nir, Louis, Doye, *J. Chem. Phys.* **142**, 234901 (2015) — oxDNA2
- Šulc, Romano, Ouldridge, Doye, Louis, *J. Chem. Phys.* **140**, 235102 (2014) — oxRNA1
- Matek, Šulc, Randisi, Doye, Louis, *J. Chem. Phys.* **143**, 243122 (2015) — oxRNA1 sequence-dependent parameters
