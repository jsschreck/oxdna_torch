# oxdna_torch

A fully differentiable PyTorch reimplementation of the **oxDNA1 coarse-grained DNA model**. Forces and torques are computed via automatic differentiation, so gradients flow through the physics — enabling backpropagation through molecular dynamics trajectories. oxDNA2 support coming soon.

## Features

- **Complete oxDNA1 potential** — all 7 interaction terms: FENE backbone, bonded/non-bonded excluded volume, stacking, hydrogen bonding, cross-stacking, and coaxial stacking
- **Sequence-dependent parameters** — optional sequence-averaged or sequence-dependent stacking/H-bond strengths (Sulc et al. 2012)
- **Inertial Langevin integrator** — velocity-Verlet + Langevin thermostat matching the reference oxDNA C++ `MD_CPUBackend`, with correct torque derivation from quaternion gradients
- **Backprop through time** — `create_graph=True` keeps the autograd graph across integration steps; optional gradient checkpointing trades compute for memory on long trajectories
- **Learnable parameters** — any subset of the potential's physical constants can be promoted to `nn.Parameter` for gradient-based optimization
- **GPU-native** — all tensors and operations run on CUDA; move model and state with `.to('cuda')`
- **Standard file I/O** — reads and writes oxDNA `.top` / `.conf` files directly

## Installation

```bash
git clone https://github.com/jsschreck/oxdna_torch.git
cd oxdna_torch
pip install -e .
```

Requires Python ≥ 3.9 and PyTorch ≥ 2.0.

## Quick Start

```python
import torch
from oxdna_torch import load_system, OxDNAEnergy
from oxdna_torch.integrator import LangevinIntegrator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load topology and initial configuration
topology, state = load_system("examples/HAIRPIN/initial.top",
                              "examples/HAIRPIN/initial.conf",
                              device=device)

# Build the energy model
# Temperature in oxDNA reduced units: T_reduced = T_kelvin / 3000
model = OxDNAEnergy(
    topology,
    temperature=0.1113,   # ~334 K
    seq_dependent=True,
).to(device)

# Evaluate energy and per-term breakdown
energy = model(state)
components = model.energy_components(state)
for name, val in components.items():
    print(f"  {name:25s}: {val.item():+.4f}")

# Compute forces via autograd
forces = model.compute_forces(state)   # (N, 3)

# Run Langevin dynamics
integrator = LangevinIntegrator(
    model,
    dt=0.003,
    gamma=1.0,
    temperature=0.1113,
)

trajectory = integrator.rollout(state, n_steps=1000)
```

A runnable demo covering energy breakdown, force verification, dynamics, and backprop through time is at `examples/pytorch_demo.py`.

## Integrator

`LangevinIntegrator` implements inertial Langevin dynamics (velocity-Verlet + Langevin thermostat) with the same algorithm as the reference oxDNA `MD_CPUBackend`:

1. Half-kick velocities and angular momenta from forces/torques
2. Thermostat noise kick (first half)
3. Drift positions; update orientations from angular velocity via Rodrigues rotation
4. Recompute forces/torques at new positions
5. Second half-kick and thermostat noise kick

Torques are derived from the quaternion gradient `dE/dq` using the rotation generator `dq/dθᵢ = 0.5 · [0, eᵢ] ⊗ q`. Rotational diffusion uses `D_rot = 3 · D_trans` per the oxDNA convention.

Key parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dt` | Timestep (oxDNA reduced units) | `0.003` |
| `gamma` | Translational friction coefficient | `1.0` |
| `temperature` | Temperature (`T_kelvin / 3000`) | `0.1` |
| `mass` | Particle mass | `1.0` |
| `inertia` | Moment of inertia | `1.0` |

For backprop through time, pass `create_graph=True` to `step()` or `rollout()`. For memory-efficient BPTT on long trajectories, use `checkpoint_every=N` in `rollout()`.

## Performance and Rollouts

`oxdna_torch` is fully differentiable, but long sequential simulations remain significantly slower than standard C++ oxDNA due to computation graph overhead and Python interpreter latency. The following items outline some optimization priorities:

* **Implement Analytical Forces**: Replace `torch.autograd.grad` with analytical derivatives for Morse and Harmonic potentials during standard rollouts to eliminate the memory and compute overhead of building a computation graph at every integration step.

* **Separate Analytical and Autograd Paths**: Refactor the `LangevinIntegrator` to use a high-performance analytical force path for standard rollouts, reserving the `create_graph=True` autograd pathway strictly for training and backpropagation through time.

* **Neighbor List Caching**: Avoid recomputing `nonbonded_pairs` at every step. Reuse the neighbor list for 10–20 steps (with appropriate skin distance logic) to significantly accelerate rollouts in larger systems.

* **`torch.compile` Optimization**: Integrate `torch.compile(model, mode="reduce-overhead")` to reduce Python interpreter latency and enable kernel fusion, which is often the dominant bottleneck for systems with fewer than ~1,000 nucleotides.

* **Force Buffer Reuse**: Optimize the BAOAB integrator splitting to reuse forces computed at the end of one step as the initial forces for the next step, reducing redundant potential energy evaluations.

* **Reduce Tensor Churn**: Pre-allocate memory for interaction site offsets and increase the use of in-place operations (e.g., `pos.add_`) during drift steps to reduce pressure on the PyTorch caching allocator.


## Learnable Parameters

Any physical constant in the potential can be made a gradient-tracked `nn.Parameter` by passing its name to `OxDNAEnergy`:

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
├── model.py            # OxDNAEnergy: assembles all 7 interaction terms
├── integrator.py       # LangevinIntegrator, VelocityVerletIntegrator
├── state.py            # SystemState dataclass (positions, quaternions, box, ...)
├── topology.py         # Topology: connectivity, base types, bonded pairs
├── io.py               # load_system, read/write_configuration (.top / .conf)
├── quaternion.py       # Quaternion arithmetic and rotation utilities
├── pairs.py            # Neighbour finding (brute-force and cell list)
├── params.py           # ParameterStore for learnable vs. frozen constants
├── constants.py        # All oxDNA1 numerical constants
└── interactions/
    ├── fene.py
    ├── excluded_volume.py
    ├── stacking.py
    ├── hbond.py
    ├── cross_stacking.py
    └── coaxial_stacking.py
examples/
├── pytorch_demo.py     # Runnable end-to-end demo
└── HAIRPIN/            # 18-nt hairpin — topology, conf, and oxDNA input files
```

## Units

oxDNA uses its own reduced unit system:

| Quantity | Reduced unit |
|----------|-------------|
| Length | ~8.518 Å (≈ phosphate–phosphate distance) |
| Energy | kT at 3000 K (so `T_reduced = T_kelvin / 3000`) |
| Time | derived from length and energy units |

## References

- Ouldridge, Louis, Doye, *J. Chem. Phys.* **134**, 085101 (2011) — oxDNA1 model
- Šulc, Romano, Ouldridge, Rovigatti, Doye, Louis, *J. Chem. Phys.* **137**, 135101 (2012) — sequence-dependent parameters
