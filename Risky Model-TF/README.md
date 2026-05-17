# Risky Model TF

TensorFlow implementation of the risky-debt corporate finance model with state
variables `(k, b, z)`, three Mailer/MMW-style neural-network objectives,
classical dynamic-programming benchmarks, and GMM/SMM/HMC estimation utilities.

## What is in this repository

The model includes:

- AR(1) productivity shocks in logs
- profits `pi(k,z) = z k^theta`
- convex capital adjustment costs
- risky debt priced in **price form** `q = 1 / (1 + r_tilde)`
- limited liability / default
- lender recovery with default haircut `alpha`
- solvency-contingent interest tax shield
- external equity issuance costs

The repository provides four main workflows:

1. **Objective 1**: lifetime reward maximization
2. **Objective 2**: Euler-residual minimization with value and continuation nets
3. **Objective 3**: Bellman/default residual minimization with constructed zero-profit pricing
4. **Estimation**: SMM, GMM, and optional HMC post-processing

It also provides a low-dimensional benchmark on the `(k,b,z)` grid using:

- value iteration (VI / VFI)
- modified policy iteration (MPI)

## Package layout

```text
project_root/
├── pyproject.toml
├── README.md
├── risky_debt/
│   ├── app/                # primary application layer
│   ├── estimation/         # namespaced estimation API
│   ├── training/           # namespaced training API
│   ├── layers.py           # custom TensorFlow layers
│   ├── networks.py         # subclassed tf.keras.Model modules
│   ├── trainer.py          # objective trainer implementations
│   ├── primitives.py
│   ├── objectives.py
│   ├── evaluation.py
│   ├── simulation.py
│   ├── grid_benchmark.py
│   ├── grid_compare.py
│   ├── plotting.py
│   └── io_utils.py
├── estimation/             # compatibility layer for legacy imports
├── Experiment/             # compatibility CLI wrapper
├── Test/                   # canonical pytest suite used by this repository
└── docs/notes/             # archived patch notes and refactor notes
```

## Installation

Editable install:

```bash
pip install -e .
```

## Testing

The repository uses the capitalized `Test/` package as the canonical pytest
suite. Running `pytest` from the project root will discover that directory
explicitly via the project configuration in `pyproject.toml`.

If TensorFlow or TensorFlow Probability are not installed in the current
environment, the pytest suite is skipped rather than failing during import
collection. In a normal project environment with the required dependencies
installed, the full suite under `Test/` runs as usual.

```bash
pytest
```

## Main entry point

The orchestrator is:

```bash
python -m Experiment.run_all
```

Example command matching the standard project workflow:

```bash
python -m Experiment.run_all \
  --out outputs/run1 \
  --epochs 40 \
  --steps_per_epoch 50 \
  --batch_size 512 \
  --hidden_units 64 \
  --do_estimation \
  --do_hmc
```

## What `run_all` does

A typical full run performs these steps:

1. train Objective 1, Objective 2, and Objective 3
2. save training histories, JSONL logs, and weights
3. generate ergodic-state and effectiveness plots
4. solve the risky-debt benchmark with VI and MPI unless disabled
5. compare NN outputs to the benchmark on multiple state sets
6. run SMM and GMM if `--do_estimation` is enabled
7. run HMC / Bayesian post-processing if `--do_hmc` is enabled

## Output structure

A standard run under `--out outputs/run1` creates subdirectories such as:

```text
outputs/run1/
├── checkpoints/
├── figures/
├── history/
├── logs/
├── benchmark/
└── estimation/
```

Typical artifacts include:

- per-objective training histories (`history/*.npz`)
- JSONL logs for training and estimation
- effectiveness plots (`figures/effectiveness_*`)
- ergodic-set plots
- benchmark convergence / policy / value / pricing plots
- NN-vs-benchmark comparison metrics (`*.json`)
- estimation reports for SMM, GMM, and optional HMC
- synthetic estimation dataset (`estimation/smm_synth_data.npz`)


## Pricing implementation note

The aligned risky-debt implementation does **not** learn the economic debt price as a free neural-network control.  The structural price is constructed from the lender zero-profit condition in `risky_debt/pricing.py`, using smooth safe pricing during training and exact nonsmoothed pricing for final economic evaluation.

A small `ConstructedPricingCompatibilityNet` module is still present, with the legacy alias `PricingNet`, because older trainer, checkpoint, and test APIs expect a `qnet` object.  That compatibility module is not the economic pricing rule; it only preserves stable public interfaces while the constructed zero-profit pricing operator supplies the model price.

## Estimation notes

The estimation pipeline is built around the baseline structural parameter vector

```text
Theta = (theta, psi0, alpha)
```

with the remaining primitives treated as fixed. The synthetic dataset stores the
main state, control, pricing, payout, and default objects used by the estimation
code, including `q`, `r_tilde`, `I`, `e`, `d`, `default`, `recovery`, and
continuation diagnostics.

## Tests

Run the test suite with:

```bash
pytest
```

Some tests are marked as slower smoke or integration coverage. The repository is
organized so the same command-line workflow can be used in Colab or a local
Python environment with TensorFlow installed.


## Why TensorFlow instead of NumPy?

TensorFlow is used here because the package needs more than array arithmetic:

- **automatic differentiation** for policy, value, continuation, and constructed-pricing residuals;
- **compiled `tf.function` training steps** for repeatable training/inference kernels;
- **stateful model objects and checkpointing** through `tf.keras.Model` and `tf.Variable`;
- **TensorFlow Probability compatibility** for the Bayesian/HMC workflow; and
- **Tensor-native control flow** such as `tf.TensorArray` inside the forward filter.

NumPy remains useful for lightweight orchestration and serialization, but the
core training and inference modules rely on TensorFlow because NumPy does not
provide trainable graph-based models or automatic gradients.

## Computational feasibility controls for risky-debt estimation

The risky-debt estimation pipeline follows the structural plan in the project summary: the baseline estimated vector remains `(theta, psi0, alpha)`, SMM/GMM use one shared synthetic dataset, fixed simulation draws/common random numbers, and candidate-specific inner model solves.  The code also uses implementation controls that reduce unnecessary computation without changing the economic model:

- **Baseline warm start:** estimation-time inner Objective 1 solves initialize from the already-trained baseline policy when available, rather than from a fresh random network for every candidate parameter vector.
- **Fixed common random numbers:** synthetic data, SMM simulations, and inner pricing draws are held fixed within an estimation run so changes in the objective mainly reflect parameter changes.
- **Candidate caching:** repeated parameter vectors reuse cached candidate moments and diagnostics.
- **Explicit inner-solve budget:** `--est_inner_epochs`, `--est_inner_steps`, `--est_T`, `--est_burn`, `--est_n_paths`, and `--est_n_starts` control the approximation budget used inside estimation.

These controls are computational devices. They are not surrogate SMM, not reduced-parameter estimation, and not placeholders for final estimation results.


### Common-budget objective defaults and evaluation random numbers

The default neural-solver budget is now common across Objective 1, Objective 2,
and Objective 3: `steps_per_epoch=20`, `batch_size=256`, `N_q=8`, and an
ergodic-buffer budget of burn-in `300`, horizon `1500`, paths `12`, and buffer
size `30000`.  Objective-specific controls such as `--obj2_steps_per_epoch` and
`--obj3_steps_per_epoch` default to `0`, which means they inherit the global
setting.  This keeps the three objectives comparable by training budget while
still allowing a user to override the residual objectives explicitly when needed.

Epoch-level test-reward evaluation uses the same evaluation seed formula for all
three objectives (`seed + 200 + epoch`), and Euler/FOC diagnostics use the same
shock-seed formula (`seed + 300 + epoch`).  This gives common-random-number
evaluation for the reported test-reward simulations across Objective 1/2/3 when
they use the same `TrainParams`.  Training random numbers remain
objective-specific because each objective solves a different optimization
problem.

### Objective 2/3 residual training implementation note

Objective 2 and Objective 3 compute FOC/KKT residuals for diagnostics and for
training the lower-bound multiplier.  In the default Colab-friendly training
mode, those FOC gradients are detached before they enter the outer network loss.
This avoids second-order automatic differentiation through the constructed
zero-profit pricing operator, which is prohibitively heavy in typical notebook
GPU environments.  The Bellman, default, and pricing-admissibility blocks still
train the policy/value/continuation networks with ordinary TensorFlow gradients,
and the reported Euler/KKT diagnostics still evaluate the residuals.

### Resume and long-run reliability

The command-line workflow checkpoints neural solver training after each epoch.
If a notebook or Colab runtime is interrupted, rerun the same command with the
same `--out` directory to skip completed objectives and continue the current
objective from the last saved epoch.  The benchmark also checkpoints outer grid
iterations, GMM/SMM store completed starts and in-progress local-optimizer state,
and Bayesian HMC stores segmented draws so long runs can be completed across
multiple sessions.

### Jacobian-rank and singular-value reporting

The estimation reports include the local-identification diagnostics requested in
the written summary.  For each GMM and SMM variant, the code writes
`table_jacobian_rank_diagnostics.csv/.tex`,
`table_jacobian_singular_values.csv/.tex`, and
`plot_jacobian_singular_values.png`.  These artifacts report the numerical rank,
full-column-rank indicator, singular values, minimum singular value, and
condition number of the moment Jacobian with respect to
`(theta, psi0, alpha)`.
