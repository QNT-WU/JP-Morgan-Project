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
3. **Objective 3**: Bellman/default residual minimization with a pricing net
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

- **automatic differentiation** for policy, value, continuation, and pricing networks;
- **compiled `tf.function` training steps** for repeatable training/inference kernels;
- **stateful model objects and checkpointing** through `tf.keras.Model` and `tf.Variable`;
- **TensorFlow Probability compatibility** for the Bayesian/HMC workflow; and
- **Tensor-native control flow** such as `tf.TensorArray` inside the forward filter.

NumPy remains useful for lightweight orchestration and serialization, but the
core training and inference modules rely on TensorFlow because NumPy does not
provide trainable graph-based models or automatic gradients.
