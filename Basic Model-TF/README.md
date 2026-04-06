# Basic Model (k, z) — Mailer/MMW Objectives in TensorFlow

This repository packages the basic stochastic growth model with state variables
`(k_t, z_t)` as a client-facing Python package built on TensorFlow.

## What is included

The package implements the three neural-network objectives used in the Mailer/MMW framework:

1. **Objective 1** — lifetime reward maximization
2. **Objective 2** — Euler residual minimization
3. **Objective 3** — Bellman plus Euler residual minimization

It also includes:

- class-based TensorFlow training workflows
- policy-induced ergodic-state simulation
- grid-based dynamic-programming benchmarks (VFI and Howard policy iteration)
- benchmark-versus-NN comparison tools
- GMM and SMM estimation modules
- JSONL logging and TensorFlow checkpointing

## Package layout

```text
basic_mailer/
  config.py              # configuration dataclasses
  primitives.py          # economic primitives
  networks.py            # backward-compatible model exports
  simulation.py          # backward-compatible simulation exports
  trainer.py             # backward-compatible training exports
  nn/                    # TensorFlow layers and models
  sim/                   # ergodic simulation classes
  training/              # class-based objective trainers
  benchmark/             # OOP wrappers around benchmark solvers/comparison
  estimation/            # GMM/SMM pipeline
  apps/                  # package entrypoints
Experiment/
  run_all.py             # legacy entrypoint kept for compatibility
```

## Installation

```bash
pip install -e .
```

## Running the pipeline

Legacy entrypoint:

```bash
python -m Experiment.run_all --out outputs/run1
```

Package entrypoint:

```bash
basic-mailer-run --out outputs/run1
```

## Notes on TensorFlow design

The package uses TensorFlow-native abstractions throughout the core training path:

- custom `tf.keras.layers.Layer` objects for reusable model components
- subclassed `tf.keras.Model` objects for policy and value networks
- `@tf.function` on repeated training and simulation kernels
- Keras-tracked `tf.Variable` state for persistent normalization statistics
- `tf.constant` for fixed numerical bounds
- `tf.TensorArray` for compiled ergodic simulation loops

These design choices make the code easier to checkpoint, test, and extend than a NumPy-only research script.


## Why TensorFlow instead of NumPy

TensorFlow is used in the core training and inference path because the project
relies on differentiable models, automatic differentiation, checkpointing, and
graph compilation. In practice this gives the package capabilities that a
NumPy-only implementation does not provide by default:

- automatic gradients for policy and value networks
- compiled kernels via `tf.function` for repeated training and simulation loops
- native checkpointing of `tf.Variable` state inside subclassed `tf.keras.Model` objects
- deployable save/load paths for registered custom Keras objects

NumPy remains useful for lightweight array manipulation, reporting, and plotting,
but the learning system itself is intentionally TensorFlow-native.

## Pipeline architecture

The end-to-end workflow is orchestrated by `basic_mailer.pipeline.BasicMailerPipeline`,
which coordinates training, benchmarking, and estimation while preserving the
legacy `python -m Experiment.run_all` entrypoint for backward compatibility.
