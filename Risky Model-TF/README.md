# Risky Debt Model (k, b, z) — Mailer/MMW Objectives in TensorFlow

This repository implements the risky-debt corporate finance model with state variables
(k_t, b_t, z_t) using TensorFlow and TensorFlow Probability, following the
Mailer/MMW deep-learning framework.

The model features:
- Endogenous risky debt pricing via lenders’ zero-profit condition
- Limited liability default option (outer max)
- Equity issuance costs (fixed + proportional when e < 0)
- Convex capital adjustment costs
- AR(1) productivity shocks in logs

The code implements three objectives:
1. Lifetime reward maximization (Objective 1)
2. Euler residual minimization (Objective 2)
3. Bellman residual minimization (Objective 3, with pricing handled consistently)

It produces:
- training logs (JSONL)
- checkpoints (TensorFlow)
- figures (ergodic-set plots; effectiveness-measure plots if enabled)

## Project Layout

Typical structure:

project_root/
    README.md
    pyproject.toml

    risky_mailer/
        config.py
        states.py
        primitives_risky.py
        networks.py
