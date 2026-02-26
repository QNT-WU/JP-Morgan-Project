# Test/test_integration_run_all.py
# It answers one question: “If I run the real entrypoint (run_all.py)
# like a user would, does the whole system execute end-to-end and produce the expected artifacts?”
# It does not test math correctness, convergence, or optimality.
# It tests engineering correctness.
import os
import subprocess
import sys


def test_run_all_creates_outputs(tmp_path):
    """
    Integration test:
    - runs the actual entrypoint module
    - verifies logs/checkpoints/figures exist
    Keep it tiny so it runs fast.
    """

    # tmp_path is a pytest fixture
    # It creates a fresh, isolated temporary directory
    # Automatically cleaned up after the test
    # Output directory setup
    # Constructs:/tmp/.../outputs/run_test
    out_dir = tmp_path / "outputs" / "run_test"
    out_dir_str = str(out_dir)

    # Build the command
    cmd = [
        sys.executable,  # Guarantees the same Python environment pytest is using
        "-m",
        "Experiment.run_all",  # Tests module resolution. Catches packaging / import bugs
        "--out",
        out_dir_str,
        "--epochs",  # Tiny hyperparameters: keeps test runtime under a few seconds
        "2",
        "--steps_per_epoch",
        "2",
        "--batch_size",
        "32",
        "--hidden_units",
        "8",
        "--no_benchmark",
    ]

    # run from project root; pytest already runs at repo root typically
    # This launches a new Python process
    res = subprocess.run(cmd, capture_output=True, text=True)

    # This is a hard gate:
    # If run_all.py crashes → test fails
    # Error message prints full stdout/stderr
    # This catches:
    # import errors
    # shape errors
    # gradient errors
    # checkpoint errors
    # plotting errors
    # filesystem permission errors
    assert (
        res.returncode == 0
    ), f"run_all failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # logs
    # Verify logs directory
    log_dir = os.path.join(out_dir_str, "logs")
    assert os.path.isdir(log_dir)
    # Verify JSONL logs exist and are non-empty
    # For each objective:
    # Ensures: logging code ran; file was written
    for name in ["obj1.jsonl", "obj2.jsonl", "obj3.jsonl"]:
        p = os.path.join(log_dir, name)
        assert os.path.isfile(p)
        # Guarantees:
        # at least one epoch logged
        # JSONL format not empty
        # training loop actually executed
        # This is crucial:
        # it verifies training + logging, not just file creation.
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) >= 1

    # checkpoints
    # Verify checkpoints
    ckpt_dir = os.path.join(out_dir_str, "checkpoints")
    assert os.path.isdir(ckpt_dir)
    for sub in ["obj1", "obj2", "obj3"]:
        p = os.path.join(ckpt_dir, sub)
        assert os.path.isdir(p)
        # TF checkpoints produce "ckpt-*" files and a "checkpoint" index file
        # We just check directory not empty
        assert len(os.listdir(p)) > 0

    # figures
    # Verify checkpoints
    # This guarantees:
    # ergodic simulation ran
    # matplotlib rendering worked
    # filesystem writes succeeded
    fig_dir = os.path.join(out_dir_str, "figures")
    assert os.path.isdir(fig_dir)

    expected_some = [
        "ergodic_set_obj1.png",
        "ergodic_set_obj2.png",
        "ergodic_set_obj3.png",
    ]
    for name in expected_some:
        assert os.path.isfile(os.path.join(fig_dir, name))


# What this test does NOT guarantee (by design):
# Economic correctness
# Convergence
# Optimal policy
# Euler residual magnitude
# Statistical validity
