# Test/test_integration_run_all.py
# End-to-end integration test:
# “If I run the real entrypoint (Experiment/run_all.py) like a user would,
# does the whole system execute and produce expected artifacts?”
import os
import subprocess
import sys


def test_run_all_creates_outputs(tmp_path):
    out_dir = tmp_path / "outputs" / "run_test"
    out_dir_str = str(out_dir)

    cmd = [
        sys.executable,
        "-m",
        "Experiment.run_all",
        "--out",
        out_dir_str,
        "--epochs",
        "2",
        "--steps_per_epoch",
        "2",
        "--batch_size",
        "32",
        "--hidden_units",
        "8",
        "--no_benchmark",
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)

    assert (
        res.returncode == 0
    ), f"run_all failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    # logs
    log_dir = os.path.join(out_dir_str, "logs")
    assert os.path.isdir(log_dir)

    for name in ["obj1.jsonl", "obj2.jsonl", "obj3.jsonl"]:
        p = os.path.join(log_dir, name)
        assert os.path.isfile(p)
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        assert len(lines) >= 1

    # checkpoints
    ckpt_dir = os.path.join(out_dir_str, "checkpoints")
    assert os.path.isdir(ckpt_dir)
    for sub in ["obj1", "obj2", "obj3"]:
        p = os.path.join(ckpt_dir, sub)
        assert os.path.isdir(p)
        assert len(os.listdir(p)) > 0

    # figures
    fig_dir = os.path.join(out_dir_str, "figures")
    assert os.path.isdir(fig_dir)

    expected_some = [
        "ergodic_set_obj1.png",
        "ergodic_set_obj2.png",
        "ergodic_set_obj3.png",
    ]
    for name in expected_some:
        assert os.path.isfile(os.path.join(fig_dir, name))
