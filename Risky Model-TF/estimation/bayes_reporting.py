"""Reporting utilities for Bayesian risky-debt estimation.

The Bayesian sampler already computes posterior draws and diagnostics. This
module is responsible only for packaging those objects into clean, stable,
report-friendly artifacts with canonical file names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import make_json_serializable


@dataclass(frozen=True)
class BayesianArtifactPaths:
    estimation_dir: str
    figures_dir: str

    @property
    def bayes_results_json(self) -> str:
        return os.path.join(self.estimation_dir, "bayes_results.json")

    @property
    def bayes_draws_npz(self) -> str:
        return os.path.join(self.estimation_dir, "bayes_draws.npz")

    @property
    def posterior_summary_csv(self) -> str:
        return os.path.join(self.estimation_dir, "bayes_posterior_summary.csv")

    @property
    def artifacts_manifest_json(self) -> str:
        return os.path.join(self.estimation_dir, "bayes_artifacts.json")

    @classmethod
    def from_dirs(cls, estimation_dir: str, figures_dir: str) -> "BayesianArtifactPaths":
        os.makedirs(estimation_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        return cls(estimation_dir=estimation_dir, figures_dir=figures_dir)


class BayesianArtifactWriter:
    def __init__(self, paths: BayesianArtifactPaths) -> None:
        self.paths = paths

    def write_all(self, *, results: Mapping[str, object], theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, draws_u: np.ndarray, accepted: np.ndarray, target_log_prob: np.ndarray, true_parameters: Optional[Mapping[str, float]] = None) -> Dict[str, str]:
        self._write_draws_npz(draws_u=draws_u, theta=theta, psi0=psi0, alpha=alpha, accepted=accepted, target_log_prob=target_log_prob)
        self._write_posterior_summary_csv(theta=theta, psi0=psi0, alpha=alpha)
        figure_paths = self._write_figures(theta=theta, psi0=psi0, alpha=alpha, accepted=accepted, target_log_prob=target_log_prob, true_parameters=true_parameters or {})
        artifact_manifest = {"results_json": self.paths.bayes_results_json, "draws_npz": self.paths.bayes_draws_npz, "posterior_summary_csv": self.paths.posterior_summary_csv, **figure_paths}
        payload = make_json_serializable(dict(results))
        payload["artifact_paths"] = make_json_serializable(artifact_manifest)
        with open(self.paths.bayes_results_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(self.paths.artifacts_manifest_json, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(artifact_manifest), f, indent=2)
        artifact_manifest["artifacts_manifest_json"] = self.paths.artifacts_manifest_json
        return artifact_manifest

    def _write_draws_npz(self, *, draws_u: np.ndarray, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, accepted: np.ndarray, target_log_prob: np.ndarray) -> None:
        np.savez_compressed(self.paths.bayes_draws_npz, draws_u=np.asarray(draws_u, dtype=np.float32), theta=np.asarray(theta, dtype=np.float32), psi0=np.asarray(psi0, dtype=np.float32), alpha=np.asarray(alpha, dtype=np.float32), accepted=np.asarray(accepted, dtype=bool), target_log_prob=np.asarray(target_log_prob, dtype=np.float32))

    def _write_posterior_summary_csv(self, *, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray) -> None:
        rows = []
        for name, arr in (("theta", theta), ("psi0", psi0), ("alpha", alpha)):
            flat = self._flatten(arr)
            rows.append({"parameter": name, "mean": float(np.mean(flat)), "median": float(np.median(flat)), "std": float(np.std(flat, ddof=0)), "p05": float(np.quantile(flat, 0.05)), "p95": float(np.quantile(flat, 0.95)), "min": float(np.min(flat)), "max": float(np.max(flat))})
        with open(self.paths.posterior_summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["parameter", "mean", "median", "std", "p05", "p95", "min", "max"])
            writer.writeheader(); writer.writerows(rows)

    def _write_figures(self, *, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, accepted: np.ndarray, target_log_prob: np.ndarray, true_parameters: Mapping[str, float]) -> Dict[str, str]:
        artifact_paths: Dict[str, str] = {}
        for name, arr in (("theta", theta), ("psi0", psi0), ("alpha", alpha)):
            trace_path = os.path.join(self.paths.figures_dir, f"bayes_trace_{name}.png")
            hist_path = os.path.join(self.paths.figures_dir, f"bayes_posterior_{name}.png")
            self._plot_trace(arr=arr, title=f"Bayesian trace: {name}", ylabel=name, out_path=trace_path)
            self._plot_histogram(arr=arr, title=f"Posterior distribution: {name}", xlabel=name, out_path=hist_path, truth=true_parameters.get(name))
            artifact_paths[f"trace_{name}"] = trace_path
            artifact_paths[f"posterior_{name}"] = hist_path
        tlp_path = os.path.join(self.paths.figures_dir, "bayes_target_log_prob.png")
        acc_path = os.path.join(self.paths.figures_dir, "bayes_acceptance_rate.png")
        self._plot_trace(arr=target_log_prob, title="Target log posterior", ylabel="log posterior", out_path=tlp_path)
        self._plot_running_acceptance(accepted=accepted, out_path=acc_path)
        artifact_paths["target_log_prob"] = tlp_path
        artifact_paths["acceptance_rate"] = acc_path
        return artifact_paths

    @staticmethod
    def _flatten(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).reshape(-1)

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        elif x.ndim > 2:
            x = x.reshape(x.shape[0], -1)
        return x

    def _plot_trace(self, *, arr: np.ndarray, title: str, ylabel: str, out_path: str) -> None:
        values = self._ensure_2d(arr)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(values.shape[0])
        for chain_idx in range(values.shape[1]):
            ax.plot(x, values[:, chain_idx], linewidth=1.1, label=f"chain {chain_idx + 1}")
        ax.set_title(title)
        ax.set_xlabel("draw")
        ax.set_ylabel(ylabel)
        if values.shape[1] > 1:
            ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_histogram(self, *, arr: np.ndarray, title: str, xlabel: str, out_path: str, truth: Optional[float] = None) -> None:
        values = self._flatten(arr)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=min(30, max(10, int(np.sqrt(values.size)))), density=False, alpha=0.85)
        ax.axvline(float(np.mean(values)), linewidth=2.0, linestyle="--", label="posterior mean")
        if truth is not None:
            ax.axvline(float(truth), linewidth=2.0, linestyle=":", label="true value")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("frequency")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_running_acceptance(self, *, accepted: np.ndarray, out_path: str) -> None:
        accepted_2d = self._ensure_2d(accepted).astype(float)
        draw_axis = np.arange(1, accepted_2d.shape[0] + 1)
        running = np.cumsum(accepted_2d, axis=0) / draw_axis[:, None]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for chain_idx in range(running.shape[1]):
            ax.plot(draw_axis, running[:, chain_idx], linewidth=1.1, label=f"chain {chain_idx + 1}")
        ax.set_title("Running acceptance rate")
        ax.set_xlabel("draw")
        ax.set_ylabel("acceptance rate")
        ax.set_ylim(0.0, 1.0)
        if running.shape[1] > 1:
            ax.legend(loc="best", frameon=False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)
