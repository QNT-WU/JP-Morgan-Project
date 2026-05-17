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
    """Canonical output paths for Bayesian estimation artifacts."""

    estimation_dir: str
    figures_dir: str

    @property
    def bayes_results_json(self) -> str:
        """Return the JSON path for Bayesian summary results."""
        return os.path.join(self.estimation_dir, "bayes_results.json")

    @property
    def bayes_draws_npz(self) -> str:
        """Return the compressed NumPy path for posterior draws."""
        return os.path.join(self.estimation_dir, "bayes_draws.npz")

    @property
    def posterior_summary_csv(self) -> str:
        """Return the CSV path for posterior summary statistics."""
        return os.path.join(self.estimation_dir, "bayes_posterior_summary.csv")

    @property
    def posterior_summary_tex(self) -> str:
        """Return the TeX path for posterior summary statistics."""
        return os.path.join(self.estimation_dir, "bayes_posterior_summary.tex")

    @property
    def prior_to_posterior_csv(self) -> str:
        """Return the CSV path for prior-to-posterior updating diagnostics."""
        return os.path.join(self.estimation_dir, "table_bayes_prior_to_posterior.csv")

    @property
    def prior_to_posterior_tex(self) -> str:
        """Return the TeX path for prior-to-posterior updating diagnostics."""
        return os.path.join(self.estimation_dir, "table_bayes_prior_to_posterior.tex")

    @property
    def coverage_csv(self) -> str:
        """Return the CSV path for Bayesian credible-interval coverage flags."""
        return os.path.join(self.estimation_dir, "table_bayes_credible_interval_coverage.csv")

    @property
    def coverage_tex(self) -> str:
        """Return the TeX path for Bayesian credible-interval coverage flags."""
        return os.path.join(self.estimation_dir, "table_bayes_credible_interval_coverage.tex")

    @property
    def artifacts_manifest_json(self) -> str:
        """Return the JSON path for the Bayesian artifact manifest."""
        return os.path.join(self.estimation_dir, "bayes_artifacts.json")

    @classmethod
    def from_dirs(cls, estimation_dir: str, figures_dir: str) -> "BayesianArtifactPaths":
        """Create output directories and return a populated path bundle."""
        os.makedirs(estimation_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        return cls(estimation_dir=estimation_dir, figures_dir=figures_dir)


class BayesianArtifactWriter:
    """Write posterior draws, summaries, figures, and manifests to disk."""

    def __init__(self, paths: BayesianArtifactPaths) -> None:
        self.paths = paths

    def write_all(self, *, results: Mapping[str, object], theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, draws_u: np.ndarray, accepted: np.ndarray, target_log_prob: np.ndarray, true_parameters: Optional[Mapping[str, float]] = None) -> Dict[str, str]:
        """Write all Bayesian artifacts and return their paths.

        In addition to posterior draws and sampler diagnostics, this writer now
        creates two synthetic-recovery reports required by the written summary:
        a prior-to-posterior updating table and explicit credible-interval
        coverage flags for the true structural parameters.
        """
        truth = dict(true_parameters or {})
        prior_summary = dict(results.get("prior_summary", {})) if isinstance(results, Mapping) else {}
        posterior_rows = self._posterior_summary_rows(theta=theta, psi0=psi0, alpha=alpha, true_parameters=truth)
        prior_posterior_rows = self._prior_to_posterior_rows(posterior_rows=posterior_rows, prior_summary=prior_summary)
        coverage_rows = self._coverage_rows(posterior_rows=posterior_rows)

        self._write_draws_npz(draws_u=draws_u, theta=theta, psi0=psi0, alpha=alpha, accepted=accepted, target_log_prob=target_log_prob)
        self._write_rows_csv(self.paths.posterior_summary_csv, posterior_rows)
        self._write_rows_tex(self.paths.posterior_summary_tex, posterior_rows, caption="Bayesian posterior summary with true-value recovery diagnostics.", label="tab:bayes_posterior_summary")
        self._write_rows_csv(self.paths.prior_to_posterior_csv, prior_posterior_rows)
        self._write_rows_tex(self.paths.prior_to_posterior_tex, prior_posterior_rows, caption="Bayesian prior-to-posterior updating diagnostics.", label="tab:bayes_prior_to_posterior")
        self._write_rows_csv(self.paths.coverage_csv, coverage_rows)
        self._write_rows_tex(self.paths.coverage_tex, coverage_rows, caption="Bayesian credible-interval coverage flags for synthetic recovery.", label="tab:bayes_credible_interval_coverage")

        figure_paths = self._write_figures(theta=theta, psi0=psi0, alpha=alpha, accepted=accepted, target_log_prob=target_log_prob, true_parameters=truth, prior_summary=prior_summary)
        artifact_manifest = {
            "results_json": self.paths.bayes_results_json,
            "draws_npz": self.paths.bayes_draws_npz,
            "posterior_summary_csv": self.paths.posterior_summary_csv,
            "posterior_summary_tex": self.paths.posterior_summary_tex,
            "prior_to_posterior_csv": self.paths.prior_to_posterior_csv,
            "prior_to_posterior_tex": self.paths.prior_to_posterior_tex,
            "coverage_csv": self.paths.coverage_csv,
            "coverage_tex": self.paths.coverage_tex,
            **figure_paths,
        }
        payload = make_json_serializable(dict(results))
        payload["posterior_summary_rows"] = make_json_serializable(posterior_rows)
        payload["prior_to_posterior_rows"] = make_json_serializable(prior_posterior_rows)
        payload["credible_interval_coverage_rows"] = make_json_serializable(coverage_rows)
        payload["artifact_paths"] = make_json_serializable(artifact_manifest)
        with open(self.paths.bayes_results_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(self.paths.artifacts_manifest_json, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(artifact_manifest), f, indent=2)
        artifact_manifest["artifacts_manifest_json"] = self.paths.artifacts_manifest_json
        return artifact_manifest

    def _write_draws_npz(self, *, draws_u: np.ndarray, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, accepted: np.ndarray, target_log_prob: np.ndarray) -> None:
        np.savez_compressed(self.paths.bayes_draws_npz, draws_u=np.asarray(draws_u, dtype=np.float32), theta=np.asarray(theta, dtype=np.float32), psi0=np.asarray(psi0, dtype=np.float32), alpha=np.asarray(alpha, dtype=np.float32), accepted=np.asarray(accepted, dtype=bool), target_log_prob=np.asarray(target_log_prob, dtype=np.float32))

    def _posterior_summary_rows(self, *, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, true_parameters: Mapping[str, float]) -> list[Dict[str, object]]:
        """Return posterior summary rows with true-value coverage fields."""
        rows: list[Dict[str, object]] = []
        for name, arr in (("theta", theta), ("psi0", psi0), ("alpha", alpha)):
            flat = self._flatten(arr)
            p05 = float(np.quantile(flat, 0.05))
            p95 = float(np.quantile(flat, 0.95))
            mean = float(np.mean(flat))
            true_value = true_parameters.get(name)
            abs_error = None if true_value is None else abs(mean - float(true_value))
            rel_error = None if true_value is None or abs(float(true_value)) < 1e-12 else abs_error / abs(float(true_value))
            covered = None if true_value is None else bool(p05 <= float(true_value) <= p95)
            rows.append({
                "parameter": name,
                "true_value": true_value,
                "mean": mean,
                "median": float(np.median(flat)),
                "std": float(np.std(flat, ddof=0)),
                "p05": p05,
                "p95": p95,
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "covered_90pct": covered,
                "absolute_error_mean": abs_error,
                "relative_error_mean": rel_error,
            })
        return rows

    def _prior_to_posterior_rows(self, *, posterior_rows: list[Dict[str, object]], prior_summary: Mapping[str, object]) -> list[Dict[str, object]]:
        """Return rows comparing prior location/scale with posterior location/scale."""
        rows: list[Dict[str, object]] = []
        for row in posterior_rows:
            name = str(row["parameter"])
            prior = dict(prior_summary.get(name, {})) if isinstance(prior_summary, Mapping) else {}
            prior_mean = prior.get("mean")
            prior_std = prior.get("std")
            posterior_mean = row.get("mean")
            posterior_std = row.get("std")
            sd_ratio = None
            mean_shift = None
            if prior_std is not None and posterior_std is not None and abs(float(prior_std)) > 1e-12:
                sd_ratio = float(posterior_std) / float(prior_std)
            if prior_mean is not None and posterior_mean is not None:
                mean_shift = float(posterior_mean) - float(prior_mean)
            rows.append({
                "parameter": name,
                "prior_family": prior.get("family"),
                "prior_mean": prior_mean,
                "prior_std": prior_std,
                "posterior_mean": posterior_mean,
                "posterior_std": posterior_std,
                "posterior_median": row.get("median"),
                "posterior_p05": row.get("p05"),
                "posterior_p95": row.get("p95"),
                "posterior_to_prior_sd_ratio": sd_ratio,
                "posterior_minus_prior_mean": mean_shift,
            })
        return rows

    def _coverage_rows(self, *, posterior_rows: list[Dict[str, object]]) -> list[Dict[str, object]]:
        """Return explicit credible-interval coverage flags for each parameter."""
        rows: list[Dict[str, object]] = []
        for row in posterior_rows:
            rows.append({
                "parameter": row.get("parameter"),
                "true_value": row.get("true_value"),
                "ci_level": 0.90,
                "ci_lower": row.get("p05"),
                "ci_upper": row.get("p95"),
                "covered_90pct": row.get("covered_90pct"),
                "posterior_mean": row.get("mean"),
                "posterior_median": row.get("median"),
                "absolute_error_mean": row.get("absolute_error_mean"),
                "relative_error_mean": row.get("relative_error_mean"),
            })
        return rows

    @staticmethod
    def _write_rows_csv(path: str, rows: list[Dict[str, object]]) -> None:
        """Write a list of dictionaries to CSV with stable field ordering."""
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_rows_tex(path: str, rows: list[Dict[str, object]], *, caption: str, label: str) -> None:
        """Write a compact booktabs-style TeX table for report inclusion."""
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        def fmt(value: object) -> str:
            if value is None:
                return ""
            if isinstance(value, (bool, np.bool_)):
                return "Yes" if bool(value) else "No"
            if isinstance(value, (float, np.floating)):
                return f"{float(value):.6g}"
            return str(value).replace("_", "\\_")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\\begin{table}[!htbp]\\centering\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")
            f.write("\\scriptsize\n")
            f.write("\\begin{tabular}{" + "l" * len(fieldnames) + "}\\toprule\n")
            f.write(" & ".join(fieldnames).replace("_", "\\_") + " \\\\ \\midrule\n")
            for row in rows:
                f.write(" & ".join(fmt(row.get(k)) for k in fieldnames) + " \\\\ \n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    def _write_figures(self, *, theta: np.ndarray, psi0: np.ndarray, alpha: np.ndarray, accepted: np.ndarray, target_log_prob: np.ndarray, true_parameters: Mapping[str, float], prior_summary: Mapping[str, object]) -> Dict[str, str]:
        artifact_paths: Dict[str, str] = {}
        for name, arr in (("theta", theta), ("psi0", psi0), ("alpha", alpha)):
            trace_path = os.path.join(self.paths.figures_dir, f"bayes_trace_{name}.png")
            hist_path = os.path.join(self.paths.figures_dir, f"bayes_posterior_{name}.png")
            self._plot_trace(arr=arr, title=f"Bayesian trace: {name}", ylabel=name, out_path=trace_path)
            self._plot_histogram(arr=arr, title=f"Posterior distribution: {name}", xlabel=name, out_path=hist_path, truth=true_parameters.get(name))
            prior_overlay_path = os.path.join(self.paths.figures_dir, f"bayes_prior_to_posterior_{name}.png")
            self._plot_prior_to_posterior(arr=arr, parameter=name, prior=dict(prior_summary.get(name, {})) if isinstance(prior_summary, Mapping) else {}, truth=true_parameters.get(name), out_path=prior_overlay_path)
            artifact_paths[f"trace_{name}"] = trace_path
            artifact_paths[f"posterior_{name}"] = hist_path
            artifact_paths[f"prior_to_posterior_{name}"] = prior_overlay_path
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


    def _plot_prior_to_posterior(self, *, arr: np.ndarray, parameter: str, prior: Mapping[str, object], truth: Optional[float], out_path: str) -> None:
        """Plot posterior draws with a prior-density overlay when available."""
        values = self._flatten(arr)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(values, bins=min(30, max(10, int(np.sqrt(values.size)))), density=True, alpha=0.55, label="posterior")
        prior_mean = prior.get("mean")
        prior_std = prior.get("std")
        if prior_mean is not None and prior_std is not None and float(prior_std) > 0.0:
            x_min = min(float(np.min(values)), float(prior_mean) - 4.0 * float(prior_std))
            x_max = max(float(np.max(values)), float(prior_mean) + 4.0 * float(prior_std))
            if parameter in {"theta", "alpha"}:
                x_min = max(0.0, x_min)
                x_max = min(1.0, x_max)
            if x_max > x_min:
                x = np.linspace(x_min, x_max, 300)
                if parameter in {"theta", "alpha"}:
                    # The current prior for theta and alpha is Beta(2,2).
                    y = 6.0 * x * (1.0 - x)
                elif parameter == "psi0":
                    # The current psi0 prior is LogNormal(log(true psi0), 0.75).
                    scale = float(prior.get("scale", 0.75))
                    loc = float(prior.get("loc", np.log(max(float(prior_mean), 1e-12))))
                    y = np.zeros_like(x)
                    pos = x > 0.0
                    y[pos] = np.exp(-0.5 * ((np.log(x[pos]) - loc) / scale) ** 2) / (x[pos] * scale * np.sqrt(2.0 * np.pi))
                else:
                    y = np.exp(-0.5 * ((x - float(prior_mean)) / float(prior_std)) ** 2) / (float(prior_std) * np.sqrt(2.0 * np.pi))
                ax.plot(x, y, linewidth=2.0, label="prior density")
        ax.axvline(float(np.mean(values)), linestyle="--", linewidth=2.0, label="posterior mean")
        if truth is not None:
            ax.axvline(float(truth), linestyle=":", linewidth=2.0, label="true value")
        ax.set_title(f"Prior-to-posterior updating: {parameter}")
        ax.set_xlabel(parameter)
        ax.set_ylabel("density")
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
