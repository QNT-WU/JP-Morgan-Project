"""Report-oriented artifact generation for risky-debt estimation.

This module adds a report layer on top of the existing risky-debt estimation
results so the saved outputs are easier to use in a LaTeX write-up. The goal is
not to change the underlying estimators, but to package the already-computed
results into stable CSV, TeX, JSON, and PNG artifacts in a style closer to the
basic-model workflow.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import make_json_serializable


def _safe_float(x, default: float = float("nan")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _safe_bool(x) -> bool:
    try:
        return bool(x)
    except Exception:
        return False


def _escape_tex(text: object) -> str:
    s = str(text)
    replacements = {
        "\\": r"\\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def _format_value(value: object, decimals: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(x):
        return ""
    if x == 0.0:
        return f"{0.0:.{decimals}f}"
    if abs(x) >= 1.0e4 or abs(x) < 1.0e-4:
        return f"{x:.3e}"
    return f"{x:.{decimals}f}"


def _write_csv(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_tex_table(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str], caption: Optional[str] = None, label: Optional[str] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = "l" + "r" * max(0, len(fieldnames) - 1)
    with open(path, "w", encoding="utf-8") as f:
        if caption or label:
            f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\hline\n")
        f.write(" & ".join(_escape_tex(name) for name in fieldnames) + r" \\" + "\n")
        f.write("\\hline\n")
        for row in rows:
            vals = [_escape_tex(_format_value(row.get(name, ""))) for name in fieldnames]
            f.write(" & ".join(vals) + r" \\" + "\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        if caption:
            f.write(f"\\caption{{{_escape_tex(caption)}}}\n")
        if label:
            f.write(f"\\label{{{_escape_tex(label)}}}\n")
        if caption or label:
            f.write("\\end{table}\n")


class RiskyDebtEstimationReportWriter:
    def __init__(self, *, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def write_all(self, *, gmm_res: Optional[Mapping[str, object]] = None, smm_res: Optional[Mapping[str, object]] = None, bayes_res: Optional[Mapping[str, object]] = None) -> Dict[str, object]:
        manifest: Dict[str, object] = {}
        if gmm_res and smm_res:
            manifest["frequentist"] = self._write_frequentist_reports(gmm_res=gmm_res, smm_res=smm_res)
        if bayes_res:
            manifest["bayesian"] = self._write_bayesian_reports(bayes_res=bayes_res)
        manifest["combined_summary"] = self._write_combined_summary(gmm_res=gmm_res, smm_res=smm_res, bayes_res=bayes_res)
        path = os.path.join(self.out_dir, "report_artifacts.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(manifest), f, indent=2)
        manifest["report_artifacts_json"] = path
        return manifest

    def _write_frequentist_reports(self, *, gmm_res: Mapping[str, object], smm_res: Mapping[str, object]) -> Dict[str, object]:
        method_records = self._build_method_records(gmm_res=gmm_res, smm_res=smm_res)
        raw_results_rows = self._build_raw_results_rows(method_records)
        raw_start_rows = self._build_raw_start_rows(method_records)
        estimation_results = self._build_estimation_results_json(method_records)
        estimation_summary = self._build_estimation_summary_json(method_records, gmm_res=gmm_res, smm_res=smm_res)
        estimation_results_path = os.path.join(self.out_dir, "estimation_results.json")
        with open(estimation_results_path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(estimation_results), f, indent=2)
        estimation_summary_path = os.path.join(self.out_dir, "estimation_summary.json")
        with open(estimation_summary_path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(estimation_summary), f, indent=2)
        self._write_table_pair(stem="table_parameter_recovery", rows=self._parameter_recovery_rows(method_records), fieldnames=["method", "parameter", "true_value", "estimate", "std_error", "abs_error", "rel_error", "recovery_score_l2", "final_success"], caption="Risky-debt parameter recovery across frequentist estimators.", label="tab:risky_parameter_recovery")
        self._write_table_pair(stem="table_computation_convergence", rows=self._computation_rows(method_records), fieldnames=["method", "runtime_seconds", "stage1_objective", "stage2_objective", "stage1_success", "stage2_success", "final_success", "n_starts", "winner_start", "stage2_loss_dispersion_std", "stage2_loss_dispersion_range", "stage2_message", "stage2_nfev", "condition_number"], caption="Risky-debt computational and convergence comparison for GMM/SMM.", label="tab:risky_computation_convergence")
        self._write_table_pair(stem="table_final_gmm_vs_smm_comparison", rows=self._final_comparison_rows(method_records), fieldnames=["method", "recovery_score", "fit_score", "fit_metric", "runtime_seconds", "final_success", "overall_ranking"], caption="Risky-debt frequentist comparison summary.", label="tab:risky_final_gmm_vs_smm_comparison")
        self._write_table_pair(stem="table_gmm_moment_fit", rows=self._gmm_moment_rows(method_records), fieldnames=["method", "g_norm_l2", "g_max_abs", "mean_abs_zero_profit_moment", "stage1_objective", "stage2_objective", "final_success"], caption="Risky-debt GMM moment-fit diagnostics.", label="tab:risky_gmm_moment_fit")
        self._write_table_pair(stem="table_smm_moment_fit", rows=self._smm_moment_rows(method_records), fieldnames=["method", "moment", "observed", "simulated", "raw_error", "percent_error", "standardized_error", "final_success"], caption="Risky-debt SMM moment fit by target moment.", label="tab:risky_smm_moment_fit")
        self._write_table_pair(stem="table_pricing_default_fit", rows=self._pricing_default_rows(method_records), fieldnames=["method", "metric", "observed", "simulated", "error", "final_success"], caption="Risky-debt pricing and default fit across frequentist estimators.", label="tab:risky_pricing_default_fit")
        _write_csv(os.path.join(self.out_dir, "raw_results.csv"), raw_results_rows, list(raw_results_rows[0].keys()) if raw_results_rows else ["method"])
        _write_csv(os.path.join(self.out_dir, "raw_start_results.csv"), raw_start_rows, list(raw_start_rows[0].keys()) if raw_start_rows else ["method"])
        self._plot_parameter_recovery(method_records, os.path.join(self.out_dir, "plot_parameter_recovery.png"))
        self._plot_smm_moment_fit(method_records, os.path.join(self.out_dir, "plot_smm_moment_fit.png"))
        self._plot_smm_standardized_errors(method_records, os.path.join(self.out_dir, "plot_smm_standardized_errors.png"))
        self._plot_objective_convergence(method_records, os.path.join(self.out_dir, "plot_objective_convergence.png"))
        self._plot_final_comparison(method_records, os.path.join(self.out_dir, "plot_final_comparison.png"))
        self._plot_pricing_default_fit(method_records, os.path.join(self.out_dir, "plot_pricing_default_fit.png"))
        return {
            "estimation_results_json": estimation_results_path,
            "estimation_summary_json": estimation_summary_path,
            "raw_results_csv": os.path.join(self.out_dir, "raw_results.csv"),
            "raw_start_results_csv": os.path.join(self.out_dir, "raw_start_results.csv"),
            "table_parameter_recovery_csv": os.path.join(self.out_dir, "table_parameter_recovery.csv"),
            "table_parameter_recovery_tex": os.path.join(self.out_dir, "table_parameter_recovery.tex"),
            "table_computation_convergence_csv": os.path.join(self.out_dir, "table_computation_convergence.csv"),
            "table_computation_convergence_tex": os.path.join(self.out_dir, "table_computation_convergence.tex"),
            "table_final_comparison_csv": os.path.join(self.out_dir, "table_final_gmm_vs_smm_comparison.csv"),
            "table_final_comparison_tex": os.path.join(self.out_dir, "table_final_gmm_vs_smm_comparison.tex"),
            "table_gmm_moment_fit_csv": os.path.join(self.out_dir, "table_gmm_moment_fit.csv"),
            "table_gmm_moment_fit_tex": os.path.join(self.out_dir, "table_gmm_moment_fit.tex"),
            "table_smm_moment_fit_csv": os.path.join(self.out_dir, "table_smm_moment_fit.csv"),
            "table_smm_moment_fit_tex": os.path.join(self.out_dir, "table_smm_moment_fit.tex"),
            "table_pricing_default_fit_csv": os.path.join(self.out_dir, "table_pricing_default_fit.csv"),
            "table_pricing_default_fit_tex": os.path.join(self.out_dir, "table_pricing_default_fit.tex"),
            "plot_parameter_recovery_png": os.path.join(self.out_dir, "plot_parameter_recovery.png"),
            "plot_smm_moment_fit_png": os.path.join(self.out_dir, "plot_smm_moment_fit.png"),
            "plot_smm_standardized_errors_png": os.path.join(self.out_dir, "plot_smm_standardized_errors.png"),
            "plot_objective_convergence_png": os.path.join(self.out_dir, "plot_objective_convergence.png"),
            "plot_final_comparison_png": os.path.join(self.out_dir, "plot_final_comparison.png"),
            "plot_pricing_default_fit_png": os.path.join(self.out_dir, "plot_pricing_default_fit.png"),
        }

    def _build_method_records(self, *, gmm_res: Mapping[str, object], smm_res: Mapping[str, object]) -> List[Dict[str, object]]:
        stage1_gmm = gmm_res.get("stage1", {})
        stage1_smm = smm_res.get("stage1", {})
        gmm_runtime = _safe_float(gmm_res.get("runtime_sec"), np.nan)
        smm_runtime = _safe_float(smm_res.get("runtime_sec"), np.nan)
        records: List[Dict[str, object]] = []
        for method, payload in gmm_res.get("variants", {}).items():
            rec = self._extract_method_record(method=method.replace("-", "_"), family="GMM", weight_method="newey_west" if method.endswith("B") else "standard", method_payload=payload, stage1_payload=stage1_gmm)
            rec["runtime_seconds"] = gmm_runtime
            records.append(rec)
        for method, payload in smm_res.get("variants", {}).items():
            rec = self._extract_method_record(method=method.replace("-", "_"), family="SMM", weight_method="newey_west" if method.endswith("B") else "standard", method_payload=payload, stage1_payload=stage1_smm)
            rec["runtime_seconds"] = smm_runtime
            records.append(rec)
        return records

    def _extract_method_record(self, *, method: str, family: str, weight_method: str, method_payload: Mapping[str, object], stage1_payload: Mapping[str, object]) -> Dict[str, object]:
        starts = list(method_payload.get("starts", []))
        best_start_id = int(_safe_float(method_payload.get("best_start_id"), 0.0)) if method_payload.get("best_start_id") is not None else 0
        best_start = starts[best_start_id] if starts and 0 <= best_start_id < len(starts) else {}
        stage1_runs = list(stage1_payload.get("runs", []))
        stage1_best_id = None
        stage1_successful = [r for r in stage1_runs if _safe_bool(r.get("success", False)) and np.isfinite(_safe_float(r.get("objective"), float("inf")))]
        stage1_best = None
        if stage1_successful:
            stage1_best = min(stage1_successful, key=lambda r: _safe_float(r.get("objective"), float("inf")))
            stage1_best_id = int(_safe_float(stage1_best.get("start_id"), 0.0))
        moment_table = list(method_payload.get("moment_table", []))
        pricing_fit = dict(method_payload.get("pricing_default_fit", {}))
        param_table = dict(method_payload.get("parameter_table", {}))
        fit_metric_name, fit_score = self._fit_metric_for_method(family=family, payload=method_payload)
        return {
            "method": method,
            "family": family,
            "weight_method": weight_method,
            "payload": method_payload,
            "stage1_payload": stage1_payload,
            "stage1_best": stage1_best,
            "stage2_best": best_start,
            "stage1_best_start_id": stage1_best_id,
            "stage2_best_start_id": best_start_id,
            "stage1_success": _safe_bool(stage1_best.get("success") if stage1_best else False),
            "stage2_success": _safe_bool(method_payload.get("success", False)),
            "final_success": _safe_bool(method_payload.get("success", False)) and _safe_bool(method_payload.get("convergence_flag", False)),
            "runtime_seconds": float("nan"),
            "parameter_table": param_table,
            "moment_table": moment_table,
            "pricing_default_fit": pricing_fit,
            "recovery_score": _safe_float(method_payload.get("recovery_score"), float("inf")),
            "fit_score": fit_score,
            "fit_metric": fit_metric_name,
            "condition_number": _safe_float(method_payload.get("weight_matrix_condition"), float("nan")),
            "stage1_objective": _safe_float(stage1_best.get("objective") if stage1_best else np.nan, np.nan),
            "stage2_objective": _safe_float(method_payload.get("objective"), np.nan),
            "stage2_nfev": int(_safe_float(best_start.get("evals"), 0.0)),
            "stage2_message": "Converged" if _safe_bool(method_payload.get("convergence_flag", False)) else "Maximum evaluations or simplex tolerance not satisfied",
            "stage2_loss_dispersion_std": _safe_float(method_payload.get("multistart_summary", {}).get("objective_std"), np.nan),
            "stage2_loss_dispersion_range": self._objective_range(method_payload.get("multistart_summary", {})),
            "n_starts": int(_safe_float(method_payload.get("multistart_summary", {}).get("n_starts"), len(starts))),
            "winner_start": best_start_id,
            "g_norm": _safe_float(method_payload.get("g_norm"), np.nan),
            "g_max_abs": _safe_float(method_payload.get("max_abs_moment"), np.nan),
            "mean_abs_zero_profit_moment": _safe_float(method_payload.get("mean_abs_zero_profit_moment"), np.nan),
            "specification_stat": _safe_float(method_payload.get("specification_stat"), np.nan),
        }

    @staticmethod
    def _objective_range(summary: Mapping[str, object]) -> float:
        lo = _safe_float(summary.get("objective_min"), np.nan)
        hi = _safe_float(summary.get("objective_max"), np.nan)
        if np.isfinite(lo) and np.isfinite(hi):
            return float(hi - lo)
        return float("nan")

    @staticmethod
    def _fit_metric_for_method(*, family: str, payload: Mapping[str, object]) -> Tuple[str, float]:
        if family == "GMM":
            return "g_norm_l2", _safe_float(payload.get("g_norm"), float("inf"))
        moment_table = list(payload.get("moment_table", []))
        if moment_table:
            vals = [abs(_safe_float(row.get("standardized_error"), np.nan)) for row in moment_table]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                return "mean_abs_standardized_error", float(np.mean(vals))
        return "objective", _safe_float(payload.get("objective"), float("inf"))

    def _build_estimation_results_json(self, method_records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for rec in method_records:
            param_table = rec.get("parameter_table", {})
            out[rec["method"]] = {
                "method_name": rec["method"],
                "family": rec["family"],
                "weight_method": rec["weight_method"],
                "final_params": {k: v.get("hat") for k, v in param_table.items()},
                "standard_errors": {k: v.get("std_error") for k, v in param_table.items()},
                "stage1": {"best_start_id": rec.get("stage1_best_start_id"), "best_loss": rec.get("stage1_objective"), "success": rec.get("stage1_success")},
                "stage2": {"best_start_id": rec.get("stage2_best_start_id"), "best_loss": rec.get("stage2_objective"), "success": rec.get("stage2_success")},
                "pricing_default_fit": rec.get("pricing_default_fit", {}),
                "moment_fit": {"fit_metric": rec.get("fit_metric"), "fit_score": rec.get("fit_score")},
                "elapsed_seconds": rec.get("runtime_seconds"),
            }
        return out

    def _build_estimation_summary_json(self, method_records: Sequence[Mapping[str, object]], *, gmm_res: Mapping[str, object], smm_res: Mapping[str, object]) -> Dict[str, object]:
        synth_path = os.path.join(self.out_dir, "smm_synth_data.npz")
        observed_sample_size = None
        if os.path.exists(synth_path):
            with np.load(synth_path) as data:
                n_paths = int(np.asarray(data.get("n_paths", 1)).reshape(())) if "n_paths" in data else 1
                t_eff = int(np.asarray(data.get("T_eff", 0)).reshape(())) if "T_eff" in data else int(np.asarray(data["k"]).reshape(-1).size)
                observed_sample_size = int(max(0, n_paths * max(0, t_eff - 1)))
        methods = {}
        for rec in method_records:
            flat = {"method": rec["method"], "family": rec["family"], "weight_method": rec["weight_method"], "recovery_score": rec["recovery_score"], "fit_metric": rec["fit_metric"], "fit_score": rec["fit_score"], "runtime_seconds": rec["runtime_seconds"], "condition_number": rec["condition_number"], "stage1_objective": rec["stage1_objective"], "stage2_objective": rec["stage2_objective"], "stage1_success": rec["stage1_success"], "stage2_success": rec["stage2_success"], "final_success": rec["final_success"]}
            for name, vals in rec.get("parameter_table", {}).items():
                flat[f"{name}_hat"] = vals.get("hat")
                flat[f"se_{name}"] = vals.get("std_error")
            methods[rec["method"]] = flat
        return {"methods": methods, "observed_sample_size": observed_sample_size, "simulated_sample_size": observed_sample_size, "true_params": make_json_serializable(gmm_res.get("theta_true", smm_res.get("theta_true", {}))), "dgp_source": smm_res.get("dgp_source", gmm_res.get("dgp_source", "obj1_nn")), "continuation_horizon": smm_res.get("continuation_horizon", gmm_res.get("continuation_horizon"))}

    def _parameter_recovery_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        rows = []
        for rec in method_records:
            for param, vals in rec.get("parameter_table", {}).items():
                rows.append({"method": rec["method"], "parameter": param, "true_value": vals.get("true"), "estimate": vals.get("hat"), "std_error": vals.get("std_error"), "abs_error": vals.get("abs_error"), "rel_error": vals.get("rel_error"), "recovery_score_l2": rec.get("recovery_score"), "final_success": rec.get("final_success")})
        return rows

    def _computation_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        return [{"method": rec["method"], "runtime_seconds": rec.get("runtime_seconds"), "stage1_objective": rec.get("stage1_objective"), "stage2_objective": rec.get("stage2_objective"), "stage1_success": rec.get("stage1_success"), "stage2_success": rec.get("stage2_success"), "final_success": rec.get("final_success"), "n_starts": rec.get("n_starts"), "winner_start": rec.get("winner_start"), "stage2_loss_dispersion_std": rec.get("stage2_loss_dispersion_std"), "stage2_loss_dispersion_range": rec.get("stage2_loss_dispersion_range"), "stage2_message": rec.get("stage2_message"), "stage2_nfev": rec.get("stage2_nfev"), "condition_number": rec.get("condition_number")} for rec in method_records]

    def _final_comparison_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        records = [dict(rec) for rec in method_records]
        recovery_rank = self._rank_metric([_safe_float(rec.get("recovery_score"), float("inf")) for rec in records])
        fit_rank = self._rank_metric([_safe_float(rec.get("fit_score"), float("inf")) for rec in records])
        rows: List[Dict[str, object]] = []
        for i, rec in enumerate(records):
            rows.append({"method": rec["method"], "recovery_score": rec.get("recovery_score"), "fit_score": rec.get("fit_score"), "fit_metric": rec.get("fit_metric"), "runtime_seconds": rec.get("runtime_seconds"), "final_success": rec.get("final_success"), "overall_ranking": recovery_rank[i] + fit_rank[i]})
        rows.sort(key=lambda r: (_safe_float(r["overall_ranking"], float("inf")), str(r["method"])))
        return rows

    @staticmethod
    def _rank_metric(values: Sequence[float]) -> List[int]:
        order = sorted(range(len(values)), key=lambda i: (not np.isfinite(values[i]), values[i]))
        ranks = [0] * len(values)
        for rank, idx in enumerate(order, start=1):
            ranks[idx] = rank
        return ranks

    def _gmm_moment_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        return [{"method": rec["method"], "g_norm_l2": rec.get("g_norm"), "g_max_abs": rec.get("g_max_abs"), "mean_abs_zero_profit_moment": rec.get("mean_abs_zero_profit_moment"), "stage1_objective": rec.get("stage1_objective"), "stage2_objective": rec.get("stage2_objective"), "final_success": rec.get("final_success")} for rec in method_records if rec.get("family") == "GMM"]

    def _smm_moment_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        rows = []
        for rec in method_records:
            if rec.get("family") != "SMM":
                continue
            for row in rec.get("moment_table", []):
                rows.append({"method": rec["method"], "moment": row.get("moment"), "observed": row.get("observed"), "simulated": row.get("simulated"), "raw_error": row.get("raw_error"), "percent_error": row.get("percent_error"), "standardized_error": row.get("standardized_error"), "final_success": rec.get("final_success")})
        return rows

    def _pricing_default_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        rows = []
        metrics = ["mean_spread", "default_rate", "mean_recovery_default", "mean_zero_profit_residual", "mean_abs_zero_profit_residual"]
        for rec in method_records:
            fit = rec.get("pricing_default_fit", {})
            observed = fit.get("observed", {})
            simulated = fit.get("simulated", {})
            errors = fit.get("errors", {})
            for metric in metrics:
                rows.append({"method": rec["method"], "metric": metric, "observed": observed.get(metric), "simulated": simulated.get(metric), "error": errors.get(metric), "final_success": rec.get("final_success")})
        return rows

    def _build_raw_results_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        rows = []
        for rec in method_records:
            row = {"method": rec["method"], "family": rec["family"], "weight_method": rec["weight_method"], "stage1_success": rec["stage1_success"], "stage2_success": rec["stage2_success"], "final_success": rec["final_success"], "stage1_best_start_id": rec["stage1_best_start_id"], "stage2_best_start_id": rec["stage2_best_start_id"], "stage1_best_loss": rec["stage1_objective"], "stage2_best_loss": rec["stage2_objective"], "stage1_n_starts": rec.get("stage1_payload", {}).get("summary", {}).get("n_starts"), "stage2_n_starts": rec.get("n_starts"), "stage1_nfev": rec.get("stage1_best", {}).get("evals") if rec.get("stage1_best") else None, "stage2_nfev": rec.get("stage2_nfev"), "elapsed_seconds": rec.get("runtime_seconds"), "condition_number": rec.get("condition_number"), "fit_metric": rec.get("fit_metric"), "fit_score": rec.get("fit_score")}
            for name, vals in rec.get("parameter_table", {}).items():
                row[f"{name}_hat"] = vals.get("hat")
                row[f"se_{name}"] = vals.get("std_error")
            for metric, val in rec.get("pricing_default_fit", {}).get("errors", {}).items():
                row[f"pricing_err_{metric}"] = val
            if rec.get("family") == "GMM":
                row["g_norm"] = rec.get("g_norm")
                row["g_max_abs"] = rec.get("g_max_abs")
                row["mean_abs_zero_profit_moment"] = rec.get("mean_abs_zero_profit_moment")
            else:
                for moment_row in rec.get("moment_table", []):
                    row[f"gap_{moment_row.get('moment')}"] = moment_row.get("raw_error")
                    row[f"std_gap_{moment_row.get('moment')}"] = moment_row.get("standardized_error")
            rows.append(row)
        return rows

    def _build_raw_start_rows(self, method_records: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
        rows = []
        for rec in method_records:
            for stage_name, starts, best_id in [("stage1", rec.get("stage1_payload", {}).get("runs", []), rec.get("stage1_best_start_id")), ("stage2", rec.get("payload", {}).get("starts", []), rec.get("stage2_best_start_id"))]:
                for start in starts:
                    sid = int(_safe_float(start.get("start_id"), 0.0)) if start.get("start_id") is not None else 0
                    rows.append({"method": rec["method"], "stage": stage_name, "start_id": sid, "is_best": bool(best_id is not None and sid == best_id), "success": _safe_bool(start.get("success", False)), "converged": _safe_bool(start.get("converged", False)), "evals": start.get("evals"), "objective": start.get("objective"), "simplex_diameter": start.get("simplex_diameter"), "f_spread": start.get("f_spread"), "x0": json.dumps(make_json_serializable(start.get("start_theta", []))), "x_hat": json.dumps(make_json_serializable(start.get("theta_hat_vector", [])))})
        return rows

    def _write_table_pair(self, *, stem: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str], caption: Optional[str], label: Optional[str]) -> None:
        _write_csv(os.path.join(self.out_dir, f"{stem}.csv"), rows, fieldnames)
        _write_tex_table(os.path.join(self.out_dir, f"{stem}.tex"), rows, fieldnames, caption=caption, label=label)

    def _plot_parameter_recovery(self, method_records, out_path):
        rows = self._parameter_recovery_rows(method_records)
        if not rows:
            return
        methods = sorted({row["method"] for row in rows})
        params = sorted({row["parameter"] for row in rows})
        x = np.arange(len(methods), dtype=float)
        width = 0.8 / max(1, len(params))
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        for j, param in enumerate(params):
            vals = []
            for method in methods:
                row = next(r for r in rows if r["method"] == method and r["parameter"] == param)
                vals.append(_safe_float(row["abs_error"], np.nan))
            ax.bar(x + (j - (len(params) - 1) / 2.0) * width, vals, width=width, label=param)
        ax.set_title("Risky-debt parameter recovery")
        ax.set_xlabel("method")
        ax.set_ylabel("absolute error")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_smm_moment_fit(self, method_records, out_path):
        rows = self._smm_moment_rows(method_records)
        if not rows:
            return
        methods = sorted({row["method"] for row in rows})
        moments = [row["moment"] for row in rows if row["method"] == methods[0]]
        x = np.arange(len(moments), dtype=float)
        width = 0.38
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        for offset, method in zip([-0.5 * width, 0.5 * width], methods[:2]):
            vals = [abs(_safe_float(next(r for r in rows if r["method"] == method and r["moment"] == m)["raw_error"], np.nan)) for m in moments]
            ax.bar(x + offset, vals, width=width, label=method)
        ax.set_title("Risky-debt SMM moment fit")
        ax.set_xlabel("moment")
        ax.set_ylabel("absolute raw error")
        ax.set_xticks(x); ax.set_xticklabels(moments, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_smm_standardized_errors(self, method_records, out_path):
        rows = self._smm_moment_rows(method_records)
        if not rows:
            return
        methods = sorted({row["method"] for row in rows})
        moments = [row["moment"] for row in rows if row["method"] == methods[0]]
        x = np.arange(len(moments), dtype=float)
        width = 0.38
        fig, ax = plt.subplots(figsize=(10.0, 4.8))
        for offset, method in zip([-0.5 * width, 0.5 * width], methods[:2]):
            vals = [_safe_float(next(r for r in rows if r["method"] == method and r["moment"] == m)["standardized_error"], np.nan) for m in moments]
            ax.bar(x + offset, vals, width=width, label=method)
        ax.axhline(0.0, linewidth=1.0)
        ax.set_title("Risky-debt SMM standardized moment errors")
        ax.set_xlabel("moment"); ax.set_ylabel("standardized error")
        ax.set_xticks(x); ax.set_xticklabels(moments, rotation=45, ha="right")
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_objective_convergence(self, method_records, out_path):
        if not method_records: return
        methods = [rec["method"] for rec in method_records]
        stage1 = [_safe_float(rec.get("stage1_objective"), np.nan) for rec in method_records]
        stage2 = [_safe_float(rec.get("stage2_objective"), np.nan) for rec in method_records]
        x = np.arange(len(methods), dtype=float)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(x, stage1, marker="o", linewidth=1.5, label="stage 1")
        ax.plot(x, stage2, marker="s", linewidth=1.5, label="stage 2")
        ax.set_title("Risky-debt objective comparison across stages")
        ax.set_xlabel("method"); ax.set_ylabel("objective")
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.grid(True, alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_final_comparison(self, method_records, out_path):
        rows = self._final_comparison_rows(method_records)
        if not rows: return
        methods = [row["method"] for row in rows]
        x = np.arange(len(methods), dtype=float)
        recovery = [_safe_float(row["recovery_score"], np.nan) for row in rows]
        fit = [_safe_float(row["fit_score"], np.nan) for row in rows]
        width = 0.38
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(x - 0.5 * width, recovery, width=width, label="recovery score")
        ax.bar(x + 0.5 * width, fit, width=width, label="fit score")
        ax.set_title("Risky-debt final frequentist comparison")
        ax.set_xlabel("method"); ax.set_ylabel("score / error")
        ax.set_xticks(x); ax.set_xticklabels(methods)
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_pricing_default_fit(self, method_records, out_path):
        rows = self._pricing_default_rows(method_records)
        if not rows: return
        metrics = ["mean_spread", "default_rate", "mean_recovery_default", "mean_abs_zero_profit_residual"]
        methods = sorted({row["method"] for row in rows})
        x = np.arange(len(metrics), dtype=float)
        width = 0.8 / max(1, len(methods))
        fig, ax = plt.subplots(figsize=(9.0, 4.8))
        for j, method in enumerate(methods):
            vals = []
            for metric in metrics:
                row = next(r for r in rows if r["method"] == method and r["metric"] == metric)
                vals.append(abs(_safe_float(row["error"], np.nan)))
            ax.bar(x + (j - (len(methods) - 1) / 2.0) * width, vals, width=width, label=method)
        ax.set_title("Risky-debt pricing/default fit errors")
        ax.set_xlabel("metric"); ax.set_ylabel("absolute error")
        ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _write_bayesian_reports(self, *, bayes_res: Mapping[str, object]) -> Dict[str, object]:
        posterior_rows = self._bayes_posterior_rows(bayes_res)
        diagnostic_rows = self._bayes_diagnostic_rows(bayes_res)
        self._write_table_pair(stem="table_bayes_posterior_summary", rows=posterior_rows, fieldnames=["parameter", "true_value", "posterior_mean", "posterior_median", "posterior_std", "p05", "p95", "abs_error"], caption="Risky-debt Bayesian posterior summary.", label="tab:risky_bayes_posterior_summary")
        self._write_table_pair(stem="table_bayes_diagnostics", rows=diagnostic_rows, fieldnames=["metric", "value"], caption="Risky-debt Bayesian sampling diagnostics.", label="tab:risky_bayes_diagnostics")
        self._plot_bayes_posterior_recovery(bayes_res, os.path.join(self.out_dir, "plot_bayes_posterior_recovery.png"))
        self._plot_bayes_interval_summary(bayes_res, os.path.join(self.out_dir, "plot_bayes_interval_summary.png"))
        bayes_summary_path = os.path.join(self.out_dir, "bayes_report_summary.json")
        with open(bayes_summary_path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable({"posterior_rows": posterior_rows, "diagnostic_rows": diagnostic_rows, "kernel": bayes_res.get("kernel"), "sampler": bayes_res.get("sampler"), "filter": bayes_res.get("filter")}), f, indent=2)
        return {"table_bayes_posterior_summary_csv": os.path.join(self.out_dir, "table_bayes_posterior_summary.csv"), "table_bayes_posterior_summary_tex": os.path.join(self.out_dir, "table_bayes_posterior_summary.tex"), "table_bayes_diagnostics_csv": os.path.join(self.out_dir, "table_bayes_diagnostics.csv"), "table_bayes_diagnostics_tex": os.path.join(self.out_dir, "table_bayes_diagnostics.tex"), "plot_bayes_posterior_recovery_png": os.path.join(self.out_dir, "plot_bayes_posterior_recovery.png"), "plot_bayes_interval_summary_png": os.path.join(self.out_dir, "plot_bayes_interval_summary.png"), "bayes_report_summary_json": bayes_summary_path}

    def _bayes_posterior_rows(self, bayes_res):
        rows = []
        summary = bayes_res.get("posterior_summary", {})
        truth = bayes_res.get("true_parameters", {})
        abs_error = bayes_res.get("absolute_error", {})
        for param in ["theta", "psi0", "alpha"]:
            stats = summary.get(param, {})
            rows.append({"parameter": param, "true_value": truth.get(param), "posterior_mean": stats.get("mean"), "posterior_median": stats.get("median"), "posterior_std": stats.get("std"), "p05": stats.get("p05"), "p95": stats.get("p95"), "abs_error": abs_error.get(param)})
        return rows

    def _bayes_diagnostic_rows(self, bayes_res):
        diag = bayes_res.get("diagnostics", {})
        usage = bayes_res.get("observation_usage", {})
        return [
            {"metric": "kernel", "value": bayes_res.get("kernel")},
            {"metric": "sampler", "value": bayes_res.get("sampler")},
            {"metric": "filter", "value": bayes_res.get("filter")},
            {"metric": "acceptance_rate", "value": diag.get("acceptance_rate")},
            {"metric": "target_log_prob_mean", "value": diag.get("target_log_prob_mean")},
            {"metric": "target_log_prob_last", "value": diag.get("target_log_prob_last")},
            {"metric": "ess_theta", "value": diag.get("effective_sample_size", {}).get("theta")},
            {"metric": "ess_psi0", "value": diag.get("effective_sample_size", {}).get("psi0")},
            {"metric": "ess_alpha", "value": diag.get("effective_sample_size", {}).get("alpha")},
            {"metric": "rhat_theta", "value": diag.get("rhat", {}).get("theta")},
            {"metric": "rhat_psi0", "value": diag.get("rhat", {}).get("psi0")},
            {"metric": "rhat_alpha", "value": diag.get("rhat", {}).get("alpha")},
            {"metric": "log_likelihood_at_posterior_mean", "value": bayes_res.get("log_likelihood_at_posterior_mean")},
            {"metric": "num_paths_used", "value": usage.get("num_paths_used")},
            {"metric": "observations_per_path", "value": usage.get("observations_per_path")},
            {"metric": "total_observations_used", "value": usage.get("total_observations_used")},
        ]

    def _plot_bayes_posterior_recovery(self, bayes_res, out_path):
        rows = self._bayes_posterior_rows(bayes_res)
        params = [row["parameter"] for row in rows]
        means = [_safe_float(row["posterior_mean"], np.nan) for row in rows]
        truth = [_safe_float(row["true_value"], np.nan) for row in rows]
        x = np.arange(len(params), dtype=float); width = 0.38
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.bar(x - 0.5 * width, truth, width=width, label="true")
        ax.bar(x + 0.5 * width, means, width=width, label="posterior mean")
        ax.set_title("Risky-debt Bayesian posterior recovery")
        ax.set_xlabel("parameter"); ax.set_ylabel("value")
        ax.set_xticks(x); ax.set_xticklabels(params)
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _plot_bayes_interval_summary(self, bayes_res, out_path):
        rows = self._bayes_posterior_rows(bayes_res)
        params = [row["parameter"] for row in rows]
        means = np.asarray([_safe_float(row["posterior_mean"], np.nan) for row in rows], dtype=float)
        lo = np.asarray([_safe_float(row["p05"], np.nan) for row in rows], dtype=float)
        hi = np.asarray([_safe_float(row["p95"], np.nan) for row in rows], dtype=float)
        truth = np.asarray([_safe_float(row["true_value"], np.nan) for row in rows], dtype=float)
        x = np.arange(len(params), dtype=float); yerr = np.vstack([means - lo, hi - means])
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=4, linewidth=1.5, label="posterior mean and 90% interval")
        ax.plot(x, truth, marker="s", linestyle="None", label="true value")
        ax.set_title("Risky-debt Bayesian posterior intervals")
        ax.set_xlabel("parameter"); ax.set_ylabel("value")
        ax.set_xticks(x); ax.set_xticklabels(params)
        ax.grid(True, axis="y", alpha=0.25); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

    def _write_combined_summary(self, *, gmm_res: Optional[Mapping[str, object]], smm_res: Optional[Mapping[str, object]], bayes_res: Optional[Mapping[str, object]]) -> str:
        payload = {"has_frequentist": bool(gmm_res and smm_res), "has_bayesian": bool(bayes_res), "dgp_source": (smm_res or gmm_res or {}).get("dgp_source"), "continuation_horizon": (smm_res or gmm_res or {}).get("continuation_horizon"), "frequentist_best": {"gmm": gmm_res.get("best_variant") if gmm_res else None, "smm": smm_res.get("best_variant") if smm_res else None}, "bayesian_kernel": bayes_res.get("kernel") if bayes_res else None, "bayesian_sampler": bayes_res.get("sampler") if bayes_res else None}
        path = os.path.join(self.out_dir, "combined_estimation_summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(make_json_serializable(payload), f, indent=2)
        return path
