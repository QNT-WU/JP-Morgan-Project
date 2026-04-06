"""Reporting helpers for estimation tables, charts, and saved artifacts."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import csv
import json
import os
from typing import Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .common import PARAMETER_NAMES

METHOD_ORDER = ["GMM_A", "GMM_B", "SMM_A", "SMM_B"]


def _ensure_dir(path: str) -> None:
    """Create ``path`` when needed before writing estimation artifacts."""
    os.makedirs(path, exist_ok=True)


def _json_safe(obj):
    """Recursively convert NumPy and dataclass objects into JSON-safe values."""
    if is_dataclass(obj):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_json(data, path: str) -> None:
    """Write a JSON artifact to disk."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_json_safe(data), f, indent=2, sort_keys=True)


def write_csv(rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Write a list of dictionaries as a CSV table."""
    rows = list(rows)
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def write_latex_table(rows: Sequence[Mapping[str, object]], path: str, caption: str = '', label: str = '') -> None:
    """Write tabular rows as a simple LaTeX table."""
    rows = list(rows)
    if not rows:
        return
    columns: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in columns:
                columns.append(str(key))
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\\begin{table}[htbp]\n\\centering\n')
        if caption:
            f.write(f'\\caption{{{caption}}}\n')
        if label:
            f.write(f'\\label{{{label}}}\n')
        f.write('\\begin{tabular}{' + 'l' * len(columns) + '}\n')
        f.write('\\hline\n')
        f.write(' & '.join(columns) + ' \\\\ \n')
        f.write('\\hline\n')
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, '')
                vals.append(f'{val:.6g}' if isinstance(val, float) else str(val))
            f.write(' & '.join(vals) + ' \\\\ \n')
        f.write('\\hline\n\\end{tabular}\n\\end{table}\n')


def _get_stage_best_losses(res) -> tuple[float, float]:
    """Extract the best loss reached at each optimization stage, if available."""
    return float(res.stage1.best_loss), float(res.stage2.best_loss)


def _get_stage2_losses(res) -> List[float]:
    """Extract stage-two loss values from per-start optimization records."""
    return [float(start.final_loss) for start in res.stage2.starts]


def make_raw_result_rows(results: Mapping[str, object]) -> List[Dict[str, object]]:
    """Create raw result rows for reporting tables."""
    return [results[m].to_flat_dict() for m in METHOD_ORDER if m in results]


def make_start_result_rows(results: Mapping[str, object]) -> List[Dict[str, object]]:
    """Create per-start optimization rows for reporting tables."""
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        if method not in results:
            continue
        res = results[method]
        for stage_summary in [res.stage1, res.stage2]:
            for start in stage_summary.starts:
                rows.append({
                    'method': method,
                    'stage': stage_summary.stage,
                    'start_id': start.start_id,
                    'is_best': bool(start.is_best),
                    'success': bool(start.success),
                    'status': int(start.status),
                    'message': str(start.message),
                    'nfev': int(start.nfev),
                    'nit': int(start.nit),
                    'final_loss': float(start.final_loss),
                    'elapsed_seconds': float(start.elapsed_seconds),
                    'x0': json.dumps(start.x0),
                    'x_hat': json.dumps(start.x_hat),
                })
    return rows


def make_parameter_recovery_rows(results: Mapping[str, object], true_params: Mapping[str, float]) -> List[Dict[str, object]]:
    """Create parameter-recovery rows against the true parameters."""
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        if method not in results:
            continue
        res = results[method]
        est_vec = np.asarray([res.final_params[name] for name in PARAMETER_NAMES], dtype=np.float64)
        true_vec = np.asarray([true_params[name] for name in PARAMETER_NAMES], dtype=np.float64)
        recovery_score = float(np.linalg.norm(est_vec - true_vec))
        for name in PARAMETER_NAMES:
            est = float(res.final_params[name])
            true = float(true_params[name])
            err = est - true
            rows.append({
                'method': method,
                'parameter': name,
                'true_value': true,
                'estimate': est,
                'abs_error': abs(err),
                'rel_error': abs(err) / max(abs(true), 1e-12),
                'recovery_score_l2': recovery_score,
                'final_success': bool(res.stage2.success),
            })
    return rows


def make_gmm_moment_fit_rows(results: Mapping[str, object]) -> List[Dict[str, object]]:
    """Create GMM moment-fit summary rows."""
    rows: List[Dict[str, object]] = []
    for method in ['GMM_A', 'GMM_B']:
        if method not in results:
            continue
        res = results[method]
        g = np.asarray([res.moment_vector[str(i)] for i in sorted(int(k) for k in res.moment_vector.keys())], dtype=np.float64)
        stage1_loss, stage2_loss = _get_stage_best_losses(res)
        rows.append({
            'method': method,
            'g_norm_l2': float(np.linalg.norm(g)),
            'g_max_abs': float(np.max(np.abs(g))),
            'stage1_objective': stage1_loss,
            'stage2_objective': stage2_loss,
            'final_success': bool(res.stage2.success),
        })
    return rows


def make_smm_moment_fit_rows(results: Mapping[str, object]) -> List[Dict[str, object]]:
    """Create SMM moment-fit summary rows."""
    rows: List[Dict[str, object]] = []
    for method in ['SMM_A', 'SMM_B']:
        if method not in results:
            continue
        res = results[method]
        moment_names = list(res.observed_moments.keys())
        if hasattr(res.covariance_info, 'covariance') and res.covariance_info.covariance is not None:
            var_diag = np.maximum(np.diag(np.asarray(res.covariance_info.covariance, dtype=np.float64)), 1e-12)
        else:
            var_diag = np.ones(len(moment_names), dtype=np.float64)
        for i, name in enumerate(moment_names):
            obs = float(res.observed_moments[name])
            sim = float(res.simulated_moments[name])
            gap = float(res.moment_gaps[name])
            standardized = float(gap / np.sqrt(var_diag[i]))
            rows.append({
                'method': method,
                'moment': name,
                'observed': obs,
                'simulated': sim,
                'raw_error': gap,
                'percent_error': gap / max(abs(obs), 1e-12),
                'standardized_error': standardized,
                'final_success': bool(res.stage2.success),
            })
    return rows


def make_computation_convergence_rows(results: Mapping[str, object]) -> List[Dict[str, object]]:
    """Create computation and convergence summary rows."""
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        if method not in results:
            continue
        res = results[method]
        stage2_losses = np.asarray(_get_stage2_losses(res), dtype=np.float64)
        rows.append({
            'method': method,
            'runtime_seconds': float(res.elapsed_seconds),
            'stage1_objective': float(res.stage1.best_loss),
            'stage2_objective': float(res.stage2.best_loss),
            'stage1_success': bool(res.stage1.success),
            'stage2_success': bool(res.stage2.success),
            'final_success': bool(res.stage2.success),
            'n_starts': int(res.stage2.n_starts),
            'winner_start': int(res.stage2.best_start_id),
            'stage2_loss_dispersion_std': float(np.std(stage2_losses)) if len(stage2_losses) else 0.0,
            'stage2_loss_dispersion_range': float(np.max(stage2_losses) - np.min(stage2_losses)) if len(stage2_losses) else 0.0,
            'stage2_message': str(res.stage2.message),
            'stage2_nfev': int(res.stage2.nfev),
            'condition_number': float(res.covariance_info.condition_number),
        })
    return rows


def make_final_comparison_rows(results: Mapping[str, object], true_params: Mapping[str, float]) -> List[Dict[str, object]]:
    """Create the final cross-method comparison rows."""
    rows: List[Dict[str, object]] = []
    true_vec = np.asarray([true_params[name] for name in PARAMETER_NAMES], dtype=np.float64)
    temp = []
    for method in METHOD_ORDER:
        if method not in results:
            continue
        res = results[method]
        est_vec = np.asarray([res.final_params[name] for name in PARAMETER_NAMES], dtype=np.float64)
        recovery_score = float(np.linalg.norm(est_vec - true_vec))
        if method.startswith('GMM'):
            g = np.asarray([res.moment_vector[str(i)] for i in sorted(int(k) for k in res.moment_vector.keys())], dtype=np.float64)
            fit_score = float(np.linalg.norm(g))
            fit_metric = 'g_norm_l2'
        else:
            gaps = []
            var_diag = np.maximum(np.diag(np.asarray(res.covariance_info.covariance, dtype=np.float64)), 1e-12)
            for i, name in enumerate(res.observed_moments.keys()):
                gaps.append(float(res.moment_gaps[name]) / np.sqrt(var_diag[i]))
            fit_score = float(np.sqrt(np.mean(np.square(np.asarray(gaps, dtype=np.float64)))))
            fit_metric = 'std_gap_rmse'
        temp.append((method, recovery_score, fit_score, float(res.elapsed_seconds), bool(res.stage2.success), fit_metric))
    # overall ranking: converged first, then lower recovery, then lower fit, then lower runtime
    sorted_methods = sorted(temp, key=lambda x: (not x[4], x[1], x[2], x[3]))
    rank_map = {m[0]: i + 1 for i, m in enumerate(sorted_methods)}
    for method, recovery_score, fit_score, runtime_s, final_success, fit_metric in temp:
        rows.append({
            'method': method,
            'recovery_score': recovery_score,
            'fit_score': fit_score,
            'fit_metric': fit_metric,
            'runtime_seconds': runtime_s,
            'final_success': final_success,
            'overall_ranking': rank_map[method],
        })
    rows.sort(key=lambda r: METHOD_ORDER.index(r['method']))
    return rows


def plot_parameter_estimates(results: Mapping[str, object], true_params: Mapping[str, float], out_path: str) -> None:
    """Plot true and estimated parameters across methods."""
    methods = [m for m in METHOD_ORDER if m in results]
    if not methods:
        return
    x = np.arange(len(PARAMETER_NAMES))
    plt.figure(figsize=(8, 4.5))
    for method in methods:
        y = [results[method].final_params[name] for name in PARAMETER_NAMES]
        plt.plot(x, y, marker='o', label=method)
    plt.plot(x, [true_params[name] for name in PARAMETER_NAMES], marker='x', linewidth=2.0, label='Truth')
    plt.xticks(x, PARAMETER_NAMES)
    plt.ylabel('Estimate')
    plt.title('Parameter recovery (stage 2 estimates)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_smm_moment_fit(results: Mapping[str, object], out_path: str) -> None:
    """Plot observed and simulated SMM moments."""
    methods = [m for m in ['SMM_A', 'SMM_B'] if m in results]
    if not methods:
        return
    moment_names = list(results[methods[0]].observed_moments.keys())
    x = np.arange(len(moment_names))
    width = 0.35
    plt.figure(figsize=(12, 4.8))
    obs = [results[methods[0]].observed_moments[m] for m in moment_names]
    plt.bar(x - width/2, obs, width=width, label='Observed')
    for j, method in enumerate(methods):
        sim = [results[method].simulated_moments[m] for m in moment_names]
        plt.bar(x + (j * width/2), sim, width=width/2, label=method)
    plt.xticks(x, moment_names, rotation=45, ha='right')
    plt.ylabel('Moment value')
    plt.title('Observed versus simulated SMM moments')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_smm_standardized_errors(results: Mapping[str, object], out_path: str) -> None:
    """Plot standardized SMM moment errors."""
    methods = [m for m in ['SMM_A', 'SMM_B'] if m in results]
    if not methods:
        return
    moment_names = list(results[methods[0]].observed_moments.keys())
    x = np.arange(len(moment_names))
    width = 0.35
    plt.figure(figsize=(12, 4.8))
    for j, method in enumerate(methods):
        res = results[method]
        var_diag = np.maximum(np.diag(np.asarray(res.covariance_info.covariance, dtype=np.float64)), 1e-12)
        errs = [float(res.moment_gaps[name]) / np.sqrt(var_diag[i]) for i, name in enumerate(moment_names)]
        plt.bar(x + (j - 0.5) * width, errs, width=width, label=method)
    plt.axhline(0.0, linewidth=1.0)
    plt.xticks(x, moment_names, rotation=45, ha='right')
    plt.ylabel('Standardized error')
    plt.title('Standardized SMM moment errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_objective_convergence(results: Mapping[str, object], out_path: str) -> None:
    """Plot optimization objective diagnostics."""
    methods = [m for m in METHOD_ORDER if m in results]
    if not methods:
        return
    x = np.arange(len(methods))
    stage1 = [float(results[m].stage1.best_loss) for m in methods]
    stage2 = [float(results[m].stage2.best_loss) for m in methods]
    width = 0.35
    plt.figure(figsize=(8, 4.5))
    plt.bar(x - width/2, stage1, width=width, label='Stage 1')
    plt.bar(x + width/2, stage2, width=width, label='Stage 2')
    plt.xticks(x, methods)
    plt.ylabel('Best objective')
    plt.title('Objective values across stages')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_method_scores(final_rows: Sequence[Mapping[str, object]], out_path: str) -> None:
    """Plot cross-method summary scores."""
    rows = list(final_rows)
    if not rows:
        return
    methods = [str(r['method']) for r in rows]
    scores = [float(r['recovery_score']) for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.bar(methods, scores)
    plt.ylabel('Recovery score')
    plt.title('Final GMM-versus-SMM comparison')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_policy_distance(policy_rows: Sequence[Mapping[str, object]], out_path: str) -> None:
    """Plot policy distance relative to the synthetic truth policy."""
    rows = list(policy_rows)
    if not rows:
        return
    methods = [str(r['method']) for r in rows]
    values = [float(r['policy_mse_vs_truth']) for r in rows]
    plt.figure(figsize=(7, 4.5))
    plt.bar(methods, values)
    plt.ylabel('Policy MSE versus truth')
    plt.title('Policy comparison diagnostic')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_estimation_report(results: Mapping[str, object], true_params: Mapping[str, float], out_dir: str) -> None:
    """Save the full estimation report bundle to disk."""
    _ensure_dir(out_dir)
    save_json(results, os.path.join(out_dir, 'estimation_results.json'))
    raw_rows = make_raw_result_rows(results)
    start_rows = make_start_result_rows(results)
    parameter_rows = make_parameter_recovery_rows(results, true_params)
    gmm_fit_rows = make_gmm_moment_fit_rows(results)
    smm_fit_rows = make_smm_moment_fit_rows(results)
    computation_rows = make_computation_convergence_rows(results)
    final_rows = make_final_comparison_rows(results, true_params)

    write_csv(raw_rows, os.path.join(out_dir, 'raw_results.csv'))
    write_csv(start_rows, os.path.join(out_dir, 'raw_start_results.csv'))
    write_csv(parameter_rows, os.path.join(out_dir, 'table_parameter_recovery.csv'))
    write_csv(gmm_fit_rows, os.path.join(out_dir, 'table_gmm_moment_fit.csv'))
    write_csv(smm_fit_rows, os.path.join(out_dir, 'table_smm_moment_fit.csv'))
    write_csv(computation_rows, os.path.join(out_dir, 'table_computation_convergence.csv'))
    write_csv(final_rows, os.path.join(out_dir, 'table_final_gmm_vs_smm_comparison.csv'))

    write_latex_table(parameter_rows, os.path.join(out_dir, 'table_parameter_recovery.tex'), caption='Parameter recovery by method (single-run stage 2 estimates)', label='tab:param_recovery')
    write_latex_table(gmm_fit_rows, os.path.join(out_dir, 'table_gmm_moment_fit.tex'), caption='GMM moment fit summary', label='tab:gmm_fit')
    write_latex_table(smm_fit_rows, os.path.join(out_dir, 'table_smm_moment_fit.tex'), caption='SMM moment fit summary', label='tab:smm_fit')
    write_latex_table(computation_rows, os.path.join(out_dir, 'table_computation_convergence.tex'), caption='Computation and convergence summary', label='tab:comp_conv')
    write_latex_table(final_rows, os.path.join(out_dir, 'table_final_gmm_vs_smm_comparison.tex'), caption='Final GMM-versus-SMM comparison', label='tab:final_compare')

    plot_parameter_estimates(results, true_params, os.path.join(out_dir, 'plot_parameter_recovery.png'))
    plot_smm_moment_fit(results, os.path.join(out_dir, 'plot_smm_moment_fit.png'))
    plot_smm_standardized_errors(results, os.path.join(out_dir, 'plot_smm_standardized_errors.png'))
    plot_objective_convergence(results, os.path.join(out_dir, 'plot_objective_convergence.png'))
    plot_method_scores(final_rows, os.path.join(out_dir, 'plot_final_comparison.png'))
