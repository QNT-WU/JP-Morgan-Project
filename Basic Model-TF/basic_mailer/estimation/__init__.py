"""Public estimation API for the basic Mailer package."""

from .common import PARAMETER_NAMES, structural_params_from_model, transform_params_to_tilde, transform_tilde_to_params, update_model_params
from .moments import CRNDesign, PathDataset, MomentSpec, build_default_moment_spec, compute_gmm_moment_series, compute_gmm_moment_vector, compute_moments, compute_smm_moment_series, make_crn_design, make_identity_weight_matrix, path_sample_size, simulate_paths_crn, summarize_smm_moments
from .weighting import CovarianceEstimate, choose_newey_west_lags, estimate_weighting_matrix, estimate_standard_covariance, estimate_newey_west_covariance, numerical_jacobian, sandwich_parameter_covariance
from .gmm import TwoStepGMMEstimator, GMMMethodResult
from .smm import TwoStepSMMEstimator, SMMMethodResult, _train_policy_obj2_inner
from .reporting import save_estimation_report, write_csv, write_latex_table, save_json, plot_policy_distance

__all__ = [
    'PARAMETER_NAMES','structural_params_from_model','transform_params_to_tilde','transform_tilde_to_params','update_model_params',
    'CRNDesign','PathDataset','MomentSpec','build_default_moment_spec','compute_gmm_moment_series','compute_gmm_moment_vector','compute_moments','compute_smm_moment_series','make_crn_design','make_identity_weight_matrix','path_sample_size','simulate_paths_crn','summarize_smm_moments',
    'CovarianceEstimate','choose_newey_west_lags','estimate_weighting_matrix','estimate_standard_covariance','estimate_newey_west_covariance','numerical_jacobian','sandwich_parameter_covariance',
    'TwoStepGMMEstimator','GMMMethodResult','TwoStepSMMEstimator','SMMMethodResult','_train_policy_obj2_inner','save_estimation_report','write_csv','write_latex_table','save_json','plot_policy_distance'
]
