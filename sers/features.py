from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score


def _safe_cosine(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if np.allclose(a, 0) or np.allclose(b, 0):
        return None
    return float(1.0 - cosine(a, b))


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(pearsonr(a, b)[0])


def compute_input_features(x_mix: np.ndarray, concentrations: np.ndarray, backgrounds: List[np.ndarray]) -> Dict[str, Optional[float]]:
    # `concentrations` is the historical field name; semantically these are
    # ordered or relative mixture levels for the task.
    flattened_mix = x_mix.reshape(-1)
    pairwise_corrs = []
    for idx in range(len(x_mix) - 1):
        pairwise_corrs.append(_safe_pearson(x_mix[idx], x_mix[idx + 1]))

    bg_corrs = []
    for spectrum in x_mix:
        for background in backgrounds:
            bg_corrs.append(_safe_cosine(spectrum, background))

    return {
        "num_concentrations": float(len(concentrations)),
        "num_points": float(x_mix.shape[1]),
        "mix_mean": float(np.mean(flattened_mix)),
        "mix_std": float(np.std(flattened_mix)),
        "mix_dynamic_range": float(np.max(flattened_mix) - np.min(flattened_mix)),
        "concentration_min": float(np.min(concentrations)),
        "concentration_max": float(np.max(concentrations)),
        "concentration_span": float(np.max(concentrations) - np.min(concentrations)),
        "adjacent_mix_corr_mean": _mean_or_none(pairwise_corrs),
        "mix_bg_cosine_mean": _mean_or_none(bg_corrs),
    }


def compute_output_features(
    f_final: np.ndarray,
    coeff_target: np.ndarray,
    recon_matrix: np.ndarray,
    x_mix: np.ndarray,
    concentrations: np.ndarray,
    final_loss: float,
) -> Dict[str, Optional[float]]:
    residuals = x_mix - recon_matrix
    smoothness = float(np.mean(np.abs(np.diff(f_final))))
    nonneg_fraction = float(np.mean(f_final >= 0))
    recon_r2 = float(r2_score(x_mix.reshape(-1), recon_matrix.reshape(-1)))
    residual_ratio = float(np.linalg.norm(residuals) / (np.linalg.norm(x_mix) + 1e-12))

    monotonicity = None
    if len(concentrations) > 1 and np.std(coeff_target) > 0:
        # This captures monotonic behavior across ordered levels. It should not
        # be interpreted as a physical concentration law.
        monotonicity = float(spearmanr(concentrations, coeff_target).statistic)

    return {
        "final_loss": float(final_loss),
        "recon_r2": recon_r2,
        "residual_norm_ratio": residual_ratio,
        "f_smoothness_l1": smoothness,
        "f_nonnegative_fraction": nonneg_fraction,
        "target_coef_mean": float(np.mean(coeff_target)),
        "target_coef_std": float(np.std(coeff_target)),
        "target_coef_spearman_vs_conc": monotonicity,
    }


def compute_ground_truth_metrics(gt_f: Optional[np.ndarray], f_final: np.ndarray) -> Dict[str, Optional[float]]:
    if gt_f is None:
        return {"gt_cosine": None, "gt_r2": None, "gt_pearson": None}
    return {
        "gt_cosine": _safe_cosine(gt_f, f_final),
        "gt_r2": float(r2_score(gt_f, f_final)),
        "gt_pearson": _safe_pearson(gt_f, f_final),
    }


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    filtered = [value for value in values if value is not None and not np.isnan(value)]
    if not filtered:
        return None
    return float(np.mean(filtered))
