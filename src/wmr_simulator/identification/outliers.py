"""Detect identification logs whose parameters disagree with the rest of a batch
(robust modified z-score in log space)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MAD -> standard deviation for normally distributed samples.
_MAD_TO_SIGMA = 1.4826
# Finite-sample correction of that consistency factor (Croux & Rousseeuw 1992).
_MAD_SMALL_SAMPLE_FACTORS = {4: 1.363, 5: 1.206, 6: 1.200, 7: 1.140, 8: 1.129, 9: 1.107}


def _mad_small_sample_factor(num_samples: int) -> float:
    return _MAD_SMALL_SAMPLE_FACTORS.get(num_samples, num_samples / (num_samples - 0.8))


@dataclass(frozen=True)
class ParameterOutlierReport:
    """Per-sample robust deviation from the batch median.

    ``z_scores`` is (num_samples, num_parameters), ``max_z_scores`` its maximum
    over the parameters. ``median`` is in physical units, ``scale`` the robust
    spread in log space. ``evaluated`` is False when the batch was too small to
    judge.
    """

    z_scores: np.ndarray
    max_z_scores: np.ndarray
    is_outlier: np.ndarray
    median: np.ndarray
    scale: np.ndarray
    evaluated: bool


def robust_parameter_outliers(
    parameter_samples: np.ndarray,
    z_threshold: float,
    min_samples: int = 4,
    min_relative_spread: float = 0.01,
) -> ParameterOutlierReport:
    """Flag parameter samples far from the per-parameter median.

    Modified z-score (Iglewicz & Hoaglin) in log space, with the MAD scaled to
    a standard deviation (finite-sample corrected) and floored at
    ``min_relative_spread``. Parameters identical on every sample are skipped.
    ``z_threshold <= 0`` or fewer than ``min_samples`` samples disable the
    detection.
    """
    samples = np.atleast_2d(np.asarray(parameter_samples, dtype=float))
    num_samples, num_parameters = samples.shape
    zeros = np.zeros_like(samples)
    disabled = ParameterOutlierReport(
        z_scores=zeros,
        max_z_scores=np.zeros(num_samples),
        is_outlier=np.zeros(num_samples, dtype=bool),
        median=np.median(samples, axis=0) if num_samples else np.zeros(num_parameters),
        scale=np.zeros(num_parameters),
        evaluated=False,
    )
    if z_threshold <= 0.0 or num_samples < min_samples:
        return disabled

    log_samples = np.log(np.maximum(samples, 1e-12))
    median = np.median(log_samples, axis=0)
    deviations = np.abs(log_samples - median)
    mad = np.median(deviations, axis=0)
    # Skip parameters identical on every log (judged on the maximum deviation,
    # so a single deviating log is still scored).
    informative = np.max(deviations, axis=0) > 1e-9
    scale = np.zeros_like(mad)
    scale[informative] = np.maximum(
        _MAD_TO_SIGMA * _mad_small_sample_factor(num_samples) * mad[informative],
        min_relative_spread,
    )
    if not np.any(informative):
        return disabled

    z_scores = np.zeros_like(log_samples)
    z_scores[:, informative] = deviations[:, informative] / scale[informative]
    max_z_scores = np.max(z_scores, axis=1)
    return ParameterOutlierReport(
        z_scores=z_scores,
        max_z_scores=max_z_scores,
        is_outlier=max_z_scores > z_threshold,
        median=np.exp(median),
        scale=scale,
        evaluated=True,
    )
