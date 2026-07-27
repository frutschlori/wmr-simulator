"""Detect logs that disagree with the rest of a recording batch.

Identification is cheap enough to run once per recorded log, which turns the
batch of a single iteration into a small sample of parameter estimates. A log
that was recorded badly (robot bumped, mocap dropout, a run that never left the
start pose) shows up as an estimate far away from the others, and dragging it
into the joint fit biases every downstream stage. The spread is measured
robustly so a bad log cannot hide itself by inflating the scale it is judged
against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# MAD -> standard deviation for normally distributed samples.
_MAD_TO_SIGMA = 1.4826
# Finite-sample correction of that consistency factor (Croux & Rousseeuw 1992);
# the asymptotic 1.4826 underestimates the spread badly for the ~5-10 logs an
# iteration records, which on its own turns ordinary scatter into "outliers".
_MAD_SMALL_SAMPLE_FACTORS = {4: 1.363, 5: 1.206, 6: 1.200, 7: 1.140, 8: 1.129, 9: 1.107}


def _mad_small_sample_factor(num_samples: int) -> float:
    return _MAD_SMALL_SAMPLE_FACTORS.get(num_samples, num_samples / (num_samples - 0.8))


@dataclass(frozen=True)
class ParameterOutlierReport:
    """Per-sample robust deviation from the batch median.

    ``z_scores`` is (num_samples, num_parameters); ``max_z_scores`` reduces it
    over the parameters, and a sample is an outlier when that exceeds the
    threshold. ``median`` is in physical units, ``scale`` is the per-parameter
    robust spread in log space, i.e. a relative one. ``evaluated`` is False when
    the batch was too small (or too degenerate) to judge, in which case nothing
    is flagged.
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
    """Flag parameter samples that sit far from the per-parameter median.

    Works in log space: the physical parameters are strictly positive and are
    identified in log-relative coordinates, so a relative deviation is the
    meaningful one and the five parameters become comparable despite their very
    different units. The scale is the median absolute deviation rescaled to a
    standard deviation (Iglewicz & Hoaglin's modified z-score, whose usual
    threshold is 3.5); unlike a plain standard deviation it does not grow with
    the very samples it is supposed to expose.

    Two corrections keep the small batches an iteration records from producing
    outliers out of thin air. The consistency factor gets its finite-sample
    correction, and the scale is floored at ``min_relative_spread`` (a relative
    deviation, since the comparison is in log space): a parameter that happens
    to come out near-identical on two logs otherwise shrinks the scale toward
    zero and makes ordinary scatter on the remaining logs look enormous. With
    the default floor, a batch that agrees to better than 1% still needs a
    ``z_threshold``-times-1% relative deviation before a log is dropped.

    A zero-spread parameter (e.g. a disabled ``a_slip_max``, identical in every
    run) carries no information and is skipped rather than producing infinite
    scores. ``z_threshold <= 0`` disables detection, as does a sample count
    below ``min_samples`` -- with two or three samples the median sits on (or
    between) the candidates themselves and every sample scores alike.
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
    # A parameter that is identical on every log (a disabled a_slip_max) is
    # skipped. Judging that on the maximum rather than the median deviation
    # matters: a single log breaking out of an otherwise identical column has a
    # zero MAD, and must still be scored (against the floor) instead of ignored.
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
