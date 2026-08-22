"""Small Autobahn-aware filters used by the rule-guided CMAB policy."""

from __future__ import annotations

from math import isfinite
from typing import Iterable, Sequence


# The first pass intentionally keeps all rule settings in one place.  The
# CloudLab interface exposes only one enable/disable flag.
STRUCTURED_INIT_EPOCHS = 8
PROTOCOL_METRIC_WINDOW = 3
CANDIDATE_CONFIRMATIONS = 2
NO_IMPROVEMENT_EPOCHS = 10

REWARD_IMPROVEMENT_THRESHOLD = 0.05
REWARD_ROLLBACK_THRESHOLD = 0.30

FPR_HIGH_THRESHOLD = 0.90
FPR_LOW_THRESHOLD = 0.20
LANE_HEALTHY_RATIO = 0.60
LANE_MIN_ACTIVITY = 0.01


def _finite_recent(values: Iterable[float], limit: int) -> list[float]:
    recent = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if isfinite(number):
            recent.append(number)
    return recent[-limit:]


def allowed_timeout_values(
    current_timeout: int,
    recent_fpr: Iterable[float],
    available_values: Iterable[int],
) -> tuple[set[int], float | None]:
    """Return legal timeout values and the recent mean fast-path ratio."""

    available = {int(value) for value in available_values}
    fpr_values = _finite_recent(recent_fpr, PROTOCOL_METRIC_WINDOW)
    mean_fpr = (
        sum(fpr_values) / len(fpr_values)
        if fpr_values
        else None
    )

    current_timeout = int(current_timeout)
    if current_timeout == 0:
        requested = {0, 100}
    elif current_timeout == 100:
        requested = (
            {0, 100}
            if mean_fpr is not None and mean_fpr >= FPR_HIGH_THRESHOLD
            else {0, 100, 300}
        )
    elif current_timeout == 300:
        if mean_fpr is not None and mean_fpr < FPR_LOW_THRESHOLD:
            requested = {0}
        elif mean_fpr is not None and mean_fpr >= FPR_HIGH_THRESHOLD:
            requested = {100, 300}
        else:
            requested = {0, 100, 300}
    else:
        requested = set(available)

    allowed = requested & available
    if not allowed:
        allowed = ({current_timeout} & available) or set(available)
    return allowed, mean_fpr


def allowed_cut_values(
    recent_lane_growth: Sequence[Sequence[float]],
    available_values: Iterable[int],
) -> tuple[set[int], list[float], int | None]:
    """Return legal cut thresholds from recent raw per-lane growth rates."""

    available = {int(value) for value in available_values}
    valid_samples: list[list[float]] = []
    for sample in recent_lane_growth[-PROTOCOL_METRIC_WINDOW:]:
        values = _finite_recent(sample, len(sample))
        if len(values) >= 2:
            valid_samples.append(values)

    if not valid_samples:
        return set(available), [], None

    lane_count = min(len(sample) for sample in valid_samples)
    averaged = [
        sum(sample[index] for sample in valid_samples) / len(valid_samples)
        for index in range(lane_count)
    ]
    ordered = sorted(averaged, reverse=True)
    second_fastest = ordered[1]

    if second_fastest <= LANE_MIN_ACTIVITY:
        healthy_lanes = min(available) if available else None
    else:
        threshold = LANE_HEALTHY_RATIO * second_fastest
        healthy_lanes = sum(value >= threshold for value in ordered)

    if healthy_lanes is None:
        allowed = set(available)
    else:
        allowed = {value for value in available if value <= healthy_lanes}
        if not allowed and available:
            allowed = {min(available)}
    return allowed, ordered, healthy_lanes
