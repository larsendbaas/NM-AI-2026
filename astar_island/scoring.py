"""Scoring helpers aligned with the competition metric."""

from __future__ import annotations

import math


def cell_entropy(distribution: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-12)) for value in distribution if value > 0.0)


def cell_kl_divergence(ground_truth: list[float], prediction: list[float]) -> float:
    return sum(
        truth * math.log(truth / max(prediction[index], 1e-12))
        for index, truth in enumerate(ground_truth)
        if truth > 0.0
    )


def weighted_kl(ground_truth: list[list[list[float]]], prediction: list[list[list[float]]]) -> float:
    entropy_total = 0.0
    weighted_total = 0.0
    for y, row in enumerate(ground_truth):
        for x, cell_truth in enumerate(row):
            entropy = cell_entropy(cell_truth)
            if entropy <= 1e-12:
                continue
            entropy_total += entropy
            weighted_total += entropy * cell_kl_divergence(cell_truth, prediction[y][x])
    if entropy_total <= 1e-12:
        return 0.0
    return weighted_total / entropy_total


def score_prediction(ground_truth: list[list[list[float]]], prediction: list[list[list[float]]]) -> float:
    kl_value = weighted_kl(ground_truth, prediction)
    return max(0.0, min(100.0, 100.0 * math.exp(-3.0 * kl_value)))


def score_round(
    ground_truths: list[list[list[list[float]]]],
    predictions: list[list[list[list[float]]]],
) -> float:
    if not ground_truths:
        return 0.0
    scores = [score_prediction(ground_truth, prediction) for ground_truth, prediction in zip(ground_truths, predictions)]
    return sum(scores) / len(scores)
