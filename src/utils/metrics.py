from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_image_auroc(labels: list[int], scores: list[float]) -> float:
    if len(set(labels)) < 2:
        raise ValueError("AUROC requires at least one normal and one anomalous sample.")
    return float(roc_auc_score(labels, scores))


def normalize_image_scores(labels: list[int], scores: list[float]) -> tuple[list[float], float, float]:
    good_scores = [score for label, score in zip(labels, scores) if label == 0]
    defect_scores = [score for label, score in zip(labels, scores) if label == 1]

    if not good_scores or not defect_scores:
        raise ValueError("Normalization requires both good and anomalous samples.")

    min_good_score = min(good_scores)
    max_defect_score = max(defect_scores)

    if max_defect_score <= min_good_score:
        raise ValueError("Invalid normalization range: max defect score must be larger than min good score.")

    normalized_scores = [
        min(1.0, max(0.0, (score - min_good_score) / (max_defect_score - min_good_score)))
        for score in scores
    ]
    return normalized_scores, float(min_good_score), float(max_defect_score)


def compute_pixel_auroc(masks: list[np.ndarray], anomaly_maps: list[np.ndarray]) -> float:
    if len(masks) != len(anomaly_maps):
        raise ValueError("Masks and anomaly maps must have the same number of samples.")

    flat_masks = np.concatenate([mask.reshape(-1) for mask in masks], axis=0)
    flat_scores = np.concatenate([anomaly_map.reshape(-1) for anomaly_map in anomaly_maps], axis=0)

    if len(np.unique(flat_masks)) < 2:
        raise ValueError("Pixel AUROC requires both normal and anomalous pixels.")

    return float(roc_auc_score(flat_masks, flat_scores))
