"""
src/interpretation/explanation_loss.py

Преобразование SHAP объяснений в «доли потери» относительно идеальной оценки 1.0.
"""

from __future__ import annotations

from typing import Mapping


def shap_categories_to_loss_shares(
    expl: Mapping[str, float],
    loss_budget: float = 1.0,
    min_share: float = 0.005,
) -> dict[str, float]:
    """
    Преобразует SHAP-вклады в доли потери качества.
    Только отрицательные вклады (то, что тянет score вниз).
    """
    losses: dict[str, float] = {}
    for k, v in expl.items():
        loss = max(0.0, -float(v))
        if loss > 0.0:
            losses[str(k)] = loss

    total_loss = sum(losses.values())
    if total_loss < 1e-8:
        return {}

    # Нормализация
    shares = {k: v / total_loss for k, v in losses.items()}

    # Отбрасываем очень маленькие доли
    filtered = {k: v for k, v in shares.items() if v >= float(min_share)}

    total_filtered = sum(filtered.values())
    if total_filtered < 1e-8:
        return {}

    # Финальная нормализация и масштабирование к "недостающему до 100%" бюджету
    budget = max(0.0, float(loss_budget))
    return {k: (v / total_filtered) * budget for k, v in filtered.items()}
