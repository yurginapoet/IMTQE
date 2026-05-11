"""
Преобразование объяснений в «доли потери» относительно идеальной оценки 1.0.

Используется для UI: показываем только то, что тянет оценку вниз (или распределяем
разрыв (1 − score) по attention нейронной головы), отбрасываем пренебрежимо малые
доли и нормируем оставшееся к сумме 1.0.
"""

from __future__ import annotations

from typing import Mapping


def shap_categories_to_loss_shares(
    expl: Mapping[str, float],
    min_share: float = 0.005,
) -> dict[str, float]:
    """
    SHAP по MQM-категориям: отрицательный вклад = снижение score.
    Берём max(0, −v), нормируем, отбрасываем категории с долей < min_share, снова нормируем.

    Ключи на выходе — те же английские имена категорий (Accuracy, …), что и во входе.
    """
    losses: dict[str, float] = {}
    for k, v in expl.items():
        lv = max(0.0, -float(v))
        if lv > 0.0:
            losses[str(k)] = lv
    total = sum(losses.values())
    if total < 1e-12:
        return {}
    shares = {k: v / total for k, v in losses.items()}
    filtered = {k: v for k, v in shares.items() if v >= float(min_share)}
    t2 = sum(filtered.values())
    if t2 < 1e-12:
        return {}
    return {k: v / t2 for k, v in filtered.items()}
