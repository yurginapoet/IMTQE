import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.interpretation.explanation_loss import shap_categories_to_loss_shares


def test_shap_categories_to_loss_shares_filters_and_renorms():
    expl = {
        "Accuracy": -0.6,
        "Fluency": -0.2,
        "Locale": -0.001,
        "Semantic": 0.1,
    }
    out = shap_categories_to_loss_shares(expl, loss_budget=0.27, min_share=0.005)
    assert set(out.keys()) == {"Accuracy", "Fluency"}
    assert abs(sum(out.values()) - 0.27) < 1e-6
    assert out["Accuracy"] == pytest.approx(0.2025)
    assert out["Fluency"] == pytest.approx(0.0675)


def test_shap_categories_all_positive_returns_empty():
    assert shap_categories_to_loss_shares({"Accuracy": 0.5}) == {}
