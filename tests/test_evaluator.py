import sys
import os
import pandas as pd

# Add project root/src to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from agents.evaluator import EvaluatorAgent


def test_evaluator_confidence_scoring():
    df = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=5),
        "spend": [100, 120, 150, 200, 250],
        "revenue": [200, 190, 170, 160, 140],
        "impressions": [10000, 11000, 12000, 13000, 14000],
        "clicks": [100, 90, 80, 70, 60],
        "campaign_name": ["Test_Campaign"] * 5
    })

    config = {"confidence_min": 0.5}

    evaluator = EvaluatorAgent(df, config)

    candidates = [{
    "id": "roas_spend_negative",
    "metric": "roas",
    "segment": {"campaign_name": "Test"},
    "baseline": 2.0,
    "current": 1.2,
    "delta_abs": -0.8,
    "delta_rel": -0.40,
    "sample_size": 200,
    "impact": "medium",
    "impact_score": 0.65,
    "confidence": 0.75,
    "evidence": {"correlation": -0.5}
    }]


    results = evaluator.validate(candidates)

    assert len(results) == 1
    result = results[0]

    assert "valid" in result
    assert "reasons" in result
    assert isinstance(result["reasons"], list)

