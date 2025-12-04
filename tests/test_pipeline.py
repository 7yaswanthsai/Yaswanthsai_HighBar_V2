import sys
import os
import pandas as pd

# Add src to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from agents.data_agent import DataAgent
from agents.insight_agent import InsightAgent
from agents.evaluator import EvaluatorAgent
from agents.creative_generator import CreativeGenerator


def test_full_pipeline_minimal():
    # small synthetic dataset
    df = pd.DataFrame({
        "campaign_name": ["A", "A", "B", "B"],
        "adset_name": ["x","x","y","y"],
        "date": pd.date_range("2025-01-01", periods=4),
        "spend": [100, 150, 120, 110],
        "impressions": [10000, 9000, 8000, 7500],
        "clicks": [100, 60, 90, 55],
        "ctr": [0.01, 0.006, 0.011, 0.007],
        "purchases": [2, 1, 3, 2],
        "revenue": [150, 120, 220, 180],
        "roas": [1.5, 0.8, 1.8, 1.6],
        "creative_type": ["img","img","video","video"],
        "creative_message": ["Sale now", "Limited offer", "New launch", "Comfort fit"],
        "audience_type": ["broad","broad","retargeting","retargeting"],
        "platform": ["fb","fb","ig","ig"],
        "country": ["IN","IN","IN","IN"]
    })

    # No file I/O — directly pass DF into agents
    data_agent = DataAgent(csv_path=None)
    data_agent.df = df
    baseline, current = data_agent.split_baseline_current()
    df_raw = df.copy()

    insight_agent = InsightAgent(baseline, current, df_raw)
    hypotheses = insight_agent.run()

    assert isinstance(hypotheses, list)

    # Evaluator
    config = {"confidence_min": 0.3}
    evaluator = EvaluatorAgent(df_raw, {"confidence_min": 0.3})
    validated = evaluator.validate(hypotheses)

    assert isinstance(validated, list)

    # Creative generator
    cg = CreativeGenerator(df_raw)
    campaigns = ["A", "B"]
    creatives = cg.generate_for_campaigns(campaigns)
    assert "A" in creatives
    assert "suggestions" in creatives["A"]
