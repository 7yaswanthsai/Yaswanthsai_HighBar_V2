import sys
import os
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from agents.data_agent import DataAgent, SchemaError

def test_schema_drift_warn(tmp_path):
    p = tmp_path / "drift_warn.csv"

    df = pd.DataFrame({
        "campaign_name": ["A"],
        "adset_name": ["A"],
        # missing many required columns
    })
    df.to_csv(p, index=False)

    agent = DataAgent(str(p), config={"schema_drift_mode": "warn"})
    df_out = agent.load()
    assert isinstance(df_out, pd.DataFrame)



def test_schema_extra_columns(tmp_path):
    p = tmp_path / "extra.csv"

    df = pd.DataFrame({
        "campaign_name": ["A"],
        "adset_name": ["x"],
        "date": ["2025-01-01"],
        # required columns...
        "spend": [10],
        "impressions": [100],
        "clicks": [1],
        "ctr": [0.01],
        "purchases": [1],
        "revenue": [100],
        "roas": [10],
        "creative_type": ["img"],
        "creative_message": ["x"],
        "audience_type": ["broad"],
        "platform": ["fb"],
        "country": ["IN"],

        "extra_column_123": ["hello"]   # <── drift
    })
    df.to_csv(p, index=False)

    agent = DataAgent(str(p), logger=None)
    # expect NO failure — only logging
    df_out = agent.load()
    assert "extra_column_123" in df_out.columns
