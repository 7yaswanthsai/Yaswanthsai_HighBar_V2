import sys
import os
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from agents.data_agent import DataAgent, SchemaError


def test_missing_columns_fails_fast(tmp_path):
    # create CSV missing required columns
    p = tmp_path / "bad.csv"
    pd.DataFrame({"date": ["2025-01-01"]}).to_csv(p, index=False)

    agent = DataAgent(str(p))

    with pytest.raises(SchemaError):
        agent.load()


def test_null_pattern_detection(tmp_path):
    # column with >50% null → should raise
    p = tmp_path / "nulls.csv"
    pd.DataFrame({
        "campaign_name": ["A", None, None],
        "adset_name": ["a","a","a"],
        "date": ["2025-01-01","2025-01-02","2025-01-03"],
        "spend": [10, 20, 30],
        "impressions": [100, 200, 300],
        "clicks": [1,2,3],
        "ctr": [0.01,0.02,0.03],
        "purchases": [1,0,1],
        "revenue": [100, 150, 200],
        "roas": [10,7,6],
        "creative_type": ["img","img","img"],
        "creative_message": ["x","y","z"],
        "audience_type": ["broad","broad","broad"],
        "platform": ["fb","fb","fb"],
        "country": ["IN","IN","IN"]
    }).to_csv(p, index=False)

    agent = DataAgent(str(p))
    with pytest.raises(SchemaError):
        agent.load()


def test_date_parsing_issue(tmp_path):
    # date fails parsing → should raise because all become NaT
    p = tmp_path / "bad_date.csv"
    pd.DataFrame({
        "campaign_name": ["A"],
        "adset_name": ["a"],
        "date": ["NOT_A_DATE"],
        "spend": [10],
        "impressions": [100],
        "clicks": [1],
        "ctr": [0.01],
        "purchases": [1],
        "revenue": [100],
        "roas": [10],
        "creative_type": ["img"],
        "creative_message": ["good"],
        "audience_type": ["broad"],
        "platform": ["fb"],
        "country": ["IN"]
    }).to_csv(p, index=False)

    agent = DataAgent(str(p))
    with pytest.raises(SchemaError):
        agent.load()
