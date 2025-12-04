import sys
import os
import pandas as pd

# Fix path so Python can resolve src/*
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from agents.data_agent import DataAgent
from agents.insight_agent import InsightAgent
from agents.evaluator import EvaluatorAgent

CSV = """campaign_name,adset_name,date,spend,impressions,clicks,ctr,purchases,revenue,roas,creative_type,creative_message,audience_type,platform,country
C1,A1,2025-01-01,100,1000,10,0.01,1,100,1.0,Image,"soft cotton",Cold,Facebook,IN
C1,A1,2025-01-02,200,2000,8,0.004,0,80,0.4,Image,"soft cotton",Cold,Facebook,IN
"""

def test_pipeline_small(tmp_path):
    p = tmp_path / "small.csv"
    p.write_text(CSV)

    da = DataAgent(str(p))
    df = da.load()
    baseline, current = da.split_baseline_current()

    ia = InsightAgent(baseline, current, df)
    cands = ia.run()


    ev = EvaluatorAgent(df, {"confidence_min": 0.5})
    validated = ev.validate(cands)


    assert isinstance(validated, list)
