import sys, os
import pandas as pd
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.orchestrator import run_analysis


def test_metrics_exist(tmp_path):
    # create dataset first
    df_path = tmp_path / "sample.csv"
    df_path.write_text("""campaign_name,adset_name,date,spend,impressions,clicks,ctr,purchases,revenue,roas,creative_type,creative_message,audience_type,platform,country
C1,A1,2025-01-01,100,1000,10,0.01,1,100,1.0,img,x,broad,fb,IN
""")

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(f"""
data_csv: {df_path}
logs_dir: logs
output_dir: reports
insights_file: reports/insights.json
creatives_file: reports/creatives.json
report_file: reports/report.md
schema_drift_mode: fail
""")


    result = run_analysis("Test Query", config_path=str(cfg_path))
    run_dir = result["run_dir"]
    metrics_path = os.path.join(run_dir, "metrics.json")

    assert os.path.exists(metrics_path)

