import pytest

def test_pipeline_detects_drift(tmp_path):
    import pytest
    csv_path = tmp_path / "drift.csv"

    csv_path.write_text("campaign_name,WRONG_COLUMN\nA,B")

    from run import main

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
data_csv: {csv_path}
logs_dir: logs
schema_drift_mode: fail
""")

    with pytest.raises(Exception):
        main("Analyze ROAS", str(cfg))


