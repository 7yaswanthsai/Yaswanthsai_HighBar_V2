import argparse
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from src.utils import load_config, save_json, set_seeds, retry, StructuredLogger, Metrics

from src.agents.data_agent import DataAgent
from src.agents.planner import PlannerAgent
from src.agents.insight_agent import InsightAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.creative_generator import CreativeGenerator


def timed_step(name: str, func, *args, **kwargs):
    """Utility wrapper to measure duration of each agent step."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, round(end - start, 4)


def main(user_query: str, config_path: str = "config/config.yaml"):
    # ─────────────────────────────────────────────
    # Preload config and create run id & logger & metrics
    # ─────────────────────────────────────────────
    config = load_config(config_path)
    set_seeds(config.get("random_seed", 42))

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    logs_dir = config.get("logs_dir", "logs")
    run_logger = StructuredLogger(name="kasparro_run", run_id=run_id, logs_dir=logs_dir)
    metrics = Metrics()

    run_log = {
        "run_id": run_id,
        "query": user_query,
        "start_time": datetime.utcnow().isoformat(),
        "steps": {},
    }

    run_logger.info({"event": "run_start", "run_id": run_id, "query": user_query})
    metrics.incr("run.start", 1)
    metrics.start_timer("run.total")

    data_path = config.get("data_csv", "data/sample_fb_ads.csv")

    # ─────────────────────────────────────────────
    # STEP 1 — Planner
    # ─────────────────────────────────────────────
    planner = PlannerAgent()
    try:
        plan, t = timed_step("planner", planner.plan, user_query)
        run_log["steps"]["planner"] = {"duration_sec": t, "plan": plan}
        run_logger.info({"event": "planner_done", "duration_sec": t, "plan_len": len(plan.get("tasks", [])) if isinstance(plan, dict) else None})
        metrics.incr("planner.runs", 1)
        metrics.start_timer("planner")
        metrics.stop_timer("planner")
    except Exception as e:
        run_logger.error({"event": "planner_failed", "error": str(e)})
        raise

    # ─────────────────────────────────────────────
    # STEP 2 — Data Agent (with config-driven drift behavior)
    # ─────────────────────────────────────────────
    data_agent = DataAgent(data_path, logger=run_logger, config=config)
    load_with_retry = retry(attempts=3, initial_delay=0.5, backoff=2.0, logger=run_logger)(data_agent.load)

    try:
        metrics.start_timer("data_load")
        df, t_load = timed_step("data_load", load_with_retry)
        metrics.stop_timer("data_load")
        metrics.incr("data.rows", len(df))
        summary, t_summary = timed_step("data_summary", data_agent.summary)

        run_log["steps"]["data_agent"] = {
            "duration_load_sec": t_load,
            "duration_summary_sec": t_summary,
            "rows": len(df),
            "columns": df.columns.tolist(),
            "sample_head": df.head(3).to_dict(orient="records"),
        }
        run_logger.info({"event": "data_loaded", "rows": len(df), "duration_load_sec": t_load})
    except Exception as e:
        run_logger.error({"event": "data_failed", "error": str(e)})
        raise

    # ─────────────────────────────────────────────
    # STEP 3 — Insight Agent (with retry)
    # ─────────────────────────────────────────────
    insight_agent = InsightAgent(df)
    generate_insights_with_retry = retry(attempts=3, initial_delay=0.5, backoff=2.0, logger=run_logger)(insight_agent.generate_candidates)
    try:
        metrics.start_timer("insights")
        hypotheses, t_h = timed_step("insight_generation", generate_insights_with_retry)
        metrics.stop_timer("insights")
        run_log["steps"]["insight_agent"] = {
            "duration_sec": t_h,
            "num_hypotheses": len(hypotheses),
            "hypothesis_titles": [h.get("hypothesis") for h in hypotheses[:100]]  # sample first 100 titles
        }
        run_logger.info({"event": "insights_generated", "num_hypotheses": len(hypotheses), "duration_sec": t_h})
        metrics.incr("insights.count", len(hypotheses))
    except Exception as e:
        run_logger.error({"event": "insight_generation_failed", "error": str(e)})
        raise

    # ─────────────────────────────────────────────
    # STEP 4 — Evaluator Agent
    # ─────────────────────────────────────────────
    evaluator = EvaluatorAgent(df, config)
    try:
        metrics.start_timer("evaluation")
        validated, t_eval = timed_step("evaluator", evaluator.validate, hypotheses)
        metrics.stop_timer("evaluation")
        run_log["steps"]["evaluator"] = {
            "duration_sec": t_eval,
            "num_valid": sum(1 for h in validated if h.get("valid")),
            "decisions": [
                {
                    "hypothesis": h.get("hypothesis"),
                    "valid": h.get("valid"),
                    "confidence": round(float(h.get("confidence", 0)), 3),
                    "campaign": h.get("campaign")
                }
                for h in validated
            ],
        }
        run_logger.info({"event": "evaluation_done", "num_valid": run_log["steps"]["evaluator"]["num_valid"], "duration_sec": t_eval})
        metrics.incr("evaluation.valid", run_log["steps"]["evaluator"]["num_valid"])
    except Exception as e:
        run_logger.error({"event": "evaluation_failed", "error": str(e)})
        raise

    # ─────────────────────────────────────────────
    # STEP 5 — Creative Generator
    # ─────────────────────────────────────────────
    low_ctr_campaigns = [
        h.get("campaign")
        for h in validated
        if h.get("valid") and h.get("campaign") is not None and "ctr" in h.get("hypothesis", "").lower()
    ]

    if not low_ctr_campaigns:
        dfc = df.groupby("campaign_name").agg({"clicks": "sum", "impressions": "sum"})
        dfc["ctr"] = dfc["clicks"] / dfc["impressions"].replace(0, 1)
        low_ctr_campaigns = dfc.sort_values("ctr").head(2).index.tolist()

    creative_gen = CreativeGenerator(df)
    try:
        metrics.start_timer("creative_generation")
        creatives, t_creative = timed_step("creative_generation", creative_gen.generate_for_campaigns, low_ctr_campaigns)
        metrics.stop_timer("creative_generation")
        run_log["steps"]["creative_generator"] = {
            "duration_sec": t_creative,
            "target_campaigns": low_ctr_campaigns,
            "output_count": {camp: len(v.get("suggestions", [])) for camp, v in creatives.items()}
        }
        run_logger.info({"event": "creatives_generated", "target_count": len(low_ctr_campaigns), "duration_sec": t_creative})
        metrics.incr("creatives.targeted", len(low_ctr_campaigns))
    except Exception as e:
        run_logger.error({"event": "creative_generation_failed", "error": str(e)})
        raise

    # ─────────────────────────────────────────────
    # SAVE OUTPUT FILES
    # ─────────────────────────────────────────────
    os.makedirs(config.get("output_dir", "reports"), exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    insights_path = config.get("insights_file", "reports/insights.json")
    creatives_path = config.get("creatives_file", "reports/creatives.json")
    report_path = config.get("report_file", "reports/report.md")
    log_file = os.path.join(logs_dir, f"log_{run_id}.json")

    save_json(
        {
            "query": user_query,
            "plan": plan,
            "summary": summary,
            "candidates": hypotheses,
            "validated": validated
        },
        insights_path
    )
    run_logger.info({"event": "insights_saved", "path": insights_path})

    save_json(creatives, creatives_path)
    run_logger.info({"event": "creatives_saved", "path": creatives_path})

    # Write log (structured run_log)
    run_log["end_time"] = datetime.utcnow().isoformat()
    # include raw logger events and metrics for observability
    try:
        run_log["_logger_events"] = run_logger.get_events()
    except Exception:
        run_log["_logger_events"] = []

    run_log["_metrics"] = metrics.snapshot()
    save_json(run_log, log_file)
    run_logger.info({"event": "run_log_saved", "path": log_file})

    # Generate Markdown Report
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Facebook Ads Performance Analysis\n\n")
            f.write(f"Query: **{user_query}**\n\n")

            f.write("## Key Insights\n")
            for v in validated:
                if v.get("valid"):
                    f.write(f"- **{v.get('hypothesis')}** (Campaign: {v.get('campaign')}, Confidence: {v.get('confidence'):.2f})\n")

            f.write("\n## Creative Recommendations\n")
            for camp, data in creatives.items():
                f.write(f"### {camp}\n")
                for s in data.get("suggestions", []):
                    f.write(f"- {s}\n")

        run_logger.info({"event": "report_saved", "path": report_path})
    except Exception as e:
        run_logger.error({"event": "report_save_failed", "error": str(e)})
        raise

    metrics.stop_timer("run.total")
    metrics.incr("run.completed", 1)

    print(f"[✓] Insights saved: {insights_path}")
    print(f"[✓] Creative ideas saved: {creatives_path}")
    print(f"[✓] Report saved: {report_path}")
    print(f"[✓] Log saved: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="User query such as 'Analyze ROAS drop'")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    main(args.query, args.config)
