# src/orchestrator.py
import os
import time
from pathlib import Path
import json
from functools import wraps
from typing import Callable, Any
from src.utils import load_config, save_json, set_seeds
from src.telemetry import MetricsCollector
from src.logging.run_logger import RunLogger
from src.agents.data_agent import DataAgent
from src.agents.planner import PlannerAgent
from src.agents.insight_agent import InsightAgent
from src.agents.evaluator import EvaluatorAgent
from src.agents.creative_generator import CreativeGenerator

# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def make_run_folder(base="logs"):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"run_{ts}"
    ensure_dir(run_dir)
    return run_dir


def retry_with_backoff(max_retries=3, base_delay=1.0, factor=2.0, on_retry=None):
    """
    Decorator for retrying functions with exponential backoff.
    `on_retry` is an optional callback invoked with (attempt, exception).
    """

    def deco(fn: Callable):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    delay = base_delay * (factor ** (attempt - 1))
                    if on_retry:
                        try:
                            on_retry(attempt, e, delay)
                        except Exception:
                            pass
                    time.sleep(delay)
        return wrapped
    return deco


# -------------------------
# Orchestrator (main)
# -------------------------
def run_analysis(user_query: str, config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    set_seeds(cfg.get("random_seed", 42))

    # Create run folder
    run_dir = make_run_folder(cfg.get("logs_dir", "logs"))

    # Create top-level logger (orchestrator)
    orch_logger = RunLogger(run_dir, agent_name="orchestrator")
    orch_logger.human("Starting run")
    orch_logger.info({"event": "run_start", "user_query": user_query})

    metrics = MetricsCollector()
    metrics.set_meta("user_query", user_query)
    metrics.set_meta("config_path", config_path)
    metrics.set_meta("start_time", time.time())

    # Planner step
    planner_logger = RunLogger(run_dir, agent_name="planner")
    try:
        planner = PlannerAgent()
        plan = planner.plan(user_query)
        planner_logger.info({"event": "plan_created", "plan": plan})
        save_json(plan, Path(run_dir) / "plan.json")
        metrics.counter("planner_runs")
    except Exception as e:
        planner_logger.error({"event": "planner_error", "error": str(e)})
        plan = {"error": str(e)}
        metrics.counter("planner_failures")

    # Data step with retry
    data_logger = RunLogger(run_dir, agent_name="data_agent")
    data_agent = DataAgent(cfg["data_csv"], logger=data_logger, config=cfg)

    @retry_with_backoff(
        max_retries=cfg.get("retry_max", 3),
        base_delay=cfg.get("retry_base_delay", 1.0),
        factor=cfg.get("retry_factor", 2.0),
        on_retry=lambda attempt, ex, delay: data_logger.warning({"event": "data_retry", "attempt": attempt, "error": str(ex), "delay": delay})
    )
    def _load_data():
        t0 = time.time()
        df = data_agent.load()
        metrics.timing("data_load_sec", time.time() - t0)
        metrics.counter("data_loaded_rows", len(df))
        return df

    try:
        df = _load_data()
        data_logger.info({"event": "data_loaded_ok", "rows": len(df)})
        save_json({"schema": df.dtypes.astype(str).to_dict()}, Path(run_dir) / "input_schema.json")
    except Exception as e:
        data_logger.error({"event": "data_load_failed", "error": str(e)})
        # persist an error summary and stop early with summary
        summary = {"error": "data_load_failed", "reason": str(e)}
        save_json(summary, Path(run_dir) / "error.json")
        orch_logger.error({"event": "run_failed", "reason": str(e)})
        metrics.set_meta("end_time", time.time())
        metrics.emit(run_dir)
        return {"status": "failed", "reason": str(e), "run_dir": str(run_dir)}

    # Baseline / current split + segment aggregation
    try:
        t0 = time.time()
        baseline, current = data_agent.split_baseline_current()
        b_agg, c_agg = data_agent.segment_analysis(baseline, current)
        metrics.timing("split_and_aggregate_sec", time.time() - t0)
        data_logger.info({"event": "split_done", "baseline_rows": len(baseline), "current_rows": len(current)})
        # save CSV aggregations for review
        b_agg.to_csv(Path(run_dir) / "baseline_agg.csv", index=False)
        c_agg.to_csv(Path(run_dir) / "current_agg.csv", index=False)
    except Exception as e:
        data_logger.error({"event": "split_error", "error": str(e)})
        save_json({"error": str(e)}, Path(run_dir) / "error.json")
        metrics.set_meta("end_time", time.time())
        metrics.emit(run_dir)
        return {"status": "failed", "reason": str(e), "run_dir": str(run_dir)}

    # Save input summary
    try:
        input_summary = data_agent.summary()
        save_json(input_summary, Path(run_dir) / "input_summary.json")
    except Exception as e:
        data_logger.warning({"event": "summary_failed", "error": str(e)})

    # Insight step with retry
    insight_logger = RunLogger(run_dir, agent_name="insight_agent")

    insight_agent = InsightAgent(baseline=b_agg, current=c_agg, raw_df=df, config=cfg, logger=insight_logger)

    @retry_with_backoff(
        max_retries=cfg.get("retry_max", 2),
        base_delay=cfg.get("retry_base_delay", 1.0),
        factor=cfg.get("retry_factor", 2.0),
        on_retry=lambda attempt, ex, delay: insight_logger.warning({"event": "insight_retry", "attempt": attempt, "error": str(ex), "delay": delay})
    )
    def _generate_insights():
        t0 = time.time()
        hyps = insight_agent.run()
        metrics.timing("insights_sec", time.time() - t0)
        metrics.counter("hypotheses_generated", len(hyps))
        return hyps

    try:
        hypotheses = _generate_insights()
        insight_logger.info({"event": "insights_generated", "count": len(hypotheses)})
        save_json(hypotheses, Path(run_dir) / "hypotheses.json")
    except Exception as e:
        insight_logger.error({"event": "insight_failed", "error": str(e)})
        save_json({"error": str(e)}, Path(run_dir) / "error.json")
        metrics.set_meta("end_time", time.time())
        metrics.emit(run_dir)
        return {"status": "failed", "reason": str(e), "run_dir": str(run_dir)}

    # Evaluation
    eval_logger = RunLogger(run_dir, agent_name="evaluator")
    evaluator = EvaluatorAgent(config=cfg, logger=eval_logger)

    try:
        t0 = time.time()
        validated = evaluator.validate(hypotheses)
        metrics.timing("evaluation_sec", time.time() - t0)
        metrics.counter("hypotheses_validated", len(validated))
        save_json(validated, Path(run_dir) / "validated.json")
        eval_logger.info({"event": "evaluation_complete", "validated_count": len(validated)})
    except Exception as e:
        eval_logger.error({"event": "evaluation_failed", "error": str(e)})
        save_json({"error": str(e)}, Path(run_dir) / "error.json")
        validated = []

    # Creative generation
    creative_logger = RunLogger(run_dir, agent_name="creative")
    try:
        # pick campaigns for creative generation using validated high-confidence ctr failures
        valid_hypotheses = [v for v in validated if v.get("valid")]
        low_ctr_campaigns = []
        for v in valid_hypotheses:
            if v.get("metric") == "ctr":
                seg = v.get("segment", {})
                if seg and seg.get("campaign_name"):
                    low_ctr_campaigns.append(seg.get("campaign_name"))

        creative_gen = CreativeGenerator(df)
        t0 = time.time()
        creatives = creative_gen.generate_for_campaigns(low_ctr_campaigns)
        metrics.timing("creative_sec", time.time() - t0)
        metrics.counter("creatives_generated", len(creatives) if creatives else 0)
        save_json(creatives, Path(run_dir) / "creatives.json")
        creative_logger.info({"event": "creatives_generated", "count": len(creatives) if creatives else 0})
    except Exception as e:
        creative_logger.error({"event": "creative_failed", "error": str(e)})
        creatives = []

    # Finalize run: metrics + report summary
    metrics.set_meta("end_time", time.time())
    metrics.emit(run_dir)

    summary = {
        "run_dir": str(run_dir),
        "plan": plan,
        "hypotheses_count": len(hypotheses),
        "validated_count": len([v for v in validated if v.get("valid")]),
        "creatives_count": len(creatives) if creatives else 0,
    }
    save_json(summary, Path(run_dir) / "report_summary.json")
    orch_logger.info({"event": "run_complete", "summary": summary})
    orch_logger.human(f"Run complete: {summary}")

    return summary
