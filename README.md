# Kasparro — High-Bar V2 Facebook Performance Analyst

**Author:** Suragani Yaswanth Sai

This repository implements a production-style multi-agent analytics system for diagnosing Facebook Ads performance changes.

It goes beyond a simple heuristic pipeline and includes:

  * Baseline vs Current segmentation
  * Metric deltas (absolute & relative)
  * Evidence-backed hypotheses
  * Strict evaluator with confidence modeling
  * Schema validation & drift detection
  * Full observability (JSONL logs per agent, readable logs, metrics)
  * Lightweight metrics layer
  * Retry logic
  * Complete test suite

**This version satisfies all requirements for P0 → P1 → P2 → V2.**

-----

## 🚀 Quick Start

Ensure you have **Python \>= 3.10**.

```bash
# Check Python version
python -V

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/run.py "Why did CTR drop?"
```

**Output files are written to:**

  * `reports/`
  * `logs/run_<timestamp>/`

-----

## 📁 Project Structure

```text
Yaswanthsai_HighBar_V2/
├── agent_graph.md
├── Makefile
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── data/
│   ├── sample_fb_ads.csv
│   ├── synthetic_fb_ads_undergarments.csv
│   └── README.md
├── prompts/
│   ├── planner_prompt.md
│   ├── insight_prompt.md
│   └── creative_prompt.md
├── src/
│   ├── run.py
│   ├── orchestrator.py
│   ├── utils.py
│   └── agents/
│       ├── planner.py
│       ├── data_agent.py
│       ├── insight_agent.py
│       ├── evaluator.py
│       └── creative_generator.py
├── logs/
│   └── run_<timestamp>/
│       ├── planner.jsonl
│       ├── data_agent.jsonl
│       ├── insight_agent.jsonl
│       ├── evaluator.jsonl
│       ├── creative.jsonl
│       ├── orchestrator.jsonl
│       ├── baseline_agg.csv
│       ├── current_agg.csv
│       ├── hypotheses.json
│       ├── validated.json
│       ├── creatives.json
│       ├── input_schema.json
│       ├── input_summary.json
│       ├── report_summary.json
│       ├── metrics.json
│       └── run_readable.log
└── tests/
    ├── test_data_agent.py
    ├── test_evaluator.py
    ├── test_pipeline.py
    ├── test_integration.py
    ├── test_metrics_layer.py
    └── test_schema_drift.py
```

-----

## ⚙️ Configuration (`config/config.yaml`)

```yaml
python: "3.10"
random_seed: 42
confidence_min: 0.6
schema_drift_mode: "fail"   # fail | warn | off
sample_window_days: 30
data_csv: "data/synthetic_fb_ads_undergarments.csv"
output_dir: "reports"
logs_dir: "logs"
report_file: "reports/report.md"
insights_file: "reports/insights.json"
creatives_file: "reports/creatives.json"
```

-----

## 🧠 Architecture Overview

A production-style multi-agent pipeline:

> **Planner → Data Agent → Insight Agent V2 → Evaluator V2 → Creative Generator → Report**

### 1\. Planner Agent

Creates a step-level plan for the run.

### 2\. Data Agent (Production Data Layer)

  * Schema validation
  * Type enforcement
  * Null-pattern checks
  * Configurable schema drift detection
  * Baseline vs current split
  * Time-series & campaign-level summaries
  * Input-schema & input-summary logs

### 3\. Insight Agent V2

Generates structured hypotheses using:

  * Baseline vs Current segmentation
  * Absolute & relative deltas
  * Metric trends (slope-based)
  * Frequency fatigue
  * Creative/message performance clusters
  * Spend vs ROAS correlations

**Example Hypothesis:**

```json
{
  "id": "...",
  "segment": {...},
  "metric": "ctr",
  "baseline": 0.012,
  "current": 0.007,
  "delta_abs": -0.005,
  "delta_rel": -0.41,
  "sample_size": 1820,
  "impact": "medium",
  "impact_score": 0.74,
  "confidence": 0.68,
  "evidence": {...}
}
```

### 4\. Evaluator Agent V2 (Strict Mode)

Assigns confidence, validity, reasons, contradiction detection, sample-size checks, and impact-weighted overrides.

**Evaluator output:**

```json
{
 "id": "ctr_drop_A",
 "valid": false,
 "confidence": 0.52,
 "reasons": [
   "sample_size_below_min",
   "impact_score_low",
   "confidence_below_min"
 ]
}
```

### 5\. Creative Generator

Produces grounded suggestions per low-CTR segment.

-----

## 🔍 Observability: What Gets Logged

Every run produces a full folder under `logs/run_<timestamp>/`. This makes the system fully diagnosable by another engineer.

**Includes:**

  * Per-agent JSONL logs
  * Human-readable log (`run_readable.log`)
  * Metrics snapshot (`metrics.json`)
  * Input schema & summary
  * Hypotheses + validated insights
  * Creatives
  * Baseline & Current aggregates
  * Orchestrator trace

-----

## 📊 Lightweight Metrics Layer

Example `metrics.json`:

```json
{
  "counters": {
    "rows_loaded": 1245,
    "hypotheses_generated": 14,
    "hypotheses_validated": 6
  },
  "timings": {
    "data_load": 0.181,
    "insight_generation": 0.432,
    "evaluation": 0.117,
    "creative_generation": 0.053,
    "run_total": 1.08
  }
}
```

-----

## 🧪 Running Tests

```bash
pytest -q
```

**Expected Output:**

```text
10 passed
```

**Tests cover:**

  * Schema validation
  * Drift detection
  * Insight generation
  * Strict evaluator
  * Metrics layer
  * Retry logic
  * Integration pipeline

-----

## 🏗️ Developer Notes

### Extending Agents

Each agent is fully isolated.

  * **To add new rules or signals:** Modify `src/agents/insight_agent.py`
  * **Adding new metrics:** Update `src/utils/metrics.py`
  * **Adding new drift rules:** Modify `src/agents/data_agent.py`

-----

## 🎯 V2 Submission Summary

### Engineering Deliverables

  * ✔ Strict Evaluator V2
  * ✔ InsightAgent V2 with baseline/current deltas
  * ✔ Schema validation + drift detection
  * ✔ Retry logic
  * ✔ Logging & observability per agent
  * ✔ Metrics layer
  * ✔ Full test suite (all green)
  * ✔ Deterministic, seeded pipeline

### Production Traits

  * ✔ Fail-fast behavior
  * ✔ Structured logs
  * ✔ Reproducible outputs
  * ✔ Clear thresholds
  * ✔ Safe fallbacks
  * ✔ End-to-end diagnosability

**This repository satisfies the High-Bar V2 requirements.**