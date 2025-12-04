# Kasparro — Agentic Facebook Performance Analyst  
Author: **Yaswanth Sai**

This project implements a multi-agent, production-style analysis pipeline for Facebook Ads.  
It diagnoses CTR/ROAS changes, validates hypotheses with evidence, and generates creative ideas grounded in the dataset.

This version includes all improvements requested in **P0, P1 and P2**:  
- Schema validation  
- Enhanced logging  
- Retry + backoff  
- Integration tests  
- Schema drift detection  
- Lightweight metrics layer  

---

## 🚀 Quick Start

```bash
python -V              # Python >= 3.10 recommended

python -m venv .venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt

python src/run.py "Analyze ROAS drop"
````

Outputs will appear in:

```
reports/report.md
reports/insights.json
reports/creatives.json
logs/log_<timestamp>.json
```

---

# 📁 Project Structure

```
kasparro-agentic-fb-analyst-yaswanth-sai/
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
├── reports/
├── logs/
└── tests/
    ├── test_data_agent.py
    ├── test_evaluator.py
    ├── test_pipeline.py
    ├── test_integration.py
    ├── test_metrics_layer.py
    └── test_schema_drift.py
```

---

# ⚙️ Configuration

`config/config.yaml`:

```yaml
random_seed: 42
confidence_min: 0.6

data_csv: "data/synthetic_fb_ads_undergarments.csv"

output_dir: "reports"
logs_dir: "logs"

report_file: "reports/report.md"
insights_file: "reports/insights.json"
creatives_file: "reports/creatives.json"

schema_drift_mode: "warn"    # fail | warn | off
sample_window_days: 30
```

---

# 🧠 Agent Architecture

## 1. Planner Agent

Creates a structured task list covering:

* Data loading
* Insight generation
* Validation
* Creative generation
* Report compilation

## 2. Data Agent

Production-style data layer:

* Schema validation
* Type enforcement
* Null-pattern checks
* Configurable schema drift detection (P2)
* Cleaned numeric columns
* Campaign + time-series summaries
* Detailed logs

## 3. Insight Agent

Generates data-backed hypotheses using:

* CTR/ROAS trends
* Spend and efficiency signals
* Message performance patterns
* Frequency / fatigue detection

## 4. Evaluator Agent

Validates hypotheses with:

* Baseline vs current comparisons
* Metric deltas
* Correlation strength
* Confidence scoring
* Structured decision explanations

## 5. Creative Generator

Produces data-grounded variations:

* Extracts themes from messages
* Recombines strong phrases
* Generates suggestions per low-CTR campaign

---

# 📄 CLI Example

```bash
python src/run.py "Analyze ROAS drop"
```

---

# 📊 Example Output

### `insights.json`

Contains: plan, summary, hypotheses, validated decisions.

### `creatives.json`

Campaign → Suggested messages.

### `report.md`

Readable summary for marketers.

---

# 🧪 Tests (P1)

Run:

```bash
pytest -q
```

Expected:

```
10 passed
```

Tests cover:

* Schema validation
* Schema drift behaviour
* Evaluator scoring
* Metrics layer
* Retry wrapping
* Full pipeline integration

---

# 🔍 Observability & Logging (P0 + P1)

Logs contain:

* Step timings
* Hypothesis count
* Evaluator decisions
* Creative summaries
* Retry attempts
* Schema drift warnings/errors
* Metrics snapshot

A new log file is written for every run.

---

# 📊 Metrics Layer (P2)

Lightweight, in-memory:

* Counters (rows, hypotheses, valid insights)
* Timers (data_load, evaluation, creative_generation, run.total)

Included directly in the final log.

---

# 🔐 Reproducibility

* Deterministic (seeded)
* Pinned dependencies
* Strict schema governance
* Fully configurable thresholds

---

# 🏷️ Release Instructions

```bash
git add .
git commit -m "P0 P1 P2 improvements: schema validation, drift detection, retry logic, metrics, tests"
git push
```

Create a PR titled **“self-review”** summarizing:

* Architecture decisions
* Why the approach was chosen
* Known limitations
* Possible next steps

---

# 🎯 Submission Summary (For Reviewer)

### **P0 – Completed**

✔ Schema validation (required columns, types, null-patterns)

✔ Fail-fast behaviour

✔ Enhanced logging with step timings and structured events

### **P1 – Completed**

✔ Retry logic with exponential backoff

✔ Integration tests for full pipeline

✔ Edge-case tests for DataAgent and Evaluator

### **P2 – Completed**

✔ Configurable schema drift detector (fail/warn/off)

✔ Lightweight metrics layer with timer + counter snapshots

This repository is updated and ready for review.
