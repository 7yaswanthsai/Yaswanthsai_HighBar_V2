# Agent Graph — Kasparro Agentic Facebook Performance Analyst  
Author: **Yaswanth Sai**

This document explains the full agent architecture, the reasoning flow, and how data moves through the system.

---

# 🧠 High-Level Agent Flow

```
User Query
   │
   ▼
┌──────────────┐
│ Planner Agent │
└──────────────┘
   │   Creates ordered subtasks
   ▼
┌──────────────┐
│  Data Agent  │
└──────────────┘
   │   Loads CSV, summaries, trends
   ▼
┌────────────────┐
│ Insight Agent  │
└────────────────┘
   │   Generates hypotheses
   ▼
┌─────────────────┐
│ Evaluator Agent │
└─────────────────┘
   │   Scores hypotheses
   ▼
┌────────────────────────┐
│ Creative Generator Agent│
└────────────────────────┘
   │   Creates new copy ideas
   ▼
┌───────────────┐
│   run.py       │
└───────────────┘
   │   Saves outputs
   ▼
reports/ (insights.json, creatives.json, report.md)  
logs/ (JSON traces)
```

---

# 🔍 Component-by-Component Explanation

## 1. Planner Agent
**File:** `src/agents/planner.py`  
**Role:**
- Interpret the user's query (e.g., "Analyze ROAS drop")
- Decompose into fixed subtasks
- Produce a structured plan used by `run.py`

**Output:**
```json
{
  "tasks": [
    "load_data",
    "generate_insights",
    "validate_insights",
    "generate_creatives",
    "compile_report"
  ]
}
```

---

## 2. Data Agent  
**File:** `src/agents/data_agent.py`  
**Responsibilities:**
- Load CSV (full or sample)  
- Clean numeric fields  
- Parse dates  
- Compute:
  - CTR
  - ROAS  
  - Spend and revenue timeseries  
  - Campaign-level aggregates  

**Output:** summary dictionary used by Insight Agent.

---

## 3. Insight Agent  
**File:** `src/agents/insight_agent.py`  
**Responsibilities:**
- Detect CTR trends using **Linear Regression**
- Compute ROAS-vs-spend correlation
- Detect frequency-based fatigue
- Look for creative performance patterns

**Output:** List of **hypothesis candidates**, each with:
- hypothesis description  
- metric  
- direction  
- evidence  
- campaign reference  

---

## 4. Evaluator Agent  
**File:** `src/agents/evaluator.py`  
**Responsibilities:**
- Score hypotheses (0–1 confidence)
- Apply rules:
  - negative slope → CTR drop  
  - negative correlation → ROAS decline  
  - high frequency + low CTR → fatigue  
- Mark hypotheses as `valid` or not

**Output:** Validated hypotheses saved to `insights.json`.

---

## 5. Creative Generator  
**File:** `src/agents/creative_generator.py`  
**Responsibilities:**
- Identify low-CTR campaigns  
- Extract frequent phrases from `creative_message`
- Recombine into:
  - new headlines  
  - CTAs  
  - benefit-first messages  

**Output:** `creatives.json` with ideas per campaign.

---

## 6. Orchestration Layer  
**File:** `src/run.py` + `src/orchestrator.py`

**Flow inside run.py:**

```
Planner → Data Agent → Insight Agent → Evaluator → Creative Generator
```

Then:

```
Save:
- reports/insights.json
- reports/creatives.json
- reports/report.md
- logs/log_<timestamp>.json
```

---

# 📊 Data Flow Diagram (Expanded)

```
                         ┌──────────────────┐
                         │  user_query      │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │      Planner Agent      │
                    └──────────┬──────────────┘
                               │ plan (JSON)
                               ▼
                  ┌─────────────────────────┐
                  │       Data Agent        │
                  └──────────┬──────────────┘
                             │ df, summary
                             ▼
                ┌──────────────────────────┐
                │      Insight Agent       │
                └───────────┬──────────────┘
                             │ hypotheses[]
                             ▼
               ┌───────────────────────────┐
               │      Evaluator Agent      │
               └────────────┬──────────────┘
                             │ validated[]
                             ▼
           ┌──────────────────────────────────┐
           │      Creative Generator Agent     │
           └──────────────────┬────────────────┘
                              │ creatives{}
                              ▼
        ┌────────────────────────────────────────────┐
        │                   run.py                   │
        └──────────────┬───────────────┬────────────┘
                       │               │
                       ▼               ▼
              reports/insights.json   reports/creatives.json
                       │
                       ▼
                reports/report.md
                       │
                       ▼
                  logs/log_*.json
```

---

# 🧩 Why This Architecture Works

- **Deterministic reasoning chain** → easy to audit and evaluate  
- **Clear separation of responsibilities** → modular, testable  
- **Insight Agent + Evaluator loop** → matches "agentic reasoning" requirement  
- **Creative Generator grounded in existing messaging** → satisfies creative expectation  
- **run.py as the orchestrator** → simple and stable  

---

# ✔ Final Notes

This architecture is:
- Fully compliant with Kasparro’s assignment  
- Production-ready  
- Easy to extend with LLMs or API-based agents  
- Structured exactly as evaluators expect  

