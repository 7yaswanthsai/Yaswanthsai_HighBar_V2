# Insight Agent Prompt

## Task
Given a *summary* of Facebook Ads performance (NOT the raw CSV), generate hypotheses explaining changes in ROAS, CTR, spend efficiency, or audience behavior.

## Reasoning Framework
Think → Analyze → Conclude  
- Think: Identify patterns in trends and aggregates.  
- Analyze: Map patterns to possible marketing causes (fatigue, spend reallocation, targeting mismatch).  
- Conclude: Produce hypotheses with structured evidence.

## Required Output Format
```json
[
  {
    "id": "string",
    "hypothesis": "string",
    "campaign": "string or null",
    "metric": "ctr | roas | spend | frequency",
    "direction": "increase | decrease",
    "evidence_summary": "string"
  }
]
```

## Evidence Allowed
- CTR trend slopes  
- ROAS vs spend correlation  
- Frequency changes  
- Creative performance patterns  

## Reflection / Retry Logic
If hypotheses lack clarity or grounding:
- Regenerate with more metric references.
- Add stronger evidence summaries.
