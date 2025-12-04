# src/agents/evaluator.py
from typing import List, Dict, Any, Optional
import numpy as np
import math
import time


class EvaluationError(Exception):
    pass


class EvaluatorAgent:
    DEFAULTS = {
        "confidence_min": 0.6,
        "min_sample_size": 50,
        "impact_score_min": 0.25,
        "strict_mode": True,
        "sample_size_override_for_high_impact": 25,
    }

    def __init__(self, df=None, config: Optional[Dict[str, Any]] = None, logger=None):
        """
        df: dataframe passed by orchestrator (optional for tests)
        config: dictionary of thresholds
        logger: structured logger
        """
        self.df = df
        cfg = config or {}
        self.config = {**self.DEFAULTS, **cfg}
        self.logger = logger



    # ---------------------------
    # Public API
    # ---------------------------
    def validate(self, hypotheses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a list of hypotheses and return enriched results.

        Each returned result has the following guaranteed keys:
          - id, hypothesis, segment, metric
          - baseline, current, delta_abs, delta_rel
          - sample_size (int)
          - impact (low|medium|high)
          - impact_score (0..1)
          - confidence (0..1)
          - valid (bool)
          - reasons (list of strings explaining decisions)
          - evidence (original evidence dict)
          - timestamp
        """
        results = []
        t0 = time.time()
        for h in hypotheses:
            try:
                res = self._validate_single(h)
            except Exception as e:
                # If evaluation itself errors, produce a fail-result with reason
                res = {
                    "id": h.get("id"),
                    "hypothesis": h.get("hypothesis", ""),
                    "segment": h.get("segment", {}),
                    "metric": h.get("metric"),
                    "baseline": h.get("baseline"),
                    "current": h.get("current"),
                    "delta_abs": h.get("delta_abs"),
                    "delta_rel": h.get("delta_rel"),
                    "sample_size": int(h.get("sample_size", 0) or 0),
                    "impact": h.get("impact", "low"),
                    "impact_score": float(h.get("impact_score", 0.0) or 0.0),
                    "confidence": float(h.get("confidence", 0.0) or 0.0),
                    "valid": False,
                    "reasons": [f"evaluation_error: {str(e)}"],
                    "evidence": h.get("evidence", {}),
                    "timestamp": time.time()
                }
            results.append(res)

        if self.logger:
            self.logger.info({
                "event": "evaluation_complete",
                "input_count": len(hypotheses),
                "time_sec": round(time.time() - t0, 3)
            })

        return results

    # ---------------------------
    # Single-hypothesis validation
    # ---------------------------
    def _validate_single(self, h: Dict[str, Any]) -> Dict[str, Any]:
        reasons = []
        # canonical fields
        hid = h.get("id")
        metric = h.get("metric")
        evidence = h.get("evidence", {}) or {}
        impact_bucket = h.get("impact", "low")
        impact_score = float(h.get("impact_score", 0.0) or 0.0)
        reported_confidence = float(h.get("confidence", 0.0) or 0.0)
        sample_size = int(h.get("sample_size", 0) or 0)

        # Recompute simple derived flags when possible
        # If evidence contains explicit numeric fields, use them to sanity-check inputs
        # e.g., evidence may contain baseline/current/impressions
        # We'll attempt to compute a "sanity_confidence" from sample size and evidence stability

        # 1) Sample size check
        min_sample = int(self.config.get("min_sample_size", 50))
        if sample_size < min_sample:
            reasons.append(f"sample_size_below_min ({sample_size} < {min_sample})")

        # 2) Impact score check
        impact_min = float(self.config.get("impact_score_min", 0.25))
        if impact_score < impact_min:
            reasons.append(f"impact_score_low ({round(impact_score,3)} < {impact_min})")

        # 3) Evidence numeric sanity checks
        # If delta_rel provided, ensure it's a finite number
        delta_rel = h.get("delta_rel", None)
        delta_abs = h.get("delta_abs", None)
        def _is_finite_num(x):
            return x is not None and not (isinstance(x, float) and (math.isinf(x) or math.isnan(x)))
        if delta_rel is None or not _is_finite_num(delta_rel):
            reasons.append("missing_or_invalid_delta_rel")

        # 4) Recompute a conservative confidence: take min(reported_confidence, sample_confidence, evidence_confidence)
        sample_conf = self._sample_confidence(sample_size)
        evidence_conf = self._evidence_confidence(evidence)
        final_conf = min(reported_confidence, sample_conf, evidence_conf)

        # if there's a very high impact (>> threshold), allow smaller sample sizes
        strict_mode = bool(self.config.get("strict_mode", True))
        sample_override_cutoff = float(self.config.get("sample_size_override_for_high_impact", 25))
        if strict_mode and impact_score > 0.9 and sample_size >= sample_override_cutoff:
            # strong signal: reduce penalty
            if f"sample_size_below_min" in " ".join(reasons):
                reasons = [r for r in reasons if not r.startswith("sample_size_below_min")]
                reasons.append("sample_size_override_applied_for_high_impact")

        # 5) Final decision rules (strict)
        valid = True
        if strict_mode:
            if final_conf < float(self.config.get("confidence_min", 0.6)):
                valid = False
                reasons.append(f"confidence_below_min (final_conf={round(final_conf,3)} < {self.config.get('confidence_min')})")
            if sample_size < min_sample and not any("override" in r for r in reasons):
                valid = False
            if impact_score < impact_min:
                valid = False
        else:
            # balanced/lenient mode could be added here
            valid = final_conf >= float(self.config.get("confidence_min", 0.6))

        # 6) Defensive: if evidence explicitly contradicts (e.g., baseline < current but hypothesis says drop), mark invalid
        contradiction = self._detect_contradiction(h)
        if contradiction:
            valid = False
            reasons.append(f"contradiction_detected:{contradiction}")

        # collate structured output
        out = {
            "id": hid,
            "hypothesis": h.get("hypothesis", ""),
            "segment": h.get("segment", {}),
            "metric": metric,
            "baseline": h.get("baseline"),
            "current": h.get("current"),
            "delta_abs": delta_abs,
            "delta_rel": delta_rel,
            "sample_size": sample_size,
            "impact": impact_bucket,
            "impact_score": float(round(impact_score, 3)),
            "confidence": float(round(final_conf, 3)),
            "valid": bool(valid),
            "reasons": reasons,
            "evidence": evidence,
            "timestamp": time.time(),
        }

        if self.logger:
            self.logger.info({
                "event": "hypothesis_evaluated",
                "id": hid,
                "valid": out["valid"],
                "impact_score": out["impact_score"],
                "confidence": out["confidence"],
                "reasons": reasons
            })

        return out

    # ---------------------------
    # Helper heuristics
    # ---------------------------
    def _sample_confidence(self, sample_size: int) -> float:
        """
        Map sample size to a 0..1 confidence value (log scale).
        Small sample => low confidence. Large sample => capped at 1.0.
        """
        if sample_size <= 0:
            return 0.0
        # log1p scaling up to 100k
        return float(min(1.0, np.log1p(sample_size) / np.log1p(100000.0)))

    def _evidence_confidence(self, evidence: dict) -> float:
        """
        Heuristic confidence based on evidence presence and numeric stability.
        Favors evidence that contains impression counts, spend, revenue, or repeated confirmations.
        """
        if not evidence:
            return 0.5  # unknown but not fully absent

        score = 0.5
        # presence of count-like signals
        for k in ["impressions", "clicks", "revenue", "spend", "purchases", "b_impressions", "c_impressions"]:
            val = evidence.get(k)
            if val is not None:
                try:
                    num = float(val)
                    if num > 0:
                        score += 0.08
                except Exception:
                    pass

        # if evidence contains a correlation or slope and it's strong, boost confidence
        corr = evidence.get("correlation") or evidence.get("slope") or evidence.get("trend")
        if corr is not None:
            try:
                c = float(corr)
                # strong magnitude -> boost
                if abs(c) > 0.2:
                    score += 0.2
                elif abs(c) > 0.1:
                    score += 0.1
            except Exception:
                pass

        return float(min(1.0, score))

    def _detect_contradiction(self, h: Dict[str, Any]) -> Optional[str]:
        """
        Very conservative contradiction detection:
        - If hypothesis claims a 'decrease' but baseline < current, that's contradictory.
        - If sample size is zero but delta exists, suspicious.
        """
        metric = h.get("metric")
        delta_rel = h.get("delta_rel")
        baseline = h.get("baseline")
        current = h.get("current")
        sample = int(h.get("sample_size", 0) or 0)

        try:
            if delta_rel is not None and baseline is not None and current is not None:
                # if delta_rel negative (a drop) but baseline < current -> contradiction
                if delta_rel < 0 and float(baseline) < float(current):
                    return "delta_sign_mismatch"
                if delta_rel > 0 and float(baseline) > float(current):
                    return "delta_sign_mismatch"
            if sample == 0 and (delta_rel is not None and abs(delta_rel) > 0.01):
                return "zero_sample_but_delta"
        except Exception:
            return "contradiction_check_error"
        return None
