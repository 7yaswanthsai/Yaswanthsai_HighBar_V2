import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from typing import Dict, Any, List, Optional


def _safe_div(num, den, eps=1e-9):
    try:
        return float(num) / (float(den) + eps)
    except Exception:
        return 0.0


class InsightAgent:
    """
    InsightAgent V2

    Inputs:
      - baseline: aggregated DataFrame (one row per segment)
      - current: aggregated DataFrame (same schema as baseline)
      - raw_df (optional): original row-level DataFrame (for trend checks)
      - config: configuration dict for thresholds, top_k, etc.
      - logger: optional logger with .info/.warning/.error

    Output:
      - List of hypothesis dicts, each has:
        {
          "id": str,
          "segment": { <segment key values> },
          "metric": "ctr" | "roas" | "spend" | ...,
          "baseline": ...,
          "current": ...,
          "delta_abs": ...,
          "delta_rel": ...,
          "impact": "low|medium|high",
          "confidence": 0.0-1.0,
          "evidence": { ... numeric evidence ... },
          "hypothesis": "human readable explanation",
          "explain": "how we scored impact/confidence"
        }
    """

    DEFAULT_CONFIG = {
        "min_sample_size": 50,
        "relative_drop_threshold": 0.15,  # 15% relative change to consider
        "absolute_drop_thresholds": {"ctr": 0.005, "roas": 0.1},  # small absolute thresholds
        "impact_weights": {"delta_rel": 0.7, "sample_size": 0.3},
        "confidence_weights": {"sample_size": 0.6, "variance": 0.4},
        "top_k": 5,
        "trend_slope_threshold": -0.005,
        "frequency_threshold": 3.0,  # scaled frequency threshold
    }

    def __init__(
        self,
        baseline: Optional[pd.DataFrame] = None,
        current: Optional[pd.DataFrame] = None,
        raw_df: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        logger=None,
    ):
        self.baseline = baseline
        self.current = current
        self.raw_df = raw_df
        self.logger = logger
        cfg = config or {}
        self.config = {**self.DEFAULT_CONFIG, **cfg}

        # Ensure baseline/current exist as copies
        if self.baseline is not None:
            self.baseline = self.baseline.copy()
        if self.current is not None:
            self.current = self.current.copy()
        if self.raw_df is not None:
            self.raw_df = self.raw_df.copy()
            if "date" in self.raw_df.columns:
                self.raw_df["date"] = pd.to_datetime(self.raw_df["date"], errors="coerce")

    # -------------------------
    # Public API
    # -------------------------
    def run(self) -> List[Dict[str, Any]]:
        """
        Generate and rank hypotheses from baseline/current (and raw_df if present).
        """
        if self.baseline is None or self.current is None:
            raise ValueError("baseline and current aggregated dataframes are required")

        # Align segments (merge on common segment columns)
        seg_cols = self._find_segment_columns()
        merged = self._merge_baseline_current(seg_cols)

        # Compute deltas and candidate flags
        merged = self._compute_deltas(merged)

        # Build candidates from deltas
        candidates = self._candidates_from_deltas(merged, seg_cols)

        # Add heuristic-based candidates from raw_df (trend, roas-spend, frequency)
        if self.raw_df is not None:
            candidates += self._candidates_from_trends(seg_cols)
            roas_spend = self._roas_spend_correlation()
            if roas_spend is not None:
                candidates.append(roas_spend)
            freq_cands = self._frequency_candidates()
            candidates += freq_cands

        # Score candidates (impact + confidence)
        scored = [self._score_candidate(cand, merged, seg_cols) for cand in candidates]

        # Deduplicate by hypothesis id (keep highest score)
        dedup = self._deduplicate(scored)

        # Rank and return top_k
        ranked = sorted(dedup, key=lambda x: (self._impact_value(x["impact"]), x["confidence"]), reverse=True)
        top_k = ranked[: self.config.get("top_k", 5)]

        if self.logger:
            self.logger.info({"event": "insights_generated", "candidates": len(dedup), "returned": len(top_k)})

        return top_k

    # -------------------------
    # Helpers: segments / merge
    # -------------------------
    def _find_segment_columns(self) -> List[str]:
        # determine intersection of columns used for segmentation
        baseline_cols = set(self.baseline.columns)
        current_cols = set(self.current.columns)
        common = list(baseline_cols.intersection(current_cols))

        # prefer known segment keys if present
        pref = ["campaign_name", "adset_name", "platform", "country", "audience_type", "creative_type"]
        seg_cols = [c for c in pref if c in common]
        if not seg_cols:
            # fallback to first non-metric columns
            non_metric = [c for c in common if c not in {"spend", "impressions", "clicks", "purchases", "revenue", "ctr", "roas", "date"}]
            seg_cols = non_metric[:1] if non_metric else []
        return seg_cols

    def _merge_baseline_current(self, seg_cols: List[str]) -> pd.DataFrame:
        # suffix baseline/current as _b / _c
        b = self.baseline.copy()
        c = self.current.copy()

        b = b.add_prefix("b_")
        c = c.add_prefix("c_")

        # rename segment columns back to unified names
        for col in seg_cols:
            b = b.rename(columns={f"b_{col}": col})
            c = c.rename(columns={f"c_{col}": col})

        merge_on = seg_cols if seg_cols else []
        if merge_on:
            merged = pd.merge(b, c, on=merge_on, how="outer", suffixes=("_b", "_c"))
        else:
            # no segment columns, use cross-join fallback and aggregate numeric totals
            merged = pd.concat([b, c], axis=1)

        # fill NaNs for numeric columns
        for col in merged.columns:
            if merged[col].dtype.kind in "fi":
                merged[col] = merged[col].fillna(0.0)

        return merged

    # -------------------------
    # Delta computations
    # -------------------------
    def _compute_deltas(self, merged: pd.DataFrame) -> pd.DataFrame:
        # expects cols like b_spend, c_spend, b_clicks, c_clicks, etc.
        merged = merged.copy()
        metrics = ["spend", "impressions", "clicks", "purchases", "revenue", "ctr", "roas"]

        for m in metrics:
            bcol = f"b_{m}"
            ccol = f"c_{m}"
            if bcol in merged.columns and ccol in merged.columns:
                merged[f"delta_abs_{m}"] = merged[ccol] - merged[bcol]
                merged[f"delta_rel_{m}"] = merged.apply(
                    lambda r: _safe_div(r[ccol] - r[bcol], r[bcol]) if r[bcol] != 0 else (np.nan if (r[ccol] == 0) else np.inf),
                    axis=1,
                )
            else:
                # if metric missing on either side, skip
                merged[f"delta_abs_{m}"] = np.nan
                merged[f"delta_rel_{m}"] = np.nan

        # Add sample size estimate: use impressions or sum of rows if impressions not present
        if "b_impressions" in merged.columns and "c_impressions" in merged.columns:
            merged["sample_size"] = merged[["b_impressions", "c_impressions"]].sum(axis=1)
        else:
            # fallback to spend based proxy
            merged["sample_size"] = merged[[c for c in merged.columns if "spend" in c]].sum(axis=1).fillna(0.0)

        return merged

    # -------------------------
    # Candidate construction from deltas
    # -------------------------
    def _candidates_from_deltas(self, merged: pd.DataFrame, seg_cols: List[str]) -> List[Dict[str, Any]]:
        cands = []
        # iterate rows
        for _, row in merged.iterrows():
            # form segment dict
            segment = {col: row.get(col, None) for col in seg_cols} if seg_cols else {}

            # check metrics of interest
            for metric in ["ctr", "roas", "spend", "impressions"]:
                da = row.get(f"delta_abs_{metric}", None)
                dr = row.get(f"delta_rel_{metric}", None)
                sample = row.get("sample_size", 0)

                # skip small samples
                if sample < self.config["min_sample_size"]:
                    continue

                # relative drop check
                if pd.notnull(dr) and dr < -self.config["relative_drop_threshold"]:
                    cands.append({
                        "id": f"delta_{metric}_{hash(tuple(segment.items()))}_{int(abs(dr)*10000)}",
                        "segment": segment,
                        "metric": metric,
                        "baseline": float(row.get(f"b_{metric}", np.nan)),
                        "current": float(row.get(f"c_{metric}", np.nan)),
                        "delta_abs": float(da) if pd.notnull(da) else None,
                        "delta_rel": float(dr) if pd.notnull(dr) else None,
                        "sample_size": int(sample) if not np.isnan(sample) else 0,
                        "source": "delta_analysis",
                        "raw_row": row.to_dict()
                    })

                # absolute drop thresholds (for small baselines)
                abs_thresh = self.config.get("absolute_drop_thresholds", {}).get(metric, None)
                if abs_thresh is not None and pd.notnull(da) and da < -abs_thresh:
                    cands.append({
                        "id": f"absdelta_{metric}_{hash(tuple(segment.items()))}_{int(abs(da))}",
                        "segment": segment,
                        "metric": metric,
                        "baseline": float(row.get(f"b_{metric}", np.nan)),
                        "current": float(row.get(f"c_{metric}", np.nan)),
                        "delta_abs": float(da),
                        "delta_rel": float(dr) if pd.notnull(dr) else None,
                        "sample_size": int(sample) if not np.isnan(sample) else 0,
                        "source": "abs_delta_analysis",
                        "raw_row": row.to_dict()
                    })

        return cands

    # -------------------------
    # Heuristic candidates from raw_df
    # -------------------------
    def _candidates_from_trends(self, seg_cols: List[str]) -> List[Dict[str, Any]]:
        cands = []
        df = self.raw_df
        by = seg_cols[0] if seg_cols else None
        if by is None:
            # fallback to campaign_name if present
            by = "campaign_name" if "campaign_name" in df.columns else None

        if by is None:
            return cands

        # compute trend per segment for ctr
        for key, group in df.groupby(by):
            g = group.sort_values("date")
            if g.shape[0] < 4:
                continue
            y = g["ctr"].fillna(0.0).values
            if len(np.unique(y)) <= 1:
                continue
            X = np.arange(len(y)).reshape(-1, 1)
            try:
                model = LinearRegression().fit(X, y)
            except Exception:
                continue
            slope = float(model.coef_[0])
            mean_ctr = float(np.mean(y))
            n = len(y)

            if slope < self.config.get("trend_slope_threshold", -0.005):
                cands.append({
                    "id": f"trend_ctr_{by}_{key}",
                    "segment": {by: key},
                    "metric": "ctr",
                    "trend_slope": slope,
                    "baseline": None,
                    "current": None,
                    "delta_abs": None,
                    "delta_rel": slope,
                    "sample_size": n,
                    "source": "trend",
                    "hypothesis": "Falling CTR trend suggests creative fatigue or audience saturation",
                    "evidence": {"slope": slope, "mean_ctr": mean_ctr, "n": n}
                })

        return cands

    def _roas_spend_correlation(self) -> Optional[Dict[str, Any]]:
        # correlation on date-aggregated values
        df = self.raw_df
        if "date" not in df.columns:
            return None
        t = df.groupby("date").agg({"spend": "sum", "revenue": "sum"}).reset_index()
        if len(t) < 4:
            return None
        t["roas"] = t["revenue"] / t["spend"].replace(0, np.nan)
        if t["roas"].isnull().all():
            return None
        corr = float(t["roas"].corr(t["spend"]))
        if pd.isnull(corr):
            return None
        if corr < -0.15:
            return {
                "id": "roas_spend_negative",
                "segment": {},
                "metric": "roas",
                "baseline": None,
                "current": None,
                "delta_abs": None,
                "delta_rel": corr,
                "sample_size": int(t["spend"].sum()),
                "source": "roas_spend_corr",
                "hypothesis": "Increasing spend correlates with decreasing ROAS",
                "evidence": {"correlation": corr}
            }
        return None

    def _frequency_candidates(self) -> List[Dict[str, Any]]:
        df = self.raw_df
        cands = []
        if "campaign_name" not in df.columns:
            return cands

        for campaign, g in df.groupby("campaign_name"):
            days = max(1, (g["date"].max() - g["date"].min()).days + 1)
            impressions = g["impressions"].sum()
            clicks = g["clicks"].sum()
            frequency = impressions / (days * 1000.0)  # scaled
            ctr = _safe_div(clicks, impressions)
            if frequency > self.config.get("frequency_threshold", 3.0) and ctr < 0.01:
                cands.append({
                    "id": f"freq_fatigue_{campaign}",
                    "segment": {"campaign_name": campaign},
                    "metric": "frequency",
                    "baseline": None,
                    "current": None,
                    "delta_abs": None,
                    "delta_rel": None,
                    "sample_size": int(impressions),
                    "source": "frequency",
                    "hypothesis": "High delivery frequency coupled with low CTR indicates audience fatigue",
                    "evidence": {"frequency": float(frequency), "ctr": float(ctr)}
                })
        return cands

    # -------------------------
    # Scoring
    # -------------------------
    def _score_candidate(self, cand: Dict[str, Any], merged_df: pd.DataFrame, seg_cols: List[str]) -> Dict[str, Any]:
        """
        Compute impact and confidence and return extended candidate.
        Simple explainable heuristics:
          - impact ~ weighted (abs(delta_rel), normalized sample_size)
          - confidence ~ weighted (sample_size, inverse variance)
        """
        # baseline numbers
        sample = cand.get("sample_size", 0)
        delta_rel = cand.get("delta_rel", None)
        delta_abs = cand.get("delta_abs", None)

        # compute impact score (0..1)
        imp_score = 0.0
        if delta_rel is not None and np.isfinite(delta_rel):
            imp_score = min(1.0, max(0.0, abs(delta_rel)))  # relative change magnitude
        # boost by sample size (log scale)
        sample_boost = min(1.0, np.log1p(sample) / np.log1p(10000))
        w = self.config.get("impact_weights", {"delta_rel": 0.7, "sample_size": 0.3})
        impact_raw = w["delta_rel"] * imp_score + w["sample_size"] * sample_boost

        # confidence: based on sample size + variance (if available)
        conf = 0.0
        var_penalty = 0.0
        # try to estimate variance from merged_df if raw_row present
        try:
            # if raw_row has b_impressions/c_impressions, use them to infer stability
            rr = cand.get("raw_row", None)
            if rr:
                b_imp = rr.get("b_impressions", 0) or 0
                c_imp = rr.get("c_impressions", 0) or 0
                total_imp = b_imp + c_imp
                # variance proxy: small sample size -> low confidence
                conf = min(1.0, np.log1p(total_imp) / np.log1p(10000))
        except Exception:
            conf = min(1.0, np.log1p(sample) / np.log1p(10000))

        # combine into human-friendly buckets
        impact_bucket = "low"
        if impact_raw > 0.6:
            impact_bucket = "high"
        elif impact_raw > 0.25:
            impact_bucket = "medium"

        # final confidence clamp
        conf = float(min(1.0, max(0.0, conf)))

        # friendly explanation
        explain = f"impact_raw={round(impact_raw,3)}, sample_boost={round(sample_boost,3)}, confidence={round(conf,3)}"

        # add fields and return
        cand_out = {
            "id": cand.get("id"),
            "segment": cand.get("segment", {}),
            "metric": cand.get("metric"),
            "baseline": cand.get("baseline"),
            "current": cand.get("current"),
            "delta_abs": cand.get("delta_abs"),
            "delta_rel": cand.get("delta_rel"),
            "sample_size": cand.get("sample_size", 0),
            "source": cand.get("source"),
            "hypothesis": cand.get("hypothesis", self._default_hypothesis_text(cand)),
            "evidence": cand.get("evidence", {}),
            "impact": impact_bucket,
            "impact_score": float(round(impact_raw, 3)),
            "confidence": float(round(conf, 3)),
            "explain": explain,
        }

        return cand_out

    def _default_hypothesis_text(self, cand: Dict[str, Any]) -> str:
        metric = cand.get("metric", "metric")
        seg = cand.get("segment") or {}
        seg_txt = ", ".join([f"{k}={v}" for k, v in seg.items()]) if seg else "overall"
        dr = cand.get("delta_rel")
        if dr is not None and np.isfinite(dr):
            pct = round(dr * 100, 1)
            return f"{metric.upper()} changed by {pct}% for {seg_txt}"
        return f"Issue detected on {metric} for {seg_txt}"

    # -------------------------
    # Utilities
    # -------------------------
    def _deduplicate(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best = {}
        for c in candidates:
            cid = c.get("id")
            if cid is None:
                cid = repr(c)[:64]
            prev = best.get(cid)
            if prev is None:
                best[cid] = c
            else:
                # keep higher confidence then impact_score
                if (c.get("confidence", 0) > prev.get("confidence", 0)) or (
                    c.get("impact_score", 0) > prev.get("impact_score", 0)
                ):
                    best[cid] = c
        return list(best.values())

    def _impact_value(self, bucket: str) -> int:
        return {"low": 0, "medium": 1, "high": 2}.get(bucket, 0)
