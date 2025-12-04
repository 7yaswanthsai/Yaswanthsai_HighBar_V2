import pandas as pd
import numpy as np
import time
import re
from datetime import timedelta


class SchemaError(Exception):
    pass


class DataAgent:
    """
    V2 Data Agent:
    - Strong schema validation
    - Schema drift detection
    - Type enforcement
    - Null-pattern checks
    - Baseline vs current dataset split
    - Segment-level aggregation
    - Structured summary outputs
    """

    EXPECTED_SCHEMA = {
        "campaign_name": str,
        "adset_name": str,
        "date": "datetime",
        "spend": float,
        "impressions": float,
        "clicks": float,
        "ctr": float,
        "purchases": float,
        "revenue": float,
        "roas": float,
        "creative_type": str,
        "creative_message": str,
        "audience_type": str,
        "platform": str,
        "country": str,
    }

    def __init__(self, csv_path, logger=None, config=None):
        self.csv_path = csv_path
        self.logger = logger
        self.config = config or {}
        self.df = None

    # ======================================================
    # LOAD + VALIDATION
    # ======================================================

    def load(self):
        t0 = time.time()
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise SchemaError(f"Failed to load CSV: {e}")

        # convert date early (safe)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # validate BEFORE type cleaning
        self._validate_schema(df)
        self._clean_types(df)

        self.df = df

        if self.logger:
            self.logger.info({
                "event": "data_loaded",
                "rows": len(df),
                "cols": list(df.columns),
                "time_sec": round(time.time() - t0, 3)
            })

        return df

    # ======================================================
    # SCHEMA VALIDATION + DRIFT
    # ======================================================

    def _normalize(self, col):
        return re.sub(r"[^a-z0-9]", "", str(col).lower())

    def _detect_drift(self, df_columns):
        mode = self.config.get("schema_drift_mode", "fail")

        expected = list(self.EXPECTED_SCHEMA.keys())
        actual = list(df_columns)

        missing = [c for c in expected if c not in actual]
        extra = [c for c in actual if c not in expected]

        # near-miss: same normalized name but different actual name
        norm_expected = {self._normalize(c): c for c in expected}
        norm_actual = {self._normalize(c): c for c in actual}

        near_miss = []
        for ncol, raw_col in norm_actual.items():
            if ncol in norm_expected and norm_expected[ncol] != raw_col:
                near_miss.append({
                    "expected": norm_expected[ncol],
                    "received": raw_col
                })

        severity = len(missing) / max(1, len(expected))

        details = {
            "missing": missing,
            "extra": extra,
            "near_miss": near_miss,
            "severity": severity
        }

        # CASE 1 — pristine
        if not missing and not near_miss:
            if extra and self.logger:
                self.logger.warning({
                    "event": "extra_columns",
                    "columns": extra
                })
            return

        # CASE 2 — missing columns
        if missing:
            if mode == "warn":
                if self.logger:
                    self.logger.warning({
                        "event": "schema_drift_warn",
                        "details": details
                    })
                return
            if mode == "fail":
                raise SchemaError(f"Schema drift detected: {details}")

        # CASE 3 — near-miss
        if near_miss:
            if mode == "fail":
                raise SchemaError(f"Schema near-miss drift detected: {details}")
            if self.logger:
                self.logger.warning({
                    "event": "schema_drift_warn",
                    "details": details
                })

    def _validate_schema(self, df):
        missing = [c for c in self.EXPECTED_SCHEMA if c not in df.columns]
        extra = [c for c in df.columns if c not in self.EXPECTED_SCHEMA]

        # Null-report
        null_report = df.isnull().mean().round(3).to_dict()
        severe_nulls = {c: r for c, r in null_report.items() if r > 0.5}

        if severe_nulls:
            raise SchemaError(f"Columns with >50% nulls found: {severe_nulls}")

        # If dates exist but all malformed → fail
        if "date" in df.columns and df["date"].isnull().all():
            raise SchemaError("All values in 'date' parsed to null")

        # Always run drift detector
        self._detect_drift(df.columns)

        if self.logger:
            self.logger.info({
                "event": "schema_validated",
                "missing": missing,
                "extra": extra,
                "null_report": null_report
            })

    # ======================================================
    # TYPE CLEANING
    # ======================================================

    def _clean_types(self, df):
        for col, expected in self.EXPECTED_SCHEMA.items():
            if col not in df.columns:
                continue

            if expected == float:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
                df[col] = df[col].replace([np.inf, -np.inf], 0.0)

            elif expected == str:
                df[col] = df[col].astype(str).replace("nan", "")

            elif expected == "datetime":
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # ======================================================
    # BASELINE / CURRENT SPLIT
    # ======================================================

    def split_baseline_current(self):
        if self.df is None:
            raise ValueError("Dataset not loaded")

        df = self.df.copy()
        df = df.sort_values("date")

        if df["date"].nunique() < 2:
            raise SchemaError("Not enough dates to create baseline/current split")

        window = self.config.get("sample_window_days", 30)

        max_date = df["date"].max()
        cutoff = max_date - timedelta(days=window)

        baseline = df[df["date"] < cutoff]
        current = df[df["date"] >= cutoff]

        # fallback for very small baseline
        if len(baseline) < 50:
            baseline = df.iloc[: len(df) // 2]
            current = df.iloc[len(df) // 2 :]

        if self.logger:
            self.logger.info({
                "event": "baseline_current_split",
                "baseline_rows": len(baseline),
                "current_rows": len(current),
                "cutoff": cutoff.strftime("%Y-%m-%d")
            })

        return baseline, current

    # ======================================================
    # SEGMENTATION + AGGREGATION
    # ======================================================

    def segment_analysis(self, baseline, current):
        """
        Produces aggregated metrics per segment.
        Segments dynamically selected based on column availability.
        """

        segment_cols = []
        for col in ["campaign_name", "adset_name", "platform", "country", "audience_type", "creative_type"]:
            if col in baseline.columns:
                segment_cols.append(col)

        if not segment_cols:
            segment_cols = ["campaign_name"]

        def agg(df):
            g = df.groupby(segment_cols).agg({
                "spend": "sum",
                "impressions": "sum",
                "clicks": "sum",
                "purchases": "sum",
                "revenue": "sum"
            })
            g["ctr"] = g["clicks"] / g["impressions"].replace(0, 1)
            g["roas"] = g["revenue"] / g["spend"].replace(0, 1)
            return g.reset_index()

        return agg(baseline), agg(current)

    # ======================================================
    # SUMMARY (unchanged)
    # ======================================================

    def summary(self):
        if self.df is None:
            raise ValueError("Dataset not loaded")

        df = self.df

        ts = df.groupby("date").agg({
            "spend": "sum",
            "impressions": "sum",
            "clicks": "sum",
            "purchases": "sum",
            "revenue": "sum",
        }).sort_index()

        ts["ctr"] = ts["clicks"] / ts["impressions"].replace(0, 1)
        ts["roas"] = ts["revenue"] / ts["spend"].replace(0, 1)

        cs = df.groupby("campaign_name").agg({
            "spend": "sum",
            "impressions": "sum",
            "clicks": "sum",
            "purchases": "sum",
            "revenue": "sum",
        })

        cs["ctr"] = cs["clicks"] / cs["impressions"].replace(0, 1)
        cs["roas"] = cs["revenue"] / cs["spend"].replace(0, 1)

        return {
            "timeseries": ts.reset_index().to_dict(orient="records"),
            "campaign_summary": cs.reset_index().to_dict(orient="records"),
            "schema": df.dtypes.astype(str).to_dict(),
        }
