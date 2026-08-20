"""Implements SRS 3.1.8 — cross-source validation that flags, never drops."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from ceynex.contracts.protocols import DQFlag


class CrossValidator:
    """Compare overlapping fact-trade observations from different sources.

    Validation is read-only: ``cross_validate`` returns discrepancy objects and
    leaves the supplied DataFrame untouched.  The caller can persist
    ``to_frame(flags)`` to the frozen ``dq_flag`` table after writing both
    original source records to ``fact_trade``.
    """

    _METRICS = ("export_volume", "export_value_usd", "price", "fx_usd_lkr")
    _KEY_COLUMNS = ("item", "hs_code", "partner_iso3", "period_start")

    def __init__(self, *, metrics: tuple[str, ...] | None = None) -> None:
        self.metrics = metrics or self._METRICS

    def cross_validate(self, records: pd.DataFrame) -> list[DQFlag]:
        """Return flags for pairwise source disagreements in overlapping records.

        A record overlaps when it has the same item, partner, period and metric
        value column as another source.  The percentage difference is measured
        against ``source_a``: ``abs(a-b)/abs(a)*100``.  Exact agreement and
        pairs with two zeroes are not flagged; zero versus non-zero is severe.
        """
        required = {"source_id", "item", "period_start"}
        missing = required.difference(records.columns)
        if missing:
            raise ValueError(f"Records missing required columns: {sorted(missing)}")

        flags: list[DQFlag] = []
        available_keys = [column for column in self._KEY_COLUMNS if column in records.columns]
        frame = records.copy()
        frame["period_start"] = pd.to_datetime(frame["period_start"], errors="raise")
        for metric in self.metrics:
            if metric not in frame.columns:
                continue
            observed = frame.loc[frame[metric].notna(), [*available_keys, "source_id", metric]].copy()
            observed[metric] = pd.to_numeric(observed[metric], errors="raise")
            for _, group in observed.groupby(available_keys, dropna=False, sort=True):
                # A source may have duplicate raw observations.  Do not compare a
                # source with itself; retain its first record deterministically.
                source_values = group.drop_duplicates(subset="source_id", keep="first")
                for (_, left), (_, right) in combinations(source_values.iterrows(), 2):
                    if left["source_id"] == right["source_id"]:
                        continue
                    pct_diff = self._pct_diff(float(left[metric]), float(right[metric]))
                    if pct_diff == 0.0:
                        continue
                    flags.append(
                        DQFlag(
                            item=str(left["item"]),
                            metric=metric,
                            source_a=str(left["source_id"]),
                            value_a=float(left[metric]),
                            source_b=str(right["source_id"]),
                            value_b=float(right[metric]),
                            pct_diff=pct_diff,
                            severity=self._severity(pct_diff),
                            hs_code=self._optional_string(left, "hs_code"),
                            partner_iso3=self._optional_string(left, "partner_iso3"),
                            period_start=pd.Timestamp(left["period_start"]).date().isoformat(),
                        )
                    )
        return flags

    @staticmethod
    def to_frame(flags: list[DQFlag]) -> pd.DataFrame:
        """Return rows matching the frozen ``dq_flag`` table's data columns."""
        columns = [
            "item", "hs_code", "partner_iso3", "period_start", "metric",
            "source_a", "value_a", "source_b", "value_b", "pct_diff", "severity",
        ]
        return pd.DataFrame(
            [{column: getattr(flag, column) for column in columns} for flag in flags],
            columns=columns,
        )

    @staticmethod
    def _pct_diff(value_a: float, value_b: float) -> float:
        if value_a == value_b:
            return 0.0
        if value_a == 0.0:
            return float("inf")
        return abs(value_a - value_b) / abs(value_a) * 100

    @staticmethod
    def _severity(pct_diff: float) -> str:
        if pct_diff < 5:
            return "minor"
        if pct_diff <= 20:
            return "material"
        return "severe"

    @staticmethod
    def _optional_string(record: pd.Series, column: str) -> str | None:
        if column not in record.index or pd.isna(record[column]):
            return None
        return str(record[column])
