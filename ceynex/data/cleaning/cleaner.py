"""Implements SRS 3.1.8 — cleaning, standardization, and time alignment."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


class DataCleaner:
    """Clean fact-trade-shaped records without deleting source observations.

    The cleaner is intentionally deterministic: it preserves original units,
    writes explicit conversion factors, never fills volume observations, and
    records every frequency-alignment rule in ``resampling_rule``.
    """

    _SRI_LANKA_NAMES = {"sri lanka", "lk", "lka"}
    _VOLUME_TO_KG = {"kg": 1.0, "kilogram": 1.0, "kilograms": 1.0, "t": 1000.0,
                     "tonne": 1000.0, "tonnes": 1000.0, "mt": 1000.0}
    _PRICE_TO_PER_KG = {
        "lkr/kg": ("LKR/kg", 1.0),
        "lkr/tonne": ("LKR/kg", 0.001),
        "lcu/tonne": ("LKR/kg", 0.001),
        "usd/kg": ("USD/kg", 1.0),
        "usd/tonne": ("USD/kg", 0.001),
    }
    _FREQUENCY_ALIASES = {"D": "D", "W": "W", "M": "M", "Q": "Q", "A": "Y", "Y": "Y"}

    def clean(self, records: pd.DataFrame, *, target_frequency: str | None = None) -> pd.DataFrame:
        """Return standardized records, optionally aligned to a target frequency.

        ``records`` uses the frozen ``fact_trade`` column names.  Extra columns
        are preserved.  A target frequency may be ``D``, ``W``, ``M``, ``Q`` or
        ``A``; price is averaged, volume summed, and FX uses the period-end
        observation.
        """
        cleaned = self.standardize_countries(records)
        cleaned = self.normalize_units(cleaned)
        cleaned = self.forward_fill_fx(cleaned)
        if target_frequency is not None:
            cleaned = self.resample(cleaned, target_frequency)
        return cleaned

    def standardize_countries(self, records: pd.DataFrame) -> pd.DataFrame:
        """Standardize Sri Lanka reporter/partner identifiers to LKA and M49 144."""
        frame = records.copy()
        for prefix in ("reporter", "partner"):
            iso_column, m49_column = f"{prefix}_iso3", f"{prefix}_m49"
            if iso_column not in frame.columns and m49_column not in frame.columns:
                continue
            if iso_column not in frame.columns:
                frame[iso_column] = pd.NA
            if m49_column not in frame.columns:
                frame[m49_column] = pd.NA

            iso = frame[iso_column].astype("string").str.strip().str.lower()
            m49 = pd.to_numeric(frame[m49_column], errors="coerce")
            sri_lanka = iso.isin(self._SRI_LANKA_NAMES) | m49.eq(144)
            frame.loc[sri_lanka, iso_column] = "LKA"
            frame.loc[sri_lanka, m49_column] = 144
        return frame

    def normalize_units(self, records: pd.DataFrame) -> pd.DataFrame:
        """Normalize mass volumes to kg and mass-denominated prices to per kg."""
        frame = records.copy()
        if "export_volume" in frame.columns:
            unit = self._normalized_text(frame.get("volume_unit"))
            factor = unit.map(self._VOLUME_TO_KG)
            frame["original_volume_unit"] = frame.get("volume_unit", pd.Series(pd.NA, index=frame.index))
            frame["volume_conversion_factor"] = factor
            convertible = frame["export_volume"].notna() & factor.notna()
            frame.loc[convertible, "export_volume"] = (
                pd.to_numeric(frame.loc[convertible, "export_volume"], errors="raise")
                * factor.loc[convertible]
            )
            frame.loc[convertible, "volume_unit"] = "kg"

        if "price" in frame.columns:
            unit = self._normalized_text(frame.get("price_unit"))
            mapping = unit.map(self._PRICE_TO_PER_KG)
            frame["original_price_unit"] = frame.get("price_unit", pd.Series(pd.NA, index=frame.index))
            frame["price_conversion_factor"] = mapping.map(
                lambda value: value[1] if isinstance(value, tuple) else pd.NA
            )
            convertible = frame["price"].notna() & mapping.notna()
            factors = frame.loc[convertible, "price_conversion_factor"].astype(float)
            frame.loc[convertible, "price"] = (
                pd.to_numeric(frame.loc[convertible, "price"], errors="raise") * factors
            )
            frame.loc[convertible, "price_unit"] = mapping.loc[convertible].map(lambda value: value[0])
        return frame

    def forward_fill_fx(self, records: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill daily FX values for at most five consecutive days."""
        if "fx_usd_lkr" not in records.columns or "period_start" not in records.columns:
            return records.copy()
        frame = records.copy()
        frequency = frame.get("frequency", pd.Series("", index=frame.index)).astype("string").str.upper()
        is_daily = frequency.eq("D")
        if not is_daily.any():
            return frame

        frame["period_start"] = pd.to_datetime(frame["period_start"], errors="raise")
        group_columns = self._available_group_columns(frame)
        daily = frame.loc[is_daily].sort_values([*group_columns, "period_start"])
        if group_columns:
            filled = daily.groupby(group_columns, dropna=False)["fx_usd_lkr"].ffill(limit=5)
        else:
            filled = daily["fx_usd_lkr"].ffill(limit=5)
        frame.loc[daily.index, "fx_usd_lkr"] = filled
        return frame

    def resample(self, records: pd.DataFrame, target_frequency: str) -> pd.DataFrame:
        """Align compatible records using price=mean, volume=sum, FX=period-end."""
        target = self._FREQUENCY_ALIASES.get(target_frequency.upper())
        if target is None:
            raise ValueError("target_frequency must be one of D, W, M, Q or A")
        if "period_start" not in records.columns:
            raise ValueError("period_start is required for frequency alignment")

        frame = records.copy()
        frame["period_start"] = pd.to_datetime(frame["period_start"], errors="raise")
        group_columns = self._available_group_columns(frame)
        period = frame["period_start"].dt.to_period(target)
        frame["_resample_period"] = period
        aggregation: dict[str, object] = {}
        for column in frame.columns:
            if column in {*group_columns, "_resample_period", "period_start", "period_end", "frequency"}:
                continue
            if column == "price":
                aggregation[column] = "mean"
            elif column == "export_volume":
                aggregation[column] = lambda values: values.sum(min_count=1)
            elif column == "fx_usd_lkr":
                aggregation[column] = self._period_end_value
            elif column == "resampling_rule":
                aggregation[column] = self._join_unique
            else:
                aggregation[column] = "first"

        result = frame.groupby([*group_columns, "_resample_period"], dropna=False, as_index=False).agg(aggregation)
        result["period_start"] = result["_resample_period"].dt.start_time
        result["period_end"] = result["_resample_period"].dt.end_time.dt.normalize()
        result["frequency"] = "A" if target == "Y" else target
        result["resampling_rule"] = "price=mean; volume=sum; fx=period-end"
        return result.drop(columns="_resample_period").sort_values([*group_columns, "period_start"], ignore_index=True)

    @staticmethod
    def _normalized_text(values: pd.Series | None) -> pd.Series:
        if values is None:
            return pd.Series(dtype="string")
        return values.astype("string").str.strip().str.lower()

    @staticmethod
    def _period_end_value(values: pd.Series) -> object:
        non_null = values.dropna()
        return non_null.iloc[-1] if not non_null.empty else pd.NA

    @staticmethod
    def _join_unique(values: Iterable[object]) -> str | pd.NA:
        unique = [str(value) for value in pd.unique(pd.Series(values).dropna())]
        return "; ".join(unique) if unique else pd.NA

    @staticmethod
    def _available_group_columns(frame: pd.DataFrame) -> list[str]:
        candidates = ["source_id", "item", "hs_code", "reporter_iso3", "reporter_m49", "partner_iso3", "partner_m49"]
        return [column for column in candidates if column in frame.columns]
