"""SRS 3.1.8 handoff from connectors to M2's future dataset writer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ceynex.contracts.protocols import CrossValidatorProtocol, DQFlag
from ceynex.data.cleaning.cleaner import DataCleaner
from ceynex.data.cleaning.cross_validator import CrossValidator


@dataclass(frozen=True)
class DataQualityResult:
    """Clean records plus non-destructive cross-source discrepancy flags."""

    records: pd.DataFrame
    flags: list[DQFlag]

    def dq_flag_rows(self) -> pd.DataFrame:
        """Return rows ready for insertion into the frozen ``dq_flag`` table."""
        return CrossValidator.to_frame(self.flags)


class DataQualityPipeline:
    """Prepare fact-trade records for ``UnifiedDatasetWriter`` injection.

    M2's writer can call ``prepare`` once after connector mappings and before
    persistence, write ``result.records`` unchanged in row count, then write
    ``result.dq_flag_rows()`` separately.  The validator uses the shared
    ``CrossValidatorProtocol`` so it may be replaced in tests.
    """

    def __init__(
        self,
        *,
        cleaner: DataCleaner | None = None,
        validator: CrossValidatorProtocol | None = None,
    ) -> None:
        self.cleaner = cleaner or DataCleaner()
        self.validator = validator or CrossValidator()

    def prepare(
        self,
        records: pd.DataFrame,
        *,
        target_frequency: str | None = None,
    ) -> DataQualityResult:
        """Clean and validate records without dropping any original observation."""
        cleaned = self.cleaner.clean(records, target_frequency=target_frequency)
        flags = self.validator.cross_validate(cleaned)
        return DataQualityResult(records=cleaned, flags=flags)
