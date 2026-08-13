"""Frozen shared contracts for CeyNex.

NEVER modify this package without 3-way approval (M1, M2, M3). Both teammates'
code is built on these shapes; a quiet change here breaks their build, not yours.
"""

from ceynex.contracts.evidence import Evidence, SourceId
from ceynex.contracts.forecast import ForecastPoint
from ceynex.contracts.protocols import (
    CrossValidatorProtocol,
    DataSourceConnector,
    DQFlag,
    ForecastModel,
    KnowledgeGraphClientProtocol,
    LLMReasoningClientProtocol,
    NullCrossValidator,
    SourceManifest,
)
from ceynex.contracts.state import (
    ALL_AGENTS,
    DEFAULT_AGENT,
    AgentName,
    AgentOutput,
    AgentState,
    Sector,
    failed_output,
    merge_agent_outputs,
    new_state,
)

__all__ = [
    "ALL_AGENTS",
    "DEFAULT_AGENT",
    "AgentName",
    "AgentOutput",
    "AgentState",
    "CrossValidatorProtocol",
    "DQFlag",
    "DataSourceConnector",
    "Evidence",
    "ForecastModel",
    "ForecastPoint",
    "KnowledgeGraphClientProtocol",
    "LLMReasoningClientProtocol",
    "NullCrossValidator",
    "Sector",
    "SourceId",
    "SourceManifest",
    "failed_output",
    "merge_agent_outputs",
    "new_state",
]
