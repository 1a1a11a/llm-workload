#!/usr/bin/env python3
"""
MetricsRecord Data Model

This module defines the MetricsRecord dataclass for representing unified
metrics data from different CSV formats.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


# Column name mappings for different CSV formats to standardize to MetricsRecord fields
COLUMN_MAPPINGS: Dict[str, str] = {
    # 1day format variations
    'timestamp': 'started_at',  # 1day format uses 'timestamp' instead of 'started_at'

    # Common alternative spellings/capitalizations that might appear in CSV headers
    'chuteid': 'chute_id',
    'invocationid': 'invocation_id',
    'function': 'function_name',
    'functionname': 'function_name',
    'userid': 'user_id',
    'inputtokens': 'input_tokens',
    'outputtokens': 'output_tokens',
    'startedat': 'started_at',
    'completedat': 'completed_at',
}


@dataclass
class MetricsRecord:
    """
    Unified representation of a metrics record from either 1day or 30day traces.

    Fields are optional to accommodate different CSV formats.
    """

    # Common fields
    chute_id: str
    input_tokens: int
    output_tokens: int
    ttft: Optional[float] = None

    # missing in 1day format
    function_name: Optional[str] = None

    # missing in 1day format
    user_id: Optional[str] = None

    # also called timestamp in 1day format
    started_at: Optional[float] = None
    # missing in 1day format
    completed_at: Optional[float] = None
    # missing in 30day format but can be calculated from started_at and completed_at
    duration: Optional[float] = None

    # missing in 30day format but can be calculated from input_tokens and output_tokens
    completion_tps: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)"""
        return self.input_tokens + self.output_tokens

    def __post_init__(self):
        """Validate required fields after initialization"""
        if not self.chute_id:
            raise ValueError("chute_id is required")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("Token counts cannot be negative")
        if self.started_at and self.completed_at:
            self.duration = self.completed_at - self.started_at
        if (
            not self.completion_tps
            and self.duration is not None
            and self.input_tokens is not None
            and self.output_tokens is not None
        ):
            self.completion_tps = self.total_tokens / self.duration
