#!/usr/bin/env python3
"""
MetricsRecord Data Model

This module defines the MetricsRecord dataclass for representing unified
metrics data from different CSV formats.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

import json
import os

# Column name mappings for different CSV formats to standardize to MetricsRecord fields
COLUMN_MAPPINGS: Dict[str, str] = {
    # 1day format variations
    "timestamp": "started_at",  # 1day format uses 'timestamp' instead of 'started_at'
    # Common alternative spellings/capitalizations that might appear in CSV headers
    "chuteid": "chute_id",
    "invocationid": "invocation_id",
    "function": "function_name",
    "functionname": "function_name",
    "userid": "user_id",
    "inputtokens": "input_tokens",
    "outputtokens": "output_tokens",
    "startedat": "started_at",
    "completedat": "completed_at",
}

# load chutes_models.json
CHUTES_MODELS: Dict[str, Any] = {}
CHUTES_MODELS_MAP: Dict[str, str] = {}
CHUTES_MODELS_NAME_MAP: Dict[str, str] = {}
MODEL_CONFIG_FILE = (
    f"{os.path.dirname(os.path.abspath(__file__))}/../chutes_models.json"
)
if not os.path.exists(MODEL_CONFIG_FILE):
    raise FileNotFoundError(
        f"Chutes models configuration file not found: {MODEL_CONFIG_FILE}. "
        "Please run `bash scripts/chutes_api_log.sh` to download the file."
    )
with open(MODEL_CONFIG_FILE, "r", encoding="utf-8") as f:
    chutes_data = json.load(f)
    # The JSON has "items" array containing chute objects
    for item in chutes_data.get("items", []):
        chute_id = item["chute_id"]
        model_name = item["name"]
        CHUTES_MODELS[chute_id] = item
        CHUTES_MODELS_MAP[chute_id] = model_name
        CHUTES_MODELS_NAME_MAP[model_name] = chute_id


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

    # will use chutes_models.json to map chute_id to model_name
    model_name: Optional[str] = None

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
        if self.chute_id:
            self.model_name = CHUTES_MODELS_MAP.get(self.chute_id, self.chute_id)

    def __str__(self):
        return (
            f"MetricsRecord(chute_id={self.chute_id}, model_name={self.model_name}, "
            + f"input_tokens={self.input_tokens}, output_tokens={self.output_tokens}, "
            + f"ttft={self.ttft}, duration={self.duration}, completion_tps={self.completion_tps}"
            + f"started_at={self.started_at}, completed_at={self.completed_at})"
        )

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    print(
        MetricsRecord(
            chute_id="14a91d88-d6d6-5046-aaf4-eb3ad96b7247",
            input_tokens=100,
            output_tokens=200,
            started_at=1714857600,
            completed_at=1714857601,
            duration=1,
            completion_tps=100,
        )
    )
