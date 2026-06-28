"""
tools.py

The set of functions the LLM agent is allowed to call. Each function operates
on the long-format DataFrame produced by decoder.CANLogDecoder, plus the list
of DTCEvents. Keeping these as plain, well-typed Python functions (rather than
baking query logic into prompts) means they're independently testable and the
agent's "reasoning" is always grounded in real computed values, never a
hallucinated number.

Each tool returns a small JSON-serializable dict -- this is what gets fed back
into the LLM's context after a tool call.
"""
from __future__ import annotations

import difflib
from typing import Optional

import pandas as pd

from src.decoder import DTCEvent


class CANSignalTools:
    def __init__(self, df: pd.DataFrame, dtc_events: list[DTCEvent]):
        self.df = df
        self.dtc_events = dtc_events

    # ---- Tool 1 -----------------------------------------------------------
    def search_signals(self, keyword: str, limit: int = 10) -> dict:
        """Fuzzy-search signal names in the DBC (e.g. 'temp' -> ControllerTemp, MotorTemp, ...)."""
        all_signals = sorted(self.df["signal"].unique())
        keyword_lower = keyword.lower()
        substring_matches = [s for s in all_signals if keyword_lower in s.lower()]
        if substring_matches:
            matches = substring_matches[:limit]
        else:
            matches = difflib.get_close_matches(keyword, all_signals, n=limit, cutoff=0.4)
        return {"keyword": keyword, "matches": matches}

    # ---- Tool 2 -----------------------------------------------------------
    def get_signal_trace(
        self,
        signal_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        max_points: int = 50,
    ) -> dict:
        """
        Returns a downsampled (timestamp, value) trace for a signal within a time
        window. Downsampling matters: a 60s log at 20Hz is 1200+ points for one
        signal -- dumping that into an LLM context is wasteful and the model
        doesn't reason better over more points, just burns tokens.
        """
        sub = self.df[self.df["signal"] == signal_name]
        if start_time is not None:
            sub = sub[sub["timestamp"] >= start_time]
        if end_time is not None:
            sub = sub[sub["timestamp"] <= end_time]
        sub = sub.sort_values("timestamp")

        if sub.empty:
            return {"signal": signal_name, "error": "no data found for this signal/time range"}

        if len(sub) > max_points:
            step = len(sub) // max_points
            sub = sub.iloc[::step]

        return {
            "signal": signal_name,
            "unit_note": "see DBC for units",
            "points": [
                {"t": round(float(r.timestamp), 3), "v": round(float(r.value), 3)}
                for r in sub.itertuples()
            ],
        }

    # ---- Tool 3 -----------------------------------------------------------
    def get_signal_stats(
        self,
        signal_name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> dict:
        """Min/max/mean/first/last for a signal over an optional time window."""
        sub = self.df[self.df["signal"] == signal_name]
        if start_time is not None:
            sub = sub[sub["timestamp"] >= start_time]
        if end_time is not None:
            sub = sub[sub["timestamp"] <= end_time]

        if sub.empty:
            return {"signal": signal_name, "error": "no data found for this signal/time range"}

        sub = sub.sort_values("timestamp")
        return {
            "signal": signal_name,
            "start_time": float(sub["timestamp"].iloc[0]),
            "end_time": float(sub["timestamp"].iloc[-1]),
            "min": round(float(sub["value"].min()), 3),
            "max": round(float(sub["value"].max()), 3),
            "mean": round(float(sub["value"].mean()), 3),
            "first_value": round(float(sub["value"].iloc[0]), 3),
            "last_value": round(float(sub["value"].iloc[-1]), 3),
            "sample_count": int(len(sub)),
        }

    # ---- Tool 4 -----------------------------------------------------------
    def get_dtc_faults(self) -> dict:
        """Returns every DTC (fault code) event found in the log, in order."""
        return {
            "fault_count": len(self.dtc_events),
            "faults": [
                {
                    "timestamp": e.timestamp,
                    "code_name": e.code_name,
                    "raw_code": e.code,
                    "status": e.status,
                    "source": e.source,
                }
                for e in self.dtc_events
            ],
        }

    # ---- Tool registry for the agent layer ---------------------------------
    def as_tool_specs(self) -> list[dict]:
        """
        JSON-schema tool specs in Anthropic/OpenAI function-calling format.
        Both APIs accept this same {name, description, input_schema} shape
        with trivial renaming, which is what makes the provider-agnostic
        agent client in agent.py possible.
        """
        return [
            {
                "name": "search_signals",
                "description": "Fuzzy-search available CAN signal names by keyword. Use this first if you're not sure of the exact signal name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "keyword": {"type": "string", "description": "Keyword to search for, e.g. 'temp', 'voltage'"},
                    },
                    "required": ["keyword"],
                },
            },
            {
                "name": "get_signal_trace",
                "description": "Get a time-series trace of values for one signal, optionally within a time window.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "signal_name": {"type": "string"},
                        "start_time": {"type": "number", "description": "Optional start time in seconds"},
                        "end_time": {"type": "number", "description": "Optional end time in seconds"},
                    },
                    "required": ["signal_name"],
                },
            },
            {
                "name": "get_signal_stats",
                "description": "Get min/max/mean/first/last statistics for one signal over an optional time window.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "signal_name": {"type": "string"},
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                    },
                    "required": ["signal_name"],
                },
            },
            {
                "name": "get_dtc_faults",
                "description": "Get every diagnostic trouble code (fault) event recorded in the log, with timestamp and source.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    def call(self, tool_name: str, tool_input: dict) -> dict:
        """Dispatch a tool call by name -- used by the agent loop."""
        fn = {
            "search_signals": self.search_signals,
            "get_signal_trace": self.get_signal_trace,
            "get_signal_stats": self.get_signal_stats,
            "get_dtc_faults": self.get_dtc_faults,
        }.get(tool_name)
        if fn is None:
            return {"error": f"unknown tool '{tool_name}'"}
        return fn(**(tool_input or {}))
