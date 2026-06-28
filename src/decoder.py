"""
decoder.py

Parses a candump-style CAN log file against a DBC and produces a tidy pandas
DataFrame of decoded signals: one row per (timestamp, message, signal, value).

This "long format" is deliberately chosen over one-column-per-signal because
messages arrive at different rates and not every signal is present at every
timestamp -- long format avoids a sparse, hard-to-reason-about wide table and
makes it trivial to filter/group by signal name for the agent tools layer.

Log line format expected (standard candump -L output):
    (1719500000.000000) can0 100#240C35034F550100
"""
import re
from dataclasses import dataclass

import cantools
import pandas as pd

LOG_LINE_RE = re.compile(
    r"^\(([\d.]+)\)\s+(\S+)\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)\s*$"
)


@dataclass
class DTCEvent:
    timestamp: float
    code: int
    code_name: str
    status: str  # "SET" or "CLEARED"
    source: str


class CANLogDecoder:
    """Decodes a candump-style .log file against a DBC file."""

    def __init__(self, dbc_path: str):
        self.db = cantools.database.load_file(dbc_path)
        self._id_to_message = {msg.frame_id: msg for msg in self.db.messages}

    def parse_log_lines(self, log_path: str):
        """Yields (timestamp: float, frame_id: int, data: bytes) for each line."""
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                m = LOG_LINE_RE.match(line)
                if not m:
                    continue
                ts_str, _channel, id_str, data_str = m.groups()
                timestamp = float(ts_str)
                frame_id = int(id_str, 16)
                data = bytes.fromhex(data_str) if data_str else b""
                yield timestamp, frame_id, data

    def decode_to_dataframe(self, log_path: str) -> pd.DataFrame:
        """
        Returns a long-format DataFrame with columns:
            timestamp, message, signal, value
        Rows where the frame_id isn't in the DBC are silently skipped (unknown
        traffic on the bus -- common in real logs, not an error condition).
        """
        rows = []
        for timestamp, frame_id, data in self.parse_log_lines(log_path):
            message = self._id_to_message.get(frame_id)
            if message is None:
                continue
            try:
                decoded = message.decode(data)
            except Exception:
                # Malformed/truncated frame -- skip rather than crash the whole parse
                continue
            for signal_name, value in decoded.items():
                rows.append((timestamp, message.name, signal_name, value))

        df = pd.DataFrame(rows, columns=["timestamp", "message", "signal", "value"])
        return df

    def extract_dtc_events(self, log_path: str, dtc_message_name: str = "DTC_REPORT"):
        """
        Returns a list of DTCEvent for every DTC_REPORT-style frame in the log.
        Resolves DTC_Code and DTC_Source through the DBC's VAL_ tables so the
        agent gets human-readable fault names, not raw integers.
        """
        message = self.db.get_message_by_name(dtc_message_name)
        events = []
        for timestamp, frame_id, data in self.parse_log_lines(log_path):
            if frame_id != message.frame_id:
                continue
            try:
                raw = message.decode(data, decode_choices=False)
                resolved = message.decode(data, decode_choices=True)
            except Exception:
                continue
            events.append(
                DTCEvent(
                    timestamp=timestamp,
                    code=int(raw.get("DTC_Code")),
                    code_name=str(resolved.get("DTC_Code")),
                    status=str(resolved.get("DTC_Status")),
                    source=str(resolved.get("DTC_Source")),
                )
            )
        return events

    def list_signals(self):
        """Returns {message_name: [signal_name, ...]} for everything in the DBC."""
        return {msg.name: [s.name for s in msg.signals] for msg in self.db.messages}
