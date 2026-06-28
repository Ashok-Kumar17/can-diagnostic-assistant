"""
Basic tests for the decoder and tools layer (the parts testable without an
LLM API key). Run with: pytest tests/
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from src.decoder import CANLogDecoder
from src.tools import CANSignalTools

DBC_PATH = "dbc/vehicle_demo.dbc"
LOG_PATH = "data/demo_drive.log"


@pytest.fixture(scope="module")
def decoder():
    return CANLogDecoder(DBC_PATH)


@pytest.fixture(scope="module")
def decoded(decoder):
    df = decoder.decode_to_dataframe(LOG_PATH)
    events = decoder.extract_dtc_events(LOG_PATH)
    return df, events


def test_decode_produces_rows(decoded):
    df, _ = decoded
    assert len(df) > 0
    assert set(df.columns) == {"timestamp", "message", "signal", "value"}


def test_known_signals_present(decoded):
    df, _ = decoded
    signals = set(df["signal"].unique())
    assert "ControllerTemp" in signals
    assert "MotorRPM" in signals


def test_dtc_event_extracted(decoded):
    _, events = decoded
    assert len(events) == 1
    assert events[0].code_name == "P0C31_CONTROLLER_OVERTEMP"
    assert events[0].status == "SET"
    assert events[0].source == "MotorCtrl"


def test_search_signals_finds_temp_signals(decoded):
    df, events = decoded
    tools = CANSignalTools(df, events)
    result = tools.search_signals("temp")
    assert "ControllerTemp" in result["matches"]
    assert "MotorTemp" in result["matches"]


def test_signal_stats_in_expected_range(decoded):
    df, events = decoded
    tools = CANSignalTools(df, events)
    stats = tools.get_signal_stats("ControllerTemp")
    assert stats["min"] >= 40
    assert stats["max"] <= 214
    assert stats["max"] >= stats["min"]


def test_signal_trace_respects_time_window(decoded):
    df, events = decoded
    tools = CANSignalTools(df, events)
    trace = tools.get_signal_trace("ControllerTemp", start_time=10, end_time=20)
    for point in trace["points"]:
        assert 10 <= point["t"] <= 20


def test_unknown_signal_returns_error_not_crash(decoded):
    df, events = decoded
    tools = CANSignalTools(df, events)
    result = tools.get_signal_stats("NotARealSignal")
    assert "error" in result


def test_temp_actually_climbs_before_fault(decoded):
    """Sanity check on the synthetic scenario itself: temp should be rising
    in the window leading up to the fault, not flat or falling -- otherwise
    the demo data wouldn't actually support a 'why did it overheat' answer."""
    df, events = decoded
    tools = CANSignalTools(df, events)
    fault_time = events[0].timestamp
    early = tools.get_signal_stats("ControllerTemp", start_time=fault_time - 20, end_time=fault_time - 15)
    late = tools.get_signal_stats("ControllerTemp", start_time=fault_time - 5, end_time=fault_time)
    assert late["mean"] > early["mean"]
