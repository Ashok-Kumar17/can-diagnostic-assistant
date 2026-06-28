"""
app.py

Streamlit chat UI for the CAN Diagnostic Assistant.

Run with:
    export ANTHROPIC_API_KEY=sk-ant-...      # or OPENAI_API_KEY=sk-...
    streamlit run app.py
"""
import os

import streamlit as st

from src.decoder import CANLogDecoder
from src.tools import CANSignalTools
from src.agent import LLMAgent

st.set_page_config(page_title="CAN Diagnostic Assistant", page_icon="🚗", layout="wide")

st.title("🚗 CAN Bus Diagnostic Assistant")
st.caption(
    "Ask natural-language questions about a vehicle's CAN log. The model calls "
    "real signal-lookup tools rather than guessing -- every number it cites was "
    "actually retrieved from the decoded log."
)

with st.sidebar:
    st.header("Data Source")
    dbc_file = st.file_uploader("DBC file", type=["dbc"])
    log_file = st.file_uploader("CAN log file (candump -L format)", type=["log", "txt"])
    use_demo = st.checkbox("Use bundled demo data instead", value=not (dbc_file and log_file))

    st.divider()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if has_key:
        st.success("API key detected in environment.")
    else:
        st.error("No ANTHROPIC_API_KEY or OPENAI_API_KEY set. Set one and restart.")


@st.cache_resource(show_spinner="Decoding CAN log...")
def load_demo_data():
    decoder = CANLogDecoder("dbc/vehicle_demo.dbc")
    df = decoder.decode_to_dataframe("data/demo_drive.log")
    dtc_events = decoder.extract_dtc_events("data/demo_drive.log")
    return df, dtc_events


def load_uploaded_data(dbc_bytes: bytes, log_bytes: bytes):
    # cantools needs a real file path, so write the uploads to a temp location
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".dbc", delete=False, mode="wb") as f_dbc:
        f_dbc.write(dbc_bytes)
        dbc_path = f_dbc.name
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="wb") as f_log:
        f_log.write(log_bytes)
        log_path = f_log.name

    decoder = CANLogDecoder(dbc_path)
    df = decoder.decode_to_dataframe(log_path)
    dtc_events = decoder.extract_dtc_events(log_path)
    return df, dtc_events


if use_demo:
    df, dtc_events = load_demo_data()
    st.sidebar.info("Using bundled synthetic demo log (motor overtemp scenario).")
elif dbc_file and log_file:
    df, dtc_events = load_uploaded_data(dbc_file.getvalue(), log_file.getvalue())
else:
    st.warning("Upload both a DBC and a log file, or check 'Use bundled demo data'.")
    st.stop()

tools = CANSignalTools(df, dtc_events)

col1, col2, col3 = st.columns(3)
col1.metric("Decoded signal samples", len(df))
col2.metric("Unique signals", df["signal"].nunique())
col3.metric("DTC faults found", len(dtc_events))

if dtc_events:
    with st.expander("Fault codes in this log", expanded=True):
        for e in dtc_events:
            st.write(f"**{e.code_name}** at t={e.timestamp:.2f}s -- source: {e.source}, status: {e.status}")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask about this CAN log, e.g. 'Why did the overtemp fault occur?'")

if question:
    if not has_key:
        st.error("Set ANTHROPIC_API_KEY or OPENAI_API_KEY before asking questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Querying signals and reasoning..."):
            agent = LLMAgent(tools)
            answer = agent.ask(question)
            st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
