"""
cli.py

Quick command-line entry point: loads the demo DBC + log, starts a chat loop.
Useful for fast iteration without spinning up the Streamlit UI.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...      # or OPENAI_API_KEY=sk-...
    python cli.py
"""
from src.decoder import CANLogDecoder
from src.tools import CANSignalTools
from src.agent import LLMAgent

DBC_PATH = "dbc/vehicle_demo.dbc"
LOG_PATH = "data/demo_drive.log"


def main():
    print(f"Loading {LOG_PATH} against {DBC_PATH} ...")
    decoder = CANLogDecoder(DBC_PATH)
    df = decoder.decode_to_dataframe(LOG_PATH)
    dtc_events = decoder.extract_dtc_events(LOG_PATH)
    print(f"Decoded {len(df)} signal samples, {len(dtc_events)} DTC event(s).\n")

    tools = CANSignalTools(df, dtc_events)
    agent = LLMAgent(tools)

    print("CAN Diagnostic Assistant ready. Ask a question (or 'quit').")
    print('Try: "Why did the controller overheat fault occur?"\n')

    while True:
        question = input("> ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        answer = agent.ask(question)
        print(f"\n{answer}\n")


if __name__ == "__main__":
    main()
