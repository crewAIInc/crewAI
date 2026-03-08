#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from cryptocurrency_collaboration.crew import CryptocurrencyCollaboration

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    inputs = {
        "symbol": "ETHUSDT",
        "intervals": ["1m", "5m", "15m", "1d", "1w"],
        "time_range": "past_24h",
        'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        CryptocurrencyCollaboration().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    inputs = {
        "symbol": "ETHUSDT",
        "intervals": ["1m", "5m", "15m", "1d", "1w"],
        "time_range": "past_24h",
        'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    try:
        CryptocurrencyCollaboration().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        CryptocurrencyCollaboration().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    inputs = {
        "symbol": "ETHUSDT",
        "intervals": ["1m", "5m", "15m", "1d", "1w"],
        "time_range": "past_24h",
        'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        CryptocurrencyCollaboration().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "symbol": "",
        "intervals": [],
        "time_range": "",
        'current_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        result = CryptocurrencyCollaboration().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
