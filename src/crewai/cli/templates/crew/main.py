#!/usr/bin/env python
import sys
import json
import warnings

from {{folder_name}}.crew import {{crew_name}}
from crewai.utilities.llm_utils import create_llm

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew, allowing CLI overrides for required inputs.
    Usage example:
        uv run run_crew -- --topic="New Topic" --some_other_field="Value"
    """
    # Default inputs
    inputs = {
        'topic': 'AI LLMs'
        # Add any other default fields here
    }

    # 1) Gather overrides from sys.argv
    #    sys.argv might look like: ['run_crew', '--topic=NewTopic']
    #    But be aware that if you're calling "uv run run_crew", sys.argv might have
    #    additional items. So we typically skip the first 1 or 2 items to get only overrides.
    overrides = parse_cli_overrides(sys.argv[1:])

    # 2) Merge the overrides into defaults
    inputs.update(overrides)

    # 3) Kick off the crew with final inputs
    try:
        {{crew_name}}().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        {{crew_name}}().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        {{crew_name}}().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        {{crew_name}}().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def fetch_inputs():
    """
    Command that gathers required placeholders/inputs from the Crew, then
    prints them as JSON to stdout so external scripts can parse them easily.
    """
    try:
        crew = {{crew_name}}().crew()
        crew_inputs = crew.fetch_inputs()
        json_string = json.dumps(list(crew_inputs))
        print(json_string)
    except Exception as e:
        raise Exception(f"An error occurred while fetching inputs: {e}")

def fetch_chat_llm():
    """
    Command that fetches the 'chat_llm' property from the Crew,
    instantiates it via create_llm(),
    and prints the resulting LLM as JSON (using LLM.to_dict()) to stdout.
    """
    try:
        crew = {{crew_name}}().crew()
        raw_chat_llm = getattr(crew, "chat_llm", None)

        if not raw_chat_llm:
            # If the crew doesn't have chat_llm, fallback to create_llm(None)
            final_llm = create_llm(None)
        else:
            # raw_chat_llm might be a dict, or an LLM, or something else
            final_llm = create_llm(raw_chat_llm)

        if final_llm:
            # Print the final LLM as JSON, so fetch_chat_llm.py can parse it
            from crewai.llm import LLM  # Import here to avoid circular references

            # Make sure it's an instance of the LLM class:
            if isinstance(final_llm, LLM):
                print(json.dumps(final_llm.to_dict()))
            else:
                # If somehow it's not an LLM, try to interpret as a dict
                # or revert to an empty fallback
                if isinstance(final_llm, dict):
                    print(json.dumps(final_llm))
                else:
                    print(json.dumps({}))
        else:
            print(json.dumps({}))
    except Exception as e:
        raise Exception(f"An error occurred while fetching chat LLM: {e}")

# TODO: Talk to Joao about making using LLM calls to analyze the crew
# and generate all of this information automatically
def fetch_chat_inputs():
    """
    Command that fetches the 'chat_inputs' property from the Crew,
    and prints it as JSON to stdout.
    """
    try:
        crew = {{crew_name}}().crew()
        raw_chat_inputs = getattr(crew, "chat_inputs", None)

        if raw_chat_inputs:
            # Convert to dictionary to print JSON
            print(json.dumps(raw_chat_inputs.model_dump()))
        else:
            # If crew.chat_inputs is None or empty, print an empty JSON
            print(json.dumps({}))
    except Exception as e:
        raise Exception(f"An error occurred while fetching chat inputs: {e}")
    
    
def parse_cli_overrides(args_list) -> dict:
    """
    Parse arguments in the form of --key=value from a list of CLI arguments.
    Return them as a dict. For example:
    ['--topic=AI LLMs', '--username=John'] => {'topic': 'AI LLMs', 'username': 'John'}
    """
    overrides = {}
    for arg in args_list:
        if arg.startswith("--"):
            # remove the leading --
            trimmed = arg[2:]
            if "=" in trimmed:
                key, val = trimmed.split("=", 1)
                overrides[key] = val
            else:
                # If someone passed something like --topic (no =),
                # either handle differently or ignore
                pass
    return overrides
