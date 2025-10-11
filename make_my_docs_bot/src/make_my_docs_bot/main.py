#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from make_my_docs_bot.crew import MakeMyDocsBot
import argparse
import os
import subprocess

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew, accepting a branch_name parameter from the CLI.
    """

    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Run MakeMyDocsBot crew.")
    parser.add_argument("--branch_name", default="main", help="Branch name to compare against main")
    args = parser.parse_args()
    branch_name = args.branch_name

    # Get parent directory of current working directory
    parent_dir = os.path.dirname(os.getcwd())

    if not os.path.exists(os.path.join(parent_dir, ".git")):
        raise Exception(f"{parent_dir} is not a git repository")

    # Run git diff in parent directory
    cmd = ["git", "diff", "--name-only", f"main...{branch_name}", "--", "*.mdx"]
    result = subprocess.run(cmd, cwd=parent_dir, capture_output=True, text=True)
    changed_files = [
        f.strip()
        for f in result.stdout.splitlines()
        if f.strip() and f.startswith("docs/en/")
    ]

    print(f"Changed .mdx files under docs/en/: {changed_files}")

    # Prepare inputs
    inputs = []

    for changed_file in changed_files:
        inputs.append({
        "branch_name": branch_name,
        "file_path": changed_file
    })
    
    if len(inputs):
        print(f"Final inputs to the crew kickoff for each {inputs}")
        

        try:
            print(f"Executing crew for branch: {branch_name}")
            MakeMyDocsBot().crew().kickoff_for_each(inputs = inputs)
            print(f"Crew executed for branch: {branch_name}")
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")
    else:
        print(f"No inputs for crew to run, run cancelled")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        MakeMyDocsBot().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        MakeMyDocsBot().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    
    try:
        MakeMyDocsBot().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
