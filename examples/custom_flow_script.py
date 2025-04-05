"""
Example script showing how to run a CrewAI flow from a custom script.

This example demonstrates how to avoid the ModuleNotFoundError when 
starting flows from custom scripts outside of the CLI command context.
"""
import os

from crewai.utilities.path_utils import add_project_to_path

add_project_to_path()

from my_flow.main import MyFlow  # noqa: E402

def main():
    """Run the flow from a custom script."""
    flow = MyFlow()
    
    result = flow.kickoff()
    
    print(f"Flow completed with result: {result}")


if __name__ == "__main__":
    main()
