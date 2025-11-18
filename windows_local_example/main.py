#!/usr/bin/env python
"""
CrewAI Local LLM Example
Demonstrates using local Ollama models on Windows 11

This example shows:
- Multiple agents with different local MoE models
- Task chaining and context passing
- Output file generation
- Performance optimized for GTX 5080
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import crewai
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@CrewBase
class LocalLLMCrew:
    """Local LLM Research and Writing Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        """Senior Research Analyst using Qwen2.5:32b"""
        return Agent(
            config=self.agents_config["researcher"],
            verbose=True,
        )

    @agent
    def analyst(self) -> Agent:
        """Data Analysis Expert using DeepSeek-R1:14b"""
        return Agent(
            config=self.agents_config["analyst"],
            verbose=True,
        )

    @agent
    def writer(self) -> Agent:
        """Professional Content Writer using Phi4:14b"""
        return Agent(
            config=self.agents_config["writer"],
            verbose=True,
        )

    @agent
    def reviewer(self) -> Agent:
        """Quality Assurance Specialist using Llama3.2:3b"""
        return Agent(
            config=self.agents_config["reviewer"],
            verbose=True,
        )

    @task
    def research_task(self) -> Task:
        """Research task for gathering information"""
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def analysis_task(self) -> Task:
        """Analysis task for data insights"""
        return Task(
            config=self.tasks_config["analysis_task"],
        )

    @task
    def writing_task(self) -> Task:
        """Writing task for content creation"""
        return Task(
            config=self.tasks_config["writing_task"],
        )

    @task
    def review_task(self) -> Task:
        """Review task for quality assurance"""
        return Task(
            config=self.tasks_config["review_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Local LLM Crew"""
        return Crew(
            agents=self.agents,  # Automatically uses all @agent decorated methods
            tasks=self.tasks,  # Automatically uses all @task decorated methods
            process=Process.sequential,  # Tasks run in order
            verbose=True,
        )


def main():
    """
    Run the crew with a topic
    """
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CrewAI Local LLM Demo - Windows 11")
    print("=" * 60)
    print()
    print("Using local Ollama models:")
    print("  - Researcher: Qwen2.5:32b (powerful reasoning)")
    print("  - Analyst: DeepSeek-R1:14b (fast reasoning)")
    print("  - Writer: Phi4:14b (balanced performance)")
    print("  - Reviewer: Llama3.2:3b (quick validation)")
    print()
    print("=" * 60)
    print()

    # Check if Ollama is running
    import requests

    try:
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code == 200:
            print("✓ Ollama service is running")
            print(f"  Version: {response.json().get('version', 'unknown')}")
        else:
            print("⚠ Warning: Ollama service may not be running properly")
    except Exception as e:
        print(f"✗ Error: Cannot connect to Ollama at http://localhost:11434")
        print(f"  {e}")
        print()
        print("Please ensure Ollama is running:")
        print("  1. Open a command prompt")
        print("  2. Run: ollama serve")
        print()
        return

    print()
    print("=" * 60)
    print()

    # Get topic from user or use default
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter a topic to research (or press Enter for default): ").strip()
        if not topic:
            topic = "the latest developments in AI agents and multi-agent systems"

    print(f"\nTopic: {topic}")
    print()
    print("Starting crew execution...")
    print("This may take several minutes depending on model speed.")
    print("=" * 60)
    print()

    # Create and run the crew
    inputs = {"topic": topic}

    try:
        crew = LocalLLMCrew().crew()
        result = crew.kickoff(inputs=inputs)

        print()
        print("=" * 60)
        print("CREW EXECUTION COMPLETE!")
        print("=" * 60)
        print()
        print(result)
        print()
        print(f"Output saved to: {output_dir / 'final_article.md'}")
        print()

    except Exception as e:
        print()
        print("=" * 60)
        print("ERROR OCCURRED")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}")
        print()
        print("Common issues:")
        print("  1. Ollama not running: Run 'ollama serve'")
        print("  2. Model not installed: Run 'ollama pull qwen2.5:32b'")
        print("  3. Out of memory: Try smaller models or reduce concurrent tasks")
        print()
        raise


if __name__ == "__main__":
    main()
