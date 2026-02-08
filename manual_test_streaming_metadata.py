"""Manual test script to verify streaming chunks contain task metadata.

This script reproduces the issue from #4347 and verifies the fix.
Run this to confirm that StreamChunk objects now have non-empty task_name and task_id.
"""

import asyncio
import os

from crewai import Agent, Crew, Process, Task, LLM
from dotenv import load_dotenv

load_dotenv()


async def main():
    # Use a simple model (can use Ollama locally or OpenAI)
    llm = LLM(
        model=os.getenv("TEST_MODEL", "gpt-4o-mini"),
    )

    agent = Agent(
        role="Scientific Vulgarizer",
        goal="Translate complex scientific concepts into simple, everyday language that anyone can understand.",
        backstory="You're an expert at breaking down complicated ideas into clear and relatable explanations.",
        llm=llm,
        allow_delegation=False,
    )

    task = Task(
        name="Explain_Quantum_Computing",
        description="Provide a simple explanation of quantum computing for a general audience.",
        expected_output="A clear and concise explanation of quantum computing in layman's terms.",
        agent=agent,
        allow_crewai_trigger_context=True,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        memory=False,
        verbose=False,
        cache=False,
        share_crew=False,
        tracing=False,
        stream=True,
        process=Process.sequential,
    )

    print("Starting streaming crew...")
    print("=" * 80)

    streaming = await crew.kickoff_async()

    # Track metadata from first chunk
    first_chunk = None
    chunk_count = 0

    # Async iteration over chunks
    async for chunk in streaming:
        chunk_count += 1
        if first_chunk is None and chunk.content:
            first_chunk = chunk
            print(f"\nFirst chunk metadata:")
            print(f"  task_name: {chunk.task_name!r}")
            print(f"  task_id: {chunk.task_id!r}")
            print(f"  task_index: {chunk.task_index}")
            print(f"  agent_role: {chunk.agent_role!r}")
            print(f"  agent_id: {chunk.agent_id!r}")
            print(f"  chunk_type: {chunk.chunk_type}")
            print()

    print("=" * 80)
    print(f"\nTotal chunks received: {chunk_count}")

    # Verify the fix
    if first_chunk:
        print("\n✓ Verification Results:")
        if first_chunk.task_name:
            print(f"  ✓ task_name is populated: {first_chunk.task_name!r}")
        else:
            print(f"  ✗ task_name is EMPTY (BUG NOT FIXED)")

        if first_chunk.task_id:
            print(f"  ✓ task_id is populated: {first_chunk.task_id!r}")
        else:
            print(f"  ✗ task_id is EMPTY (BUG NOT FIXED)")

        if first_chunk.agent_role:
            print(f"  ✓ agent_role is populated: {first_chunk.agent_role!r}")

        print(f"\n  Result preview: {streaming.result[:100]}...")
    else:
        print("  ✗ No chunks received!")


if __name__ == "__main__":
    asyncio.run(main())
