#!/usr/bin/env python
import asyncio
from {{folder_name}}.src.{{folder_name}}.pipeline import {{pipeline_name}}Pipeline


async def run():
    """
    Run the pipeline.
    """
    inputs = [
        {"topic": "AI LLMs"},
    ]
    await {{pipeline_name}}Pipeline().pipeline().kickoff(inputs=inputs)


if __name__ == "__main__":
    asyncio.run(run())