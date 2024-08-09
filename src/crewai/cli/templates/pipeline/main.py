#!/usr/bin/env python
import asyncio
from {{folder_name}}.pipelines.pipeline import {{pipeline_name}}Pipeline

async def run():
    """
    Run the pipeline.
    """
    inputs = [
        {"topic": "AI wearables"},
    ]
    pipeline = {{pipeline_name}}Pipeline()
    results = await pipeline.kickoff(inputs)
    
    # Process and print results
    for result in results:
        print(f"Raw output: {result.raw}")
        if result.json_dict:
            print(f"JSON output: {result.json_dict}")
        print("\n")

def main():
    asyncio.run(run())

if __name__ == "__main__":
    main()