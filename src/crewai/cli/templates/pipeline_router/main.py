#!/usr/bin/env python
import asyncio
from {

from crewai.routers.router import Route{folder_name}}.pipeline import {{pipeline_name}}Pipeline

async def run():
    """
    Run the pipeline.
    """
    inputs = [
        {"topic": "AI wearables"},
    ]

    # TODO: Pull all of these out dynamically from the /pipelines folder
    pipeline_categorzie = {{pipeline_name}}Pipeline()
    pipeline_high_priority = {{pipeline_name}}Pipeline()
    pipeline_low_priority = {{pipeline_name}}Pipeline()

    router = Router(
        routes={
            "high_urgency": Route(
                condition=lambda x: x.get("urgency_score", 0) > 7,
                pipeline=pipeline_high_priority
            ),
            "low_urgency": Route(
                condition=lambda x: x.get("urgency_score", 0) <= 7,
                pipeline=pipeline_low_priority
            )
        },
        default=Pipeline(stages=[pipeline_low_priority]) 
    )

    pipeline = pipeline_categorzie >> router

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