#!/usr/bin/env python
import asyncio
from crewai.routers.router import Route
from crewai.routers.router import Router

from {{folder_name}}.pipelines.pipeline_classifier import EmailClassifierPipeline
from {{folder_name}}.pipelines.pipeline_normal import NormalPipeline
from {{folder_name}}.pipelines.pipeline_urgent import UrgentPipeline

async def run():
    """
    Run the pipeline.
    """
    inputs = [
       {
        "email": """
            Subject: URGENT: Marketing Campaign Launch - Immediate Action Required
            Dear Team,
            I'm reaching out regarding our upcoming marketing campaign that requires your immediate attention and swift action. We're facing a critical deadline, and our success hinges on our ability to mobilize quickly.
            Key points:
            
            Campaign launch: 48 hours from now
            Target audience: 250,000 potential customers
            Expected ROI: 35% increase in Q3 sales
            
            What we need from you NOW:
            
            Final approval on creative assets (due in 3 hours)
            Confirmation of media placements (due by end of day)
            Last-minute budget allocation for paid social media push
            
            Our competitors are poised to launch similar campaigns, and we must act fast to maintain our market advantage. Delays could result in significant lost opportunities and potential revenue.
            Please prioritize this campaign above all other tasks. I'll be available for the next 24 hours to address any concerns or roadblocks.
            Let's make this happen!
            [Your Name]
            Marketing Director
            P.S. I'll be scheduling an emergency team meeting in 1 hour to discuss our action plan. Attendance is mandatory.
        """
       }
    ]

    pipeline_classifier = EmailClassifierPipeline().create_pipeline()
    pipeline_urgent = UrgentPipeline().create_pipeline()
    pipeline_normal = NormalPipeline().create_pipeline()

    router = Router(
        routes={
            "high_urgency": Route(
                condition=lambda x: x.get("urgency_score", 0) > 7,
                pipeline=pipeline_urgent
            ),
            "low_urgency": Route(
                condition=lambda x: x.get("urgency_score", 0) <= 7,
                pipeline=pipeline_normal
            )
        },
        default=pipeline_normal
    )

    pipeline = pipeline_classifier >> router

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