#!/usr/bin/env python
from {{folder_name}}.crew import {{crew_name}}Crew


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'topic': 'AI LLMs'
    }
    {{crew_name}}Crew().crew().kickoff(inputs=inputs)