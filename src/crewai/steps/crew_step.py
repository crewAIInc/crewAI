import threading
from typing import Dict, Any, List, Optional
from crewai.crew import Crew
from crewai.steps.step import Step


class CrewStep(Step):
    def __init__(self, crew: Crew):
        self.crew = crew

    def kickoff(self, inputs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        results = []

        def kickoff_task(input_data: Dict[str, Any]):
            result = self.crew.kickoff(inputs=input_data)
            results.append({"result": result})

        if inputs is None:
            # Handle the case where inputs is None
            kickoff_task({})
        else:
            threads = []
            for input_data in inputs:
                thread = threading.Thread(
                    target=kickoff_task, args=(input_data,))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        return results
