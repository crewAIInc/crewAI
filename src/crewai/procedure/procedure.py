import asyncio
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput


class Procedure(BaseModel):
    crews: List[Crew] = Field(
        ..., description="List of crews to be executed in sequence"
    )

    async def kickoff(self, inputs: List[Dict[str, Any]]) -> List[CrewOutput]:
        current_inputs = inputs

        for crew in self.crews:
            # Process all inputs for the current crew
            crew_outputs = await self._process_crew(crew, current_inputs)
            print("Crew Outputs", crew_outputs)

            # Prepare inputs for the next crew
            current_inputs = [output.to_dict() for output in crew_outputs]

        # Return the final outputs
        return crew_outputs

    async def _process_crew(
        self, crew: Crew, inputs: List[Dict[str, Any]]
    ) -> List[CrewOutput]:
        # Kickoff crew asynchronously for each input
        crew_kickoffs = [crew.kickoff_async(inputs=input_data) for input_data in inputs]

        # Wait for all kickoffs to complete
        outputs = await asyncio.gather(*crew_kickoffs)

        return outputs
