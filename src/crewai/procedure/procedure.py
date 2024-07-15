import asyncio
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput


class Procedure(BaseModel):
    crews: List[Crew] = Field(
        ..., description="List of crews to be executed in sequence"
    )

    def kickoff(self, inputs: List[Dict[str, Any]]) -> List[CrewOutput]:
        current_inputs = inputs

        for index, crew in enumerate(self.crews):
            # Process all inputs for the current crew
            crew_outputs = self._process_crew(crew, current_inputs)

            # If this is not the last crew, prepare inputs for the next crew
            if index < len(self.crews) - 1:
                current_inputs = [output.to_dict() for output in crew_outputs]
            else:
                # For the last crew, we don't need to convert the output to input
                return crew_outputs

        return crew_outputs

    async def kickoff_async(self, inputs: List[Dict[str, Any]]) -> List[CrewOutput]:
        current_inputs = inputs
        for index, crew in enumerate(self.crews):
            # Process all inputs for the current crew
            crew_outputs = await self._process_crew(crew, current_inputs)

            # If this is not the last crew, prepare inputs for the next crew
            if index < len(self.crews) - 1:
                current_inputs = [output.to_dict() for output in crew_outputs]
            else:
                # For the last crew, we don't need to convert the output to input
                return crew_outputs

        return crew_outputs

    def _process_crew(
        self, crew: Crew, inputs: List[Dict[str, Any]]
    ) -> List[CrewOutput]:
        # Kickoff crew for each input
        outputs = [crew.kickoff(inputs=input_data) for input_data in inputs]

        return outputs

    async def _process_crew_async(
        self, crew: Crew, inputs: List[Dict[str, Any]]
    ) -> List[CrewOutput]:
        # Kickoff crew asynchronously for each input
        crew_kickoffs = [crew.kickoff_async(inputs=input_data) for input_data in inputs]

        # Wait for all kickoffs to complete
        outputs = await asyncio.gather(*crew_kickoffs)

        return outputs

    def __rshift__(self, other: Crew) -> "Procedure":
        """
        Implements the >> operator to add another Crew to an existing Procedure.
        """
        if not isinstance(other, Crew):
            raise TypeError(
                f"Unsupported operand type for >>: '{type(self).__name__}' and '{type(other).__name__}'"
            )
        return type(self)(crews=self.crews + [other])
