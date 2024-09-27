#!/usr/bin/env python
import asyncio
from random import randint

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start
from .crews.poem_crew.poem_crew import PoemCrew

class PoemState(BaseModel):
    sentence_count: int = 1
    poem: str = ""

class PoemFlow(Flow[PoemState]):

    @start()
    def generate_sentence_count(self):
        print("Generating sentence count")
        # Generate a number between 1 and 5
        self.state.sentence_count = randint(1, 5)  

    @listen(generate_sentence_count)
    def generate_poem(self):
        print("Generating poem")
        print(f"State before poem: {self.state}")
        poem_crew = PoemCrew().crew()
        result = poem_crew.kickoff(inputs={"sentence_count": self.state.sentence_count})
        
        print("Poem generated", result.raw)
        self.state.poem = result.raw
        
        print(f"State after generate_poem: {self.state}")

    @listen(generate_poem)
    def save_poem(self):
        print("Saving poem")
        print(f"State before save_poem: {self.state}")
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)
        print(f"State after save_poem: {self.state}")

async def run():
    """
    Run the flow.
    """
    poem_flow = PoemFlow()
    await poem_flow.kickoff()


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
