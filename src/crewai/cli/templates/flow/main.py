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
        result = PoemCrew().crew().kickoff(inputs={"sentence_count": self.state.sentence_count})
        
        print("Poem generated", result.raw)
        self.state.poem = result.raw
        
    @listen(generate_poem)
    def save_poem(self):
        print("Saving poem")
        with open("poem.txt", "w") as f:
            f.write(self.state.poem)

async def kickoff():
    poem_flow = PoemFlow()
    await poem_flow.kickoff()


def plot():
    poem_flow = PoemFlow()
    poem_flow.plot()


def main():
    asyncio.run(kickoff())


if __name__ == "__main__":
    main()
