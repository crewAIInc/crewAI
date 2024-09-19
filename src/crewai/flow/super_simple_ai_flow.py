import asyncio

from crewai.flow.flow import Flow, listen, start
from openai import OpenAI


class ExampleFlow(Flow):
    client = OpenAI()
    model = "gpt-4o-mini"

    @start()
    def generate_city(self):
        print("Starting flow")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": "Return the name of a random city in the world.",
                },
            ],
        )

        random_city = response.choices[0].message.content
        print("---- Random City ----")
        print(random_city)
        return random_city

    @listen(generate_city)
    def generate_fun_fact(self, random_city):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": f"Tell me a fun fact about {random_city}",
                },
            ],
        )

        fun_fact = response.choices[0].message.content
        print("---- Fun Fact ----")
        print(fun_fact)


async def main():
    flow = ExampleFlow()
    await flow.kickoff()


asyncio.run(main())
