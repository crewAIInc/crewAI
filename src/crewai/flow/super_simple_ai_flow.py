import asyncio

from crewai.flow.flow import Flow, listen, start
from openai import OpenAI


class ExampleFlow(Flow):

    @start()
    def start_method(self):
        print("Starting flow")
        client = OpenAI()

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Return the name of a random city in the world.",
                },
            ],
        )

        random_city = response.choices[0].message.content
        print("random_city", random_city)

        return random_city

    @listen(start_method)
    def second_method(self, result):
        print("Second method received:", result)
        # print("Second city", result)


async def main():
    flow = ExampleFlow()
    await flow.kickoff()


asyncio.run(main())
