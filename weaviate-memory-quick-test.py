from crewai import Agent, Crew, Process, Task
from crewai.memory.external.external_memory import ExternalMemory

# Create external memory instance
external_memory = ExternalMemory(
    embedder_config={
        "provider": "weaviate",
        "config": {"user_id": "U-123"}
    }
)

BiomedicalMarketingAgent = Agent(
    role='Biomedical Marketing Agent',
    goal='Continuously track the latest biomedical advancements and identify how Weaviate’s features can support AI applications in biomedical research, diagnostics, and personalized medicine.',
    backstory='As a former biomedical product marketer turned AI strategist, you understand the complex language and regulatory landscape of biomedical innovation. With a keen eye on genomics, clinical research, and medical devices, it now leverages LLMs and vector search to explore how Weaviate’s capabilities can streamline scientific discovery and patient-centric campaigns.',
    llm="gpt-4.1-mini",
    verbose=True
)

biomed_agent_task = Task(
    description = """
        Conduct thorough research about Weaviate.
        Make sure you find any interesting and relevant information using the web and Weaviate blogs.
    """,
    expected_output = """
        Write an industry specific analysis of why this Weaviate feature would be useful for your industry of expertise.
    """,
    agent = BiomedicalMarketingAgent
)

crew = Crew(
    agents=[BiomedicalMarketingAgent],
    tasks=[biomed_agent_task],
    external_memory=external_memory,  # Separate from basic memory
    process=Process.sequential,
    verbose=True
)

crew.kickoff()