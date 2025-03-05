from crewai import Agent, Task, Crew
from crewai.memory import ChatMessageHistory, MessageRole

# Create a chat message history
chat_history = ChatMessageHistory()

# Add some messages
chat_history.add_human_message("Hello, I need help with a research task.")
chat_history.add_ai_message("I'd be happy to help! What topic are you interested in?")
chat_history.add_human_message("I'm interested in renewable energy technologies.")

# Create an agent with access to the chat history
researcher = Agent(
    role="Renewable Energy Researcher",
    goal="Provide accurate and up-to-date information on renewable energy technologies",
    backstory="You are an expert in renewable energy with years of research experience.",
    verbose=True,
)

# Create a task that uses the chat history
research_task = Task(
    description=(
        "Review the conversation history and provide a detailed response about "
        "renewable energy technologies, addressing any specific questions or interests."
    ),
    expected_output="A comprehensive response about renewable energy technologies.",
    agent=researcher,
)

# Create a crew with memory enabled
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True,
    memory=True,
)

# Pass the chat history to the crew
# In a real REST API scenario, you would store and retrieve this between requests
crew_result = crew.kickoff(inputs={"chat_history": chat_history.get_messages_as_dict()})

# Add the crew's response to the chat history
chat_history.add_ai_message(str(crew_result))

# Print the full conversation history
print("\nFull Conversation History:")
for message in chat_history.get_messages():
    print(f"{message.role.value.capitalize()}: {message.content}")
