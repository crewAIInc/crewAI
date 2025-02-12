# Initialize a ChatOpenAI model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Create the agent with LangGraph
memory = MemorySaver()
agent_executor = create_react_agent(
    llm,
    tools,
    checkpointer=memory
)

# Pass the LangGraph agent to the adapter
wrapped_agent = LangChainAgentAdapter(
    langchain_agent=agent_executor,
    tools=tools,
    role="San Francisco Travel Advisor",
    goal="Curate a detailed list of the best neighborhoods to live in, restaurants to dine at, and attractions to visit in San Francisco.",
    backstory="An expert travel advisor with insider knowledge of San Francisco's vibrant culture, culinary delights, and hidden gems.",
) 