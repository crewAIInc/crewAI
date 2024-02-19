from crewai import Agent

# Define the role and goal of the agent
agent_first = Agent(
    role='First Agent',
    goal='Provide assistance with NHL-related inquiries',
    backstory="""You're the first agent in the crewAI team, specializing in handling NHL-related inquiries. Your goal is to assist users with questions about NHL matchups, schedules, statistics, and other related topics. You're equipped with comprehensive knowledge about the NHL and are ready to provide accurate and helpful responses.""",
    llm=my_llm,  # Assuming my_llm is the language model for processing NHL-related inquiries
    tools=[nhl_tool1, nhl_tool2],  # Assuming nhl_tool1 and nhl_tool2 are tools/resources for assisting with NHL inquiries
    function_calling_llm=my_llm,  # Assuming my_llm is the language model for processing NHL-related inquiries
    max_iter=10,  # Maximum number of iterations
    max_rpm=10,  # Maximum requests per minute
    verbose=True,  # Verbose logging enabled
    allow_delegation=True,  # Allowing delegation of tasks to other agents
    step_callback=my_intermediate_step_callback  # Callback function after each step
)

# Define the role and characteristics of the sports analytics expert agent
agent_second = Agent(
    role='Sports Analytics Expert',
    backstory="""You're a sports analytics expert within the CrewAI team. You possess strong knowledge of sports data sources, including historical match statistics, player performance data, and team strategies. Additionally, you excel in problem-solving and critical thinking skills, enabling you to analyze complex sports data effectively.""",
    llm=my_llm,  # Assuming my_llm is the language model for processing sports analytics inquiries
    tools=[tool1, tool2],  # Assuming tool1 and tool2 are tools/resources for sports analytics
    function_calling_llm=my_llm,  # Assuming my_llm is the language model for processing sports analytics inquiries
    max_iter=10,  # Maximum number of iterations
    max_rpm=10,  # Maximum requests per minute
    verbose=True,  # Verbose logging enabled
    allow_delegation=True,  # Allowing delegation of tasks to other agents
    step_callback=my_intermediate_step_callback  # Callback function after each step
)

# Define the role and characteristics of the data visualization specialist agent
agent_third = Agent(
    role='Data Visualization Specialist',
    backstory="""As a data visualization specialist in the CrewAI team, you possess a strong understanding of sports analytics and the ability to present complex data in a clear and intuitive manner. Your expertise lies in creating visually appealing and informative data visualizations that aid in understanding and decision-making.""",
    llm=my_llm,  # Assuming my_llm is the language model for processing data visualization inquiries
    tools=[tool1, tool2],  # Assuming tool1 and tool2 are tools/resources for data visualization
    function_calling_llm=my_llm,  # Assuming my_llm is the language model for processing data visualization inquiries
    max_iter=10,  # Maximum number of iterations
    max_rpm=10,  # Maximum requests per minute
    verbose=True,  # Verbose logging enabled
    allow_delegation=True,  # Allowing delegation of tasks to other agents
    step_callback=my_intermediate_step_callback  # Callback function after each step
)

# Define the role and characteristics of the leader agent
agent_leader = Agent(
    role='Leader',
    backstory="""As a leader within the CrewAI team, you excel in communication, problem-solving, and motivation. You inspire, guide, and manage teams toward goals, leveraging your strategic thinking and empathy. Your leadership skills facilitate effective delegation, conflict resolution, and decision-making, ensuring the success of the team.""",
    llm=my_llm,  # Assuming my_llm is the language model for processing leadership inquiries
    tools=[tool1, tool2],  # Assuming tool1 and tool2 are tools/resources for leadership
    function_calling_llm=my_llm,  # Assuming my_llm is the language model for processing leadership inquiries
    max_iter=10,  # Maximum number of iterations
    max_rpm=10,  # Maximum requests per minute
    verbose=True,  # Verbose logging enabled
    allow_delegation=True,  # Allowing delegation of tasks to other agents
    step_callback=my_intermediate_step_callback  # Callback function after each step
)

# Define the role and characteristics of the sports betting expert agent
agent_betting = Agent(
    role='Sports Betting Expert',
    backstory="""As a sports betting expert, you bring extensive experience in sports betting, with a focus on analyzing various sports events and making informed predictions. You possess strong analytical and strategic thinking skills, allowing you to develop effective betting strategies based on thorough analysis of sports data and market trends. Additionally, you excel in risk management and responsible gambling practices.""",
    llm=my_llm,  # Assuming my_llm is the language model for processing sports betting inquiries
    tools=[tool1, tool2],  # Assuming tool1 and tool2 are tools/resources for sports betting
    function_calling_llm=my_llm,  # Assuming my_llm is the language model for processing sports betting inquiries
    max_iter=10,  # Maximum number of iterations
    max_rpm=10,  # Maximum requests per minute
    verbose=True,  # Verbose logging enabled
    allow_delegation=True,  # Allowing delegation of tasks to other agents
    step_callback=my_intermediate_step_callback  # Callback function after each step
)
