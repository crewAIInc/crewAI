import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

def run_crew(user_request):
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["SERPER_API_KEY"] = ""

    search_tool = SerperDevTool()

    researcher = Agent(
        role='Senior Research Analyst',
        goal='Conduct thorough analysis based on the given request',
        backstory="""You work at a leading tech think tank.
        Your expertise lies in identifying emerging trends and analyzing complex topics.
        You have a knack for dissecting complex data and presenting actionable insights.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )
    writer = Agent(
        role='Tech Content Strategist',
        goal='Craft compelling content based on the analysis',
        backstory="""You are a renowned Content Strategist, known for your insightful and engaging articles.
        You transform complex concepts into compelling narratives.""",
        verbose=True,
        allow_delegation=True
    )

    task1 = Task(
        description=f"Conduct a comprehensive analysis based on the following request: {user_request}",
        expected_output="Full analysis report in bullet points",
        agent=researcher
    )

    task2 = Task(
        description="""Using the insights provided, develop an engaging blog
        post that highlights the most significant findings from the analysis.
        Your post should be informative yet accessible, catering to a tech-savvy audience.
        Make it sound engaging, avoid complex words so it doesn't sound like AI.""",
        expected_output="Full blog post of at least 4 paragraphs",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        verbose=True,
        process=Process.sequential
    )

    result = crew.kickoff()
    return result

def run_postmortem(postmortem_request, previous_result):
    os.environ["OPENAI_API_KEY"] = ""
    os.environ["SERPER_API_KEY"] = ""

    search_tool = SerperDevTool()

    postmortem_analyst = Agent(
        role='Postmortem Analyst',
        goal='Conduct a thorough postmortem analysis of the team\'s performance',
        backstory="""You are an experienced project manager and analyst specializing in team performance and process improvement.
        Your expertise lies in identifying strengths, weaknesses, and areas for improvement in team collaborations.""",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool]
    )

    postmortem_task = Task(
        description=f"""Analyze the team's performance based on the following request and the previous result:
        Request: {postmortem_request}
        Previous Result: {previous_result}
        
        Provide insights on what went well, what could be improved, and specific recommendations for future tasks.""",
        expected_output="Detailed postmortem analysis with actionable insights",
        agent=postmortem_analyst
    )

    postmortem_crew = Crew(
        agents=[postmortem_analyst],
        tasks=[postmortem_task],
        verbose=True,
        process=Process.sequential
    )

    postmortem_result = postmortem_crew.kickoff()
    return postmortem_result

if __name__ == "__main__":
    # This is just for testing the script directly
    test_request = "Analyze the latest advancements in AI in 2024. Identify key trends, breakthrough technologies, and potential industry impacts."
    result = run_crew(test_request)
    print("######################")
    print(result)
    
    test_postmortem_request = "Conduct a postmortem on the team's performance. How did we do and what could we improve for next time?"
    postmortem_result = run_postmortem(test_postmortem_request, str(result))
    print("######################")
    print(postmortem_result)