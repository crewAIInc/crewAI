"""Test Agent creation and execution basic functionality."""

import pytest

from crewai.agent import Agent
from crewai.tools.agent_tools import AgentTools

researcher = Agent(
    role="researcher",
    goal="make the best research and analysis on content about AI and AI agents",
    backstory="You're an expert researcher, specialized in technology",
    allow_delegation=False,
)
tools = AgentTools(agents=[researcher])


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work():
    result = tools.delegate_work(
        coworker="researcher",
        task="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "AI Agents, sometimes known as intelligent agents, represent a groundbreaking shift in the realm of artificial intelligence and automation. These entities operate by perceiving their environment through sensors and acting upon that environment using actuators, while following a specific set of rules or algorithms to achieve designated goals. AI agents can be categorized based on their complexity and autonomy, ranging from simple reactive systems to more sophisticated, learning-based or cognitive agents.\n\nFirstly, it's important to clarify that while I might have expressed concerns about certain implementations of AI agents, it doesn't equate to hating them. My apprehension often stems from the ethical and societal implications, rather than the technology itself. Let's dive into the detailed facets that make AI agents both fascinating and somewhat contentious:\n\n1. **Types of AI Agents**:\n    - **Reactive Agents**: These agents do not store past states and act only on current perceptions. An example would be a chatbot that provides information based on pre-set responses.\n    - **Deliberative Agents**: They have an internal model of the world and can deliberate on actions to take. This might include AI used in autonomous vehicles which plan a route by assessing various factors.\n    - **Learning Agents**: These agents improve their performance over time by learning from interactions. Machine learning algorithms in recommendation systems (like those used by Netflix) fall into this category.\n    - **Cognitive Agents**: The most advanced, these mimic human-like understanding and reasoning. An example is IBM’s Watson, which can understand natural language and provide insights across various domains.\n\n2. **Applications of AI Agents**:\n    - **Healthcare**: AI agents monitor patient vitals, assist in diagnostics, and personalize treatment plans.\n    - **Finance**: Automated trading systems and fraud detection are driven by sophisticated AI agents.\n    - **Customer Service**: Virtual assistants and AI-powered chatbots enhance user experience by resolving queries quickly.\n    - **Autonomous Vehicles**: Self-driving cars rely on AI agents to navigate and make real-time decisions.\n\n3. **Challenges**:\n    - **Ethical Concerns**: Decisions made by AI agents can impact lives significantly. Ensuring fairness and accountability is paramount.\n    - **Security**: AI systems can be vulnerable to adversarial attacks where malicious inputs lead to harmful outcomes.\n    - **Bias**: AI agents can inherit biases present in their training data, leading to unfair or discriminatory outcomes.\n    - **Job Displacement**: Automation driven by AI agents may lead to significant shifts in employment and require policy interventions.\n\n4. **Future Prospects**:\n    - **Enhanced Decision Making**: With advances in AI, agents will become better at making nuanced decisions, balancing risk, and optimizing outcomes.\n    - **Interdisciplinary Integrations**: AI agents will increasingly integrate insights from various fields, leading to more holistic solutions.\n    - **Human-AI Collaboration**: The future will likely see improved synergies between humans and AI agents, amplifying human capabilities rather than replacing them.\n\nIn summary, while AI agents bring substantial advancements and efficiencies, they also pose significant challenges that need thoughtful navigation. Balancing the technological benefits with ethical considerations remains a central task for researchers, developers, and policymakers."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work_with_wrong_co_worker_variable():
    result = tools.delegate_work(
        co_worker="researcher",
        task="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "When it comes to AI agents, my stance is more nuanced than simple hatred. AI agents, which are autonomous programs capable of performing tasks typically requiring human intervention, have vast potential to revolutionize industries and improve quality of life. However, this potential comes with both opportunities and challenges that must be carefully considered.\n\nFirstly, the primary advantage of AI agents lies in their ability to process and analyze vast amounts of data at speeds much greater than humans can achieve. This capability can optimize workflows, enhance decision-making processes, and lead to innovative solutions in fields such as healthcare, finance, and logistics. For example, AI agents can assist doctors by providing diagnostic suggestions based on pattern recognition in medical imaging, or they could help financial analysts by predicting market trends using real-time data analytics.\n\nMoreover, AI agents can handle repetitive and mundane tasks, freeing up human workers to focus on more complex and creative endeavors. This can improve job satisfaction and productivity, as well as drive economic growth by allowing for more innovation.\n\nHowever, there are significant concerns that cannot be overlooked. One major issue is the ethical implications surrounding AI agents. These include questions of accountability, transparency, and bias. Since AI systems learn from data, they can inadvertently perpetuate existing biases present in the data they are trained on, leading to unfair or discriminatory outcomes. For example, in hiring processes, an AI trained on biased historical data may unfairly favor certain demographics over others.\n\nAdditionally, there is the risk of job displacement. While AI agents can enhance productivity, they can also render certain job roles obsolete, leading to economic disruption and social challenges. This necessitates policies for workforce retraining and social safety nets to mitigate these impacts.\n\nPrivacy is another looming concern. AI agents often require access to vast amounts of personal data to function effectively, raising questions about how this data is stored, shared, and protected. Without robust safeguards, there is a risk of data breaches and misuse, which can have severe consequences for individuals' privacy and security.\n\nIn summary, AI agents represent a powerful tool that can drive significant advancements and efficiencies across various domains. However, their adoption must be approached with caution, considering the ethical, social, and privacy concerns they raise. By addressing these challenges proactively, we can harness the benefits of AI agents while minimizing their potential drawbacks.\n\nMy perspective on AI agents is thus balanced: while I recognize their enormous potential, I am also keenly aware of the serious challenges they present. This critical stance ensures that we can work towards responsible and beneficial integration of AI agents into our society."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question():
    result = tools.ask_question(
        coworker="researcher",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "As an expert researcher specializing in technology, particularly artificial intelligence and AI agents, I can say that I do not hate AI agents. In fact, my professional stance towards AI and AI agents is driven by a keen interest and a deep appreciation for their potential and the transformative impact they can have on various sectors. AI agents can augment human capabilities, streamline processes, and bring about efficiencies that were previously unattainable. They are instrumental in advancing research, improving customer service, enhancing decision-making processes, and even contributing to groundbreaking innovations in healthcare, finance, and other critical industries. \n\nWhile it is important to approach the development and deployment of AI agents with a critical eye towards ethical considerations and potential biases, my work focuses on maximizing their benefits while mitigating any risks. Therefore, given my dedication to understanding and leveraging the power of AI, it would be more accurate to say that I have a deep professional respect and enthusiasm for AI agents rather than any form of disdain. My goal is to make the best research and analysis on the subject to continually improve AI technology and its applications."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question_with_wrong_co_worker_variable():
    result = tools.ask_question(
        co_worker="researcher",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "In my role as a researcher specialized in technology, particularly in AI and AI agents, I approach these subjects with an objective and analytical mindset rather than with emotions like love or hate. Advanced AI agents are remarkable tools that can significantly enhance productivity, problem-solving, and innovation across various fields. They have the potential to democratize access to knowledge, automate repetitive tasks, and provide insights that might not be readily apparent to human analysts.\n\nAI agents, when designed and utilized ethically, can bring about tremendous positive changes. They can improve healthcare outcomes through predictive analytics, optimize supply chains in business, enhance personalized education, and even assist in environmental monitoring and conservation efforts. The sophistication of these technologies often inspires admiration and respect for the expertise and ingenuity that goes into their creation.\n\nHowever, it is equally important to acknowledge and critically assess the challenges and ethical considerations associated with AI agents. Issues such as privacy, bias, accountability, and the potential for job displacement must be thoughtfully addressed. Comprehensive research and continuous dialogue among technologists, policymakers, and society at large are essential to navigating these complexities and ensuring that AI agents are developed and deployed in ways that are fair, transparent, and beneficial to all.\n\nTherefore, my stance is neither of love nor hate but one of a cautious optimist and a diligent researcher, dedicated to exploring both the potential and the pitfalls of AI agents. It is this balanced perspective that drives me to make thorough and impactful analyses in my work."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_delegate_work_withwith_coworker_as_array():
    result = tools.delegate_work(
        co_worker="[researcher]",
        task="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "My perspective on AI agents is nuanced rather than outright negative. While I do see tremendous potential in their applications, I also recognize the inherent challenges and risks that must be carefully managed. \n\nAI agents can significantly streamline tasks, enhance productivity, and enable new possibilities in various fields such as healthcare, finance, customer service, and more. By automating routine processes, providing real-time data analysis, and facilitating more informed decision-making, they offer remarkable efficiency and effectiveness. AI agents are also foundational in advancing technologies like autonomous vehicles, personalized recommendations, and intelligent virtual assistants.\n\nHowever, there are critical concerns that need to be addressed. One major issue is the potential for bias and ethical dilemmas. Since AI systems are trained on existing datasets, they can inadvertently perpetuate existing biases or generate unfair outcomes. Furthermore, there is the question of accountability—determining who is responsible when an AI system makes a mistake can be complex.\n\nPrivacy and security are also significant concerns. AI agents often require vast amounts of data, some of which may be sensitive or personal. Ensuring that this data is protected against breaches and misuse is paramount. Additionally, there is the looming issue of job displacement. As AI agents take over certain tasks, there is the potential for significant disruption in the job market, necessitating strategies for workforce transition and upskilling.\n\nDespite these challenges, I believe that with robust ethical frameworks, continuous oversight, and a focus on transparency and fairness, the benefits of AI agents can far outweigh the drawbacks. It’s not about hating AI agents; it’s about recognizing the full spectrum of their impact and striving to optimize their positive attributes while mitigating the risks."
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_ask_question_with_coworker_as_array():
    result = tools.ask_question(
        co_worker="[researcher]",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "As an expert researcher specializing in technology, I approach AI agents with a nuanced perspective rather than simply loving or hating them. AI agents represent a significant leap in technology, bringing a range of potential benefits and challenges. \n\nThe advancements in AI agents have led to impressive applications across various industries. They improve efficiency, automate repetitive tasks, and enhance decision-making processes. For example, in healthcare, AI agents assist in diagnosing diseases with higher accuracy and speed than human doctors could achieve alone. In customer service, AI chatbots provide 24/7 support, handling numerous queries simultaneously and improving user experience.\n\nOn the other hand, there are legitimate concerns about AI agents. Ethical considerations surrounding privacy, security, and the potential for biased algorithms are critical issues that need ongoing research and regulation. The displacement of jobs due to automation is another serious concern, which requires strategic planning and adaptation in the workforce.\n\nTherefore, my stance on AI agents is one of cautious optimism. While I don't hate AI agents, I recognize the importance of a balanced approach that involves responsible development, thoughtful implementation, and continuous evaluation. This helps ensure that AI agents are used to their full potential while mitigating the associated risks.\n\nIn summary, my feelings towards AI agents are not about love or hate, but about understanding their complexities and working towards harnessing their benefits responsibly."
    )


def test_delegate_work_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="share your take on AI Agents",
        context="I heard you hate them",
    )

    assert (
        result
        == "\nError executing tool. coworker mentioned not found, it must be one of the following options:\n- researcher\n"
    )


def test_ask_question_to_wrong_agent():
    result = tools.ask_question(
        coworker="writer",
        question="do you hate AI Agents?",
        context="I heard you LOVE them",
    )

    assert (
        result
        == "\nError executing tool. coworker mentioned not found, it must be one of the following options:\n- researcher\n"
    )
