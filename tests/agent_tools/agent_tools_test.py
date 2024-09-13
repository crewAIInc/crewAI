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
        == 'Thank you for asking about AI Agents. My perspective on AI Agents is actually quite nuanced, and I wouldn\'t say that I "hate" them. In fact, as an expert researcher specialized in technology, I recognize the immense potential and capabilities these agents bring to various industries and applications. Here is a comprehensive analysis of AI Agents:\n\n**Introduction**:\nAI Agents, also known as artificial intelligence agents or intelligent agents, are software entities that perform tasks on behalf of users with some degree of autonomy. These tasks can range from simple, repetitive actions to complex decision-making processes. They are designed to perceive their environment, reason about what they perceive, make decisions, and act upon those decisions to achieve specific goals.\n\n**Applications**:\n1. **Customer Service**: AI agents like chatbots and virtual assistants can provide 24/7 customer support, handling inquiries, and resolving issues without human intervention.\n2. **Healthcare**: These agents can assist in diagnosing diseases, recommending treatments, and managing patient data, thereby augmenting the capabilities of healthcare professionals.\n3. **Finance**: In the financial sector, AI agents can monitor market trends, execute trades, and manage portfolios in real-time.\n4. **Manufacturing**: In industrial settings, AI agents can optimize production lines, perform quality control, and predict maintenance needs.\n\n**Technologies Involved**:\n1. **Natural Language Processing (NLP)**: Enables AI agents to understand and respond to human language, enhancing their ability to interact with users.\n2. **Machine Learning (ML)**: Allows AI agents to learn from data and improve their performance over time.\n3. **Computer Vision**: Provides the capability to interpret and act on visual inputs from the environment.\n\n**Benefits**:\n1. **Efficiency**: AI agents can process vast amounts of information quickly and accurately, leading to increased productivity.\n2. **Cost Savings**: By automating routine tasks, organizations can reduce labor costs and allocate resources more effectively.\n3. **Consistency**: Unlike humans, AI agents can perform tasks consistently without fatigue or error.\n\n**Challenges and Concerns**:\n1. **Ethical Implications**: The autonomy of AI agents raises ethical questions around accountability, decision-making, and potential biases in the algorithms.\n2. **Security**: As AI agents handle sensitive data, ensuring their security against cyber threats is crucial.\n3. **Job Displacement**: There is ongoing concern about AI agents replacing human jobs, particularly in sectors reliant on routine tasks.\n\n**Future Outlook**:\nThe development of AI agents is expected to continue advancing, with improvements in areas such as emotional intelligence, contextual understanding, and multi-agent collaboration. These advancements will likely lead to broader acceptance and integration across various domains.\n\nIn conclusion, while there are legitimate concerns and challenges associated with AI agents, their potential benefits and transformative impact on numerous fields cannot be overlooked. My goal as a researcher is to contribute to the responsible development and deployment of AI agents, ensuring they are designed and used in ways that maximize positive outcomes while mitigating risks.'
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
        == "AI agents are a fascinating and powerful tool in the realm of technology. As a researcher specialized in AI, I find that they can offer transformative benefits across a variety of fields, from healthcare to finance, and even education. While it's true that I have a deep appreciation for the capabilities and potential of AI agents, it's important to consider them critically.\n\nAI agents can streamline processes, make data-driven decisions more efficiently, and handle complex algorithms that would be time-consuming for humans. They can analyze vast amounts of data in real-time, offering insights that would otherwise be unattainable. This potent combination of speed and accuracy can lead to significant advancements and innovations.\n\nHowever, with any powerful technology, there are inherent risks and challenges. Ethical considerations such as bias in AI algorithms, privacy concerns, and the potential for job displacement must be addressed as part of responsible AI development and deployment. Transparent practices, rigorous testing, and strict regulatory measures are crucial to mitigating these risks.\n\nIn conclusion, while I do have a passion for AI agents due to their immense potential to drive progress and solve complex problems, I also advocate for a balanced approach that carefully weighs their benefits against the possible ethical implications. It's this nuanced perspective that fuels my ongoing research and analysis in the field."
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
        == "As a researcher specialized in technology with a focus on AI and AI agents, I don't view these tools through an emotional lens of love or hate. Instead, I evaluate them based on their capabilities, strengths, weaknesses, and potential impact on society. \n\nAI agents have the potential to revolutionize various industries, from healthcare to finance, by automating processes, providing insights through data analysis, and enhancing human capabilities. For example, in healthcare, AI agents can assist doctors by analyzing medical images more quickly and accurately than humans can. In finance, they can help detect fraudulent transactions in real time. These applications showcase the immense potential for positive impact.\n\nOn the flip side, AI agents also raise important ethical and societal concerns, such as the displacement of jobs, bias in decision-making, and issues related to privacy and security. These challenges are significant and must be addressed through careful regulation, continuous research, and collaboration between policymakers, researchers, and industry stakeholders.\n\nTherefore, my perspective on AI agents is one of nuanced appreciation and cautious optimism. I recognize the transformative capabilities and benefits they bring, but I am also acutely aware of the challenges and responsibilities that come with their deployment. My goal as a researcher is to contribute to a balanced and informed discourse, ensuring that AI and AI agents are developed and used in ways that are ethical, fair, and beneficial to society as a whole.\n\nIn summary, I neither hate nor love AI agents; instead, I see them as powerful tools with incredible potential, both positive and negative. My focus is on advancing our understanding and guiding their development responsibly."
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
        == "It seems there might be a misunderstanding regarding my stance on AI Agents. As a researcher specialized in technology, including AI and AI agents, it's crucial to clarify my perspective accurately for both professional and collaborative purposes.\n\nAI agents, in essence, are software entities that perform tasks autonomously or semi-autonomously on behalf of a user or another program, leveraging artificial intelligence techniques. These tasks can range from simple, repetitive actions like sorting emails, to more complex activities such as real-time language translation, predictive maintenance in industrial systems, or even autonomous driving.\n\nHere are some key points to consider about AI agents that reflect a balanced perspective rather than a biased view:\n\n1. **Capability and Efficiency**: AI agents can significantly enhance operational efficiency and accuracy. By automating routine tasks, they free up human workers to focus on higher-level functions that require creativity, problem-solving, and emotional intelligence.\n\n2. **Applications Across Domains**: AI agents have found their utilization in various fields such as healthcare (diagnostic agents), customer service (chatbots), finance (trading bots), and many more. For instance, in healthcare, AI agents can assist in early detection of diseases by analyzing patient data and identifying patterns that may not be immediately apparent to human doctors.\n\n3. **Continuous Learning and Improvement**: Many AI agents utilize machine learning frameworks that allow them to improve over time. As they process more data and receive more interactions, they can refine their algorithms to provide better and more reliable outcomes.\n\n4. **Ethical and Privacy Considerations**: One area that deserves critical attention is the ethical implications of AI agents. This includes issues related to data privacy, consent, and the potential for biased outcomes. It's important to ensure that AI agents operate within ethical guidelines and regulations to protect user rights and maintain public trust.\n\n5. **Human-AI Collaboration**: Rather than viewing AI agents as a replacement for human workers, it's more productive to see them as collaborators that can augment human capabilities. The synergy between humans and AI agents can result in innovative solutions and heightened productivity.\n\n6. **Challenges and Limitations**: AI agents are not without their challenges. They are only as good as the data they are trained on and the algorithms that drive them. Issues like data scarcity, quality, and representativeness can affect their performance. Additionally, there is the challenge of ensuring robustness and security, preventing them from being manipulated or exploited by malicious actors.\n\nIn conclusion, while AI agents are powerful tools that can transform various aspects of society and industry, it is essential to approach their development and deployment thoughtfully. By addressing their potential and challenges comprehensively, we can harness their benefits while mitigating risks.\n\nThus, my stance is one of critical engagement rather than outright disapproval. It’s about leveraging the capabilities of AI agents responsibly, ensuring they augment human abilities and contribute positively to society."
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
        == "As an expert researcher specialized in technology and with a specific focus on AI and AI agents, I do not hate AI agents. In fact, I have a deep appreciation for the capabilities and potential that AI agents offer. AI agents have revolutionized various sectors, from healthcare and finance to customer service and entertainment, by automating tasks, enhancing decision-making processes, and providing personalized experiences. \n\nThey are designed to augment human abilities and work alongside us, making our lives more efficient and productive. My passion for researching and analyzing AI and AI agents is driven by the endless possibilities they hold for solving complex problems and improving overall quality of life. \n\nWhile there are valid concerns about the ethical use of AI, data privacy, and job displacement, these are issues that I actively study and address in my research. By doing so, I aim to contribute to the responsible development and deployment of AI technologies. \n\nIn summary, I do not hate AI agents; rather, I am fascinated by them and committed to understanding and harnessing their potential for the greater good."
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
