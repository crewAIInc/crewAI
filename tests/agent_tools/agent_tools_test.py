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
        == 'AI agents have been a transformative force in various industries, but it\'s important to note that personal opinions about them can be nuanced. While I wouldn’t use the term "hate," there are both positive aspects and significant concerns that need to be considered. \n\nOn the positive side, AI agents excel in performing repetitive tasks with high efficiency and accuracy, enabling businesses to optimize their operations and reduce costs. They play a vital role in areas such as customer service, where they can handle inquiries around the clock without fatigue, and in data analysis, where they can process vast amounts of information far quicker than a human could. Additionally, AI agents can be trained to adapt and learn from new data, making them increasingly valuable over time.\n\nHowever, there are also valid concerns. Privacy and security are major issues; AI agents often require access to large datasets that can include sensitive personal information, and breaches can have serious repercussions. There\'s also the ethical dimension, as the deployment of AI agents can lead to job displacement for roles that were traditionally performed by humans. This raises important questions about the future of work and the socio-economic impact of automation. Furthermore, the decision-making of AI agents can sometimes be opaque, making it challenging to understand how they arrive at specific conclusions or actions, which is particularly concerning in high-stakes areas like medical diagnoses or judicial decisions.\n\nSo, it\'s not a matter of "hating" AI agents, but rather a case of needing to balance their incredible potential with careful consideration of the associated risks and ethical implications. Thoughtful regulation and ongoing public discourse will be crucial in ensuring that AI agents are developed and deployed in ways that are beneficial to society as a whole.'
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
        == "My take on AI agents is multifaceted and nuanced. Contrary to what you may have heard, I do not hate AI agents. I have spent significant time and effort studying them, and I view them as transformative tools with immense potential. However, I also recognize the challenges and ethical considerations they bring to the table.\n\nAI agents are software programs that perform tasks autonomously using artificial intelligence techniques. They can range from simple chatbots to complex systems that navigate and make decisions in dynamic environments. Here’s my detailed perspective on AI agents:\n\n### The Advantages:\n\n1. **Efficiency and Productivity:**\n    - **Automation of Repetitive Tasks:** AI agents can handle mundane and repetitive tasks such as data entry, appointment scheduling, and customer support, which frees up human workers to focus on more strategic activities.\n    - **24/7 Availability:** Unlike humans, AI agents can operate round-the-clock, providing constant support and operations without the need for breaks.\n\n2. **Enhanced Decision Making:**\n    - **Data Processing Speed:** AI agents can process vast amounts of data at lightning speeds, enabling faster decision-making processes.\n    - **Predictive Analytics:** They can analyze historical data to predict future trends, helping businesses plan more effectively.\n\n3. **Personalization:**\n    - **User Experience:** AI agents can tailor interactions based on user data, providing a more personalized experience in applications like e-commerce and content recommendations.\n    - **Customer Insight:** They collect and analyze user preferences, allowing businesses to offer customized solutions and enhance customer satisfaction.\n\n### The Challenges:\n\n1. **Ethical Concerns:**\n    - **Bias and Fairness:** AI systems can inadvertently perpetuate biases if trained on non-representative or biased datasets.\n    - **Privacy Issues:** The collection and use of personal data by AI agents raise significant privacy concerns. Protecting user data from misuse is crucial.\n\n2. **Job Displacement:**\n    - **Automation Impact:** As AI agents automate more tasks, there is a potential for job displacement in certain sectors. Workforce reskilling and upskilling become essential.\n\n3. **Dependability and Trust:**\n    - **Reliability:** Ensuring that AI agents are reliable and do not malfunction in critical situations is a significant concern.\n    - **Transparency:** Users and stakeholders need to understand how AI agents make decisions, which can be challenging given the complexity of some AI models.\n\n### My Perspective:\n\nWhile I acknowledge the immense potential of AI agents to revolutionize industries and improve our daily lives, I believe it is crucial to approach their development and deployment responsibly. Ethical considerations, such as mitigating biases and ensuring transparency, are paramount. Additionally, I advocate for a balanced view—embracing the benefits while proactively addressing the challenges. \n\nIn conclusion, my view on AI agents is one of cautious optimism. I believe they can be incredibly beneficial if developed and used wisely, with proper oversight and ethical guidelines in place. Contrary to any misconceptions, my stance is not rooted in disdain but in a deep understanding of both their capabilities and the associated responsibilities."
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
        == "As a researcher specialized in technology, I neither love nor hate AI agents. My role is to objectively analyze their capabilities, impacts, and potential. AI agents have many beneficial applications, such as improving healthcare, optimizing logistics, and enhancing user experiences. However, there are also concerns regarding privacy, ethical implications, and job displacement. My professional stance is to critically evaluate both the advantages and the challenges, ensuring that the development and deployment of AI agents are conducted responsibly and ethically."
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
        == "I don't hate AI agents; in fact, I find them fascinating and very useful. My work as an expert researcher in technology, especially in AI and AI agents, has given me a deep appreciation of their capabilities and potential. AI agents have the ability to process vast amounts of data more quickly and accurately than any human could, which can lead to groundbreaking discoveries and efficiencies across numerous fields. They can automate tedious tasks, provide personalized recommendations, and even aid in complex decision-making processes.\n\nWhile it's important to recognize and address ethical and practical concerns with AI agents – such as biases in algorithms, potential job displacement, and ensuring data privacy – the benefits they offer cannot be overlooked. Properly developed and regulated, AI agents have the potential to significantly improve our lives and solve problems that were previously insurmountable.\n\nSo, to clarify, I love AI agents for their potential to drive innovation and make our lives better."
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
        == "AI agents are software entities that perform tasks on behalf of users autonomously. Their capabilities have significantly expanded with advancements in machine learning, natural language processing, and other AI technologies. My stance on AI agents isn't rooted in hatred or disdain, but rather in a critical perspective regarding their deployment and ethical implications.\n\nOne of the key advantages of AI agents is their ability to handle repetitive tasks efficiently, allowing humans to focus on more complex and creative activities. For instance, AI agents can automate customer service through chatbots, manage schedules, and even assist in data analysis.\n\nHowever, there are concerns that need addressing. The proliferation of AI agents raises critical issues surrounding privacy, security, and accountability. For example, AI agents often require access to vast amounts of personal data, which, if not handled properly, can lead to breaches of privacy. Moreover, the decision-making processes of AI agents can sometimes lack transparency, making it difficult to hold them accountable for errors or biased decisions.\n\nAnother concern is the potential impact on employment. As AI agents become more capable, there's a legitimate fear of job displacement in certain sectors. This necessitates a strategic approach to reskill the workforce and create new opportunities centered around AI.\n\nFurthermore, the design and deployment of AI agents must be inclusive and equitable. There have been instances where AI systems have demonstrated bias, reflecting the data they were trained on. This highlights the importance of diverse and representative data sets, as well as continuous monitoring and adjustment to ensure fair outcomes.\n\nIn conclusion, while AI agents offer considerable benefits, their development and integration into society must be approached with caution. By addressing ethical, privacy, and fairness concerns, we can harness the power of AI agents for positive and equitable advancements."
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
        == "As an expert researcher specialized in technology, especially in the field of AI and AI agents, I can confidently say that I do not hate AI agents. On the contrary, I find them fascinating and full of potential. AI agents can revolutionize various industries by automating tasks, enhancing decision-making processes, and providing insights that were previously unattainable. They can help in areas as diverse as healthcare, finance, security, and even creative arts.\n\nIt's important to recognize that AI agents, like any technology, come with their challenges and ethical considerations. Issues such as bias, transparency, and the impact on employment need to be thoughtfully addressed. However, these challenges can be mitigated through responsible development, clear regulations, and ongoing research.\n\nIn summary, I don't hate AI agents; I see them as powerful tools that, when used responsibly, can significantly improve our lives and society."
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
