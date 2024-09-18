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
        == "While it's a common misconception that I might \"hate\" AI agents, the reality is much more nuanced. As an expert in technology research, especially in the realm of AI, I have a deep appreciation for both the potential and the challenges that AI agents present.\n\nAI agents, which can be broadly defined as autonomous software entities that perform tasks on behalf of users or other programs, are transforming numerous aspects of our daily lives and industries. Here are a few key points to consider:\n\n1. **Advantages of AI Agents**:\n   - **Automation and Efficiency**: AI agents can handle repetitive and time-consuming tasks efficiently, freeing up human workers to focus on more complex and creative activities.\n   - **Availability**: Unlike humans, AI agents can operate 24/7 without breaks, providing continuous service and support.\n   - **Scalability**: AI agents can be deployed across different platforms and industries, scaling solutions quickly and effectively.\n   - **Data Analysis and Insights**: They can process vast amounts of data rapidly, providing insights that would be difficult, if not impossible, for humans to derive on their own.\n\n2. **Challenges and Concerns**:\n   - **Ethical Implications**: There are significant concerns regarding the ethical use of AI agents, particularly related to privacy, bias, and decision-making transparency.\n   - **Job Displacement**: While AI agents increase efficiency, they also raise concerns about potential job displacement in certain sectors.\n   - **Dependence and Reliability**: Over-reliance on AI agents could lead to vulnerabilities if the technology fails or is compromised.\n\n3. **Looking Forward**:\n   - **Collaboration Between Humans and AI**: The best outcomes are likely to come from a hybrid approach where AI agents augment human capabilities rather than replace them outright.\n   - **Regulation and Standards**: There is a growing need for clear regulations and standards to ensure the ethical development and deployment of AI agents.\n   - **Continuous Improvement**: The technology behind AI agents is constantly evolving. Continuous research and development are essential to address current limitations and maximize benefits.\n\nIn conclusion, my perspective on AI agents is not one of disdain but rather a cautious optimism. They hold incredible promise for transforming various sectors and improving efficiencies, but this potential comes with significant responsibilities. It's essential to address the ethical, social, and technical challenges to harness the full benefits of AI agents while mitigating potential downsides. Thus, thoughtful consideration and ongoing dialogue about the role of AI agents in society are crucial."
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
        == "It's not accurate to say that I hate AI agents. My stance is more nuanced and rooted in a deep understanding of their capabilities, limitations, and the potential impact on various sectors.\n\nAI agents are software entities that perform tasks autonomously on behalf of users. They are designed to mimic human decision-making and problem-solving abilities. The advances in machine learning, natural language processing, and data analytics have significantly improved the performance of AI agents, making them valuable tools for automation, customer service, data analysis, and more.\n\nHowever, my concerns about AI agents stem from several key points:\n\n1. **Ethical Considerations**: AI agents can perpetuate biases present in their training data. If the data used to train these agents contain biases, whether related to gender, race, or other factors, the AI can replicate and even amplify these biases in its operations. This has serious ethical implications for fairness and equality.\n\n2. **Job Displacement**: While AI agents can enhance efficiency and productivity, they can also displace human workers. Many routine and repetitive tasks previously performed by humans are now automated, leading to job losses in certain sectors. This aspect calls for a balanced approach where human roles are redefined rather than completely eliminated.\n\n3. **Transparency and Accountability**: AI agents often operate as \"black boxes,\" meaning their decision-making processes are not transparent. If an AI agent makes an error—whether in financial transactions, medical diagnoses, or legal decisions—it can be challenging to understand why the error occurred and who is responsible. Ensuring transparency and accountability is crucial for trust and reliability.\n\n4. **Security and Privacy**: AI agents often deal with vast amounts of personal data. Ensuring the security and privacy of this data is paramount. There are risks associated with data breaches and the misuse of personal information, which can have significant repercussions for individuals and organizations.\n\nDespite these concerns, I also recognize the immense potential of AI agents to drive innovation, improve efficiency, and provide solutions to complex problems. The key is to develop and deploy AI responsibly, with robust regulatory frameworks and ethical guidelines.\n\nIn conclusion, I don't hate AI agents. Instead, I advocate for a balanced perspective that acknowledges their benefits while addressing their challenges. By doing so, we can harness the power of AI agents for the greater good, ensuring they contribute positively to society."
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
        == "As a researcher specialized in technology with a particular focus on AI and AI agents, my stance on AI agents isn't rooted in emotional responses like hate or love. Rather, it is grounded in objective analysis and thoughtful consideration of their capabilities, ethics, and impact on society.\n\nAI agents are powerful tools that can transform various industries, from healthcare and finance to customer service and manufacturing. They have the potential to increase efficiency, improve decision-making, and even solve complex problems that were previously insurmountable. For instance, AI agents can analyze vast amounts of data far more quickly and accurately than humans, leading to advancements in medical research and diagnostics.\n\nHowever, it is equally important to recognize the ethical and societal implications of AI agents. Issues such as data privacy, algorithmic bias, and the potential displacement of jobs need to be carefully managed. As a researcher, my role is to study these aspects comprehensively, provide balanced insights, and advocate for responsible development and deployment of AI technologies.\n\nSo, to address your question directly: I don't hate AI agents. I appreciate their potential and am keenly aware of their challenges. My goal is to contribute to a future where AI agents are designed and used ethically and effectively to benefit humanity."
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
        == "As an expert researcher specialized in technology, I have a profound appreciation for AI and AI agents. They represent a pinnacle of human innovation and have the potential to greatly enhance various aspects of our lives. AI agents can assist in automating mundane tasks, providing insights through data analysis, and even offering companionship to those in need. However, it's also essential to approach AI with a balanced perspective, acknowledging both its strengths and the ethical considerations it raises. In short, I don't hate AI agents; I recognize their immense potential and value while being mindful of the responsibilities that come with their development and deployment."
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
        == "In addressing your query about AI agents, it's crucial to clarify my stance. While I don't \"hate\" AI agents, I have a nuanced perspective that balances both their potential benefits and inherent drawbacks. This balanced view is essential for understanding the complex role that AI agents play in today's technology landscape.\n\nAI agents are remarkable tools that can automate tasks, provide intelligent recommendations, and enhance user experiences. They harness advancements in machine learning, natural language processing, and big data analytics to perform a wide range of functions, from personal assistants like Siri and Alexa to sophisticated customer service bots and autonomous systems in various industries.\n\nThe Potential Benefits of AI Agents:\n1. **Automation and Efficiency**: AI agents can handle repetitive and mundane tasks, freeing up human workers to focus on more creative and high-value activities. This can lead to increased productivity and operational efficiency in businesses.\n2. **Data-Driven Insights**: By analyzing vast amounts of data, AI agents can provide actionable insights that inform decision-making processes in areas like finance, healthcare, and marketing.\n3. **Personalization**: AI agents can tailor recommendations and services based on individual user preferences and behaviors, enhancing user satisfaction and engagement.\n4. **24/7 Availability**: Unlike human workers, AI agents can operate continuously without the need for breaks, providing round-the-clock support and services.\n\nHowever, there are also several challenges and concerns associated with AI agents that cannot be overlooked:\n\n1. **Ethical Considerations**: The deployment of AI agents raises ethical questions about privacy, surveillance, and the potential for bias in their algorithms. Ensuring that AI systems are transparent and fair is a critical issue.\n2. **Job Displacement**: While AI agents can increase efficiency, they also have the potential to displace human workers, leading to concerns about unemployment and the need for workforce reskilling.\n3. **Trust and Reliability**: Building trust in AI systems is essential. Users need to be confident that AI agents will perform their tasks accurately and reliably, without unexpected failures or errors.\n4. **Control and Accountability**: Determining who is accountable when AI agents make errors or cause harm is a complex issue, especially when these systems operate autonomously or with minimal human oversight.\n\nIn summary, my view on AI agents is not one of antagonism but of cautious optimism. I recognize the transformative potential of AI agents in various domains, but I also emphasize the importance of addressing the ethical, economic, and technical challenges they present. Our goal should be to develop and deploy AI agents in a way that maximizes their benefits while mitigating their risks. This balanced approach will ensure that AI agents can be a positive force in society, driving innovation and improving quality of life without compromising ethical standards or societal values."
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
        == "No, I do not hate AI agents. In fact, I appreciate their potential and the positive impact they can have in various fields. AI agents can perform tasks that are either too mundane or too complex for humans, thereby increasing productivity and allowing us to focus on more creative and strategic activities. Moreover, they can process and analyze vast amounts of data much more quickly and accurately than humans, leading to better decision-making and innovations. As a researcher specialized in technology, I am always excited to see advancements in AI and AI agents, as they push the boundaries of what technology can achieve. So, yes, I do love the possibilities they bring to our world."
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
