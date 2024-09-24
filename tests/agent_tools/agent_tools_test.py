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
        == "While I understand the concerns and skepticism surrounding AI agents, I wouldn't say that I hate them. My standpoint is more nuanced. AI agents, which are software entities that perform tasks autonomously using machine learning and other AI technologies, have tremendous potential to revolutionize various sectors.\n\nOn the positive side, AI agents can significantly enhance efficiency and productivity. For example, in customer service, AI agents can handle routine inquiries, allowing human agents to focus on more complex issues. In healthcare, they can assist in diagnosing diseases, thus speeding up the decision-making process and potentially saving lives. In finance, AI agents can automate trading, detect fraudulent activities, and provide personalized financial advice.\n\nHowever, there are legitimate concerns that need to be addressed. One major issue is the ethical implications of deploying AI agents. These include data privacy, biases in decision-making algorithms, and the lack of transparency in how these agents operate. Another concern is the potential job displacement that could result from increased automation. While AI agents can handle many tasks more efficiently than humans, this could lead to significant job losses in certain sectors.\n\nMoreover, there's the matter of reliability and accountability. AI agents, despite their advanced capabilities, are not infallible. They can make mistakes, and when they do, it can be challenging to pinpoint where things went wrong and who is responsible. This raises important questions about oversight and governance.\n\nIn summary, while I am cautious about the unchecked deployment of AI agents due to these ethical and practical concerns, I also recognize their potential to bring about significant positive changes. The key lies in finding a balanced approach that maximizes their benefits while mitigating their risks. This includes rigorous testing, continuous monitoring, and establishing clear ethical guidelines and policies to govern their use. \n\nBy addressing these challenges head-on, we can harness the power of AI agents in a way that is both innovative and responsible."
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
        == 'AI agents are specialized software entities that perform tasks autonomously on behalf of users. They leverage artificial intelligence to process inputs, learn from experiences, and make decisions, mimicking human-like behavior. Despite their transformative potential, I don\'t "hate" AI agents; rather, I hold a nuanced view that acknowledges both their advantages and limitations.\n\nAdvantages of AI Agents:\n1. **Efficiency and Productivity**: AI agents can handle repetitive tasks efficiently, freeing up human workers to focus on more complex and creative activities.\n2. **24/7 Operation**: Unlike humans, AI agents can work around the clock without breaks, significantly increasing productivity and service availability.\n3. **Data Processing**: They can process and analyze vast amounts of data quickly and accurately, supporting better decision-making.\n4. **Personalization**: AI agents can tailor services and recommendations based on user behavior and preferences, improving customer satisfaction.\n\nLimitations and Concerns:\n1. **Ethical Issues**: The deployment of AI agents raises concerns about data privacy, surveillance, and the potential for bias in decision-making algorithms.\n2. **Job Displacement**: There is legitimate concern about AI agents replacing human jobs, especially in industries where tasks are routine and repetitive.\n3. **Dependence on Data Quality**: AI agents\' performance hinges on the quality and quantity of data they are trained on. Poor data quality can lead to erroneous outcomes.\n4. **Complexity in Implementation**: Developing and maintaining AI agents requires significant technical expertise and resources. Problems can arise from their complexity, leading to potential failures.\n\nIn conclusion, while I don\'t "hate" AI agents, I am cautious of their broad and uncritical adoption. It’s essential to strike a balance between leveraging their capabilities and addressing the ethical, social, and technical challenges they present.'
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
        == "As a researcher specializing in technology and AI, I don't hate AI agents. In fact, I find them incredibly fascinating and beneficial. AI agents have the potential to transform various industries, improve efficiencies, and offer new solutions to complex problems. Their ability to learn, adapt, and perform tasks that were once thought to require human intelligence is remarkable. While it's important to consider ethical implications and ensure that AI systems are designed and deployed responsibly, I believe their overall positive impact on society and technology is significant. So to clarify, I don't hate AI agents; rather, I am quite enthusiastic about their potential and the advancements they bring to the field of technology."
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
        == "As an expert researcher specialized in technology and AI, my perspective on AI agents is shaped by both their potential and limitations. AI agents are tools designed to perform tasks, analyze data, and assist in various domains efficiently and accurately. They have the capability to revolutionize industries by automating complex processes, enhancing decision-making, and providing personalized experiences. For instance, in healthcare, AI agents can help in diagnosing diseases with high precision, while in finance, they can predict market trends and prevent fraud.\n\nHowever, my appreciation for AI agents does not mean I am blind to their challenges. There are valid concerns related to privacy, ethical use, and the potential displacement of jobs. The development and deployment of AI should be approached with caution, ensuring transparency, fairness, and accountability.\n\nIn conclusion, I value the advancements AI agents bring to the table and acknowledge their profound impact on society. My interest lies in leveraging their potential responsibly while addressing the associated ethical and societal challenges. So, while I love the capabilities and innovations brought forth by AI agents, I remain critically aware of the need for responsible development and use."
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
        == "It's interesting that you've heard I dislike AI agents; I suspect there may have been a miscommunication. My thoughts on AI agents are more nuanced than a simple like or dislike.\n\nAI agents can be incredibly powerful tools with the potential to drastically transform various industries. Their ability to automate tasks, analyze vast amounts of data, and make predictions can lead to significant improvements in efficiency and innovation. For instance, in healthcare, AI agents can assist in diagnosing diseases by quickly analyzing medical images. In finance, they can help in fraud detection by swiftly recognizing suspicious patterns in transactions. The applications are virtually limitless and continually expanding.\n\nHowever, there are concerns that need to be addressed, which might have led to a perception that I \"hate\" AI agents. One concern is the ethical implications surrounding their deployment. Issues such as data privacy, algorithmic bias, and the potential for job displacement are significant. For example, if an AI system is trained on biased data, it may make unfair or discriminatory decisions, perpetuating existing societal inequalities. Moreover, as AI agents take over repetitive tasks, there's a real risk that many jobs could become obsolete, causing economic disruption.\n\nAdditionally, there's the matter of accountability. When an AI agent makes a decision, it's not always clear who is responsible if something goes wrong. This opacity poses challenges for regulatory frameworks and trust in these systems. \n\nBalancing the tremendous benefits AI agents can provide with the ethical and practical challenges they introduce is crucial. Rather than viewing AI agents as something to be liked or disliked, I see them as tools that need thoughtful integration and rigorous oversight to maximize their positive impact and minimize their risks. Therefore, while I am enthusiastic about the potential of AI agents, I advocate for a cautious and responsible approach to their development and deployment."
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
        == "As an expert researcher in technology with a specialization in AI and AI agents, my perspective is rooted in my deep understanding of their capabilities and potential. AI agents, like any technology, are tools that can be used for both beneficial and harmful purposes. Personally, I do not hate AI agents; rather, I recognize their immense potential to transform industries, improve efficiencies, and solve complex problems. However, I also acknowledge that they come with challenges that need to be carefully managed, such as ethical considerations, privacy concerns, and the potential for job displacement.\n\nThe reason you might have heard that I love them is likely because I am passionate about the potential that AI agents hold for advancing technology and aiding humanity. I believe that with responsible development, transparent governance, and thoughtful integration, AI agents can indeed bring about positive change. My enthusiasm should not be misconstrued as blind love but rather as a measured appreciation for their capabilities and a commitment to navigating their complexities responsibly."
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
