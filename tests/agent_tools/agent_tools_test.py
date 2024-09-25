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
        == "I understand why you might think I dislike AI agents, but my perspective is more nuanced. AI agents, in essence, are incredibly versatile tools designed to perform specific tasks autonomously or semi-autonomously. They harness various artificial intelligence techniques, such as machine learning, natural language processing, and computer vision, to interpret data, understand tasks, and execute them efficiently. \n\nFrom a technological standpoint, AI agents have revolutionized numerous industries. In customer service, for instance, AI agents like chatbots and virtual assistants handle customer inquiries 24/7, providing quick and efficient solutions. In healthcare, AI agents can assist in diagnosing diseases, managing patient data, and even predicting outbreaks. The automation capabilities of AI agents also enhance productivity in areas such as logistics, finance, and cybersecurity by identifying patterns and anomalies at speeds far beyond human capabilities.\n\nHowever, it's important to acknowledge the potential downsides and challenges associated with AI agents. Ethical considerations are paramount. Issues such as data privacy, security, and biases in AI algorithms need to be carefully managed. There is also the human aspect to consider—over-reliance on AI agents might lead to job displacement in certain sectors, and ensuring a fair transition for affected workers is crucial.\n\nMy concerns generally stem from these ethical and societal implications rather than from the technology itself. I advocate for responsible AI development, which includes transparency, fairness, and accountability. By addressing these concerns, we can harness the full potential of AI agents while mitigating the associated risks.\n\nSo, to clarify, I don't hate AI agents; I recognize their immense potential and the significant benefits they bring to various fields. However, I am equally aware of the challenges they present and advocate for a balanced approach to their development and deployment."
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
        == "AI agents are essentially autonomous software programs that perform tasks or provide services on behalf of humans. They're built on complex algorithms and often leverage machine learning and neural networks to adapt and improve over time. \n\nIt's important to clarify that I don't \"hate\" AI agents, but I do approach them with a critical eye for a couple of reasons. AI agents have enormous potential to transform industries, making processes more efficient, providing insightful data analytics, and even learning from user behavior to offer personalized experiences. However, this potential comes with significant challenges and risks:\n\n1. **Ethical Concerns**: AI agents operate on data, and the biases present in data can lead to unfair or unethical outcomes. Ensuring that AI operates within ethical boundaries requires rigorous oversight, which is not always in place.\n\n2. **Privacy Issues**: AI agents often need access to large amounts of data, raising questions about privacy and data security. If not managed correctly, this can lead to unauthorized data access and potential misuse of sensitive information.\n\n3. **Transparency and Accountability**: The decision-making process of AI agents can be opaque, making it difficult to understand how they arrive at specific conclusions or actions. This lack of transparency poses challenges for accountability, especially if something goes wrong.\n\n4. **Job Displacement**: As AI agents become more capable, there are valid concerns about their impact on employment. Tasks that were traditionally performed by humans are increasingly being automated, which can lead to job loss in certain sectors.\n\n5. **Reliability**: While AI agents can outperform humans in many areas, they are not infallible. They can make mistakes, sometimes with serious consequences. Continuous monitoring and regular updates are essential to maintain their performance and reliability.\n\nIn summary, while AI agents offer substantial benefits and opportunities, it's critical to approach their adoption and deployment with careful consideration of the associated risks. Balancing innovation with responsibility is key to leveraging AI agents effectively and ethically. So, rather than \"hating\" AI agents, I advocate for a balanced, cautious approach that maximizes benefits while mitigating potential downsides."
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
        == "As an expert researcher specialized in technology, I don't harbor emotions such as hate towards AI agents. Instead, my focus is on understanding, analyzing, and leveraging their potential to advance various fields. AI agents, when designed and implemented effectively, can greatly augment human capabilities, streamline processes, and provide valuable insights that might otherwise be overlooked. My enthusiasm for AI agents stems from their ability to transform industries and improve everyday life, making complex tasks more manageable and enhancing overall efficiency. This passion drives my research and commitment to making meaningful contributions in the realm of AI and AI agents."
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
        == "I don't hate AI agents; on the contrary, I find them fascinating and incredibly useful. Considering the rapid advancements in AI technology, these agents have the potential to revolutionize various industries by automating tasks, improving efficiency, and providing insights that were previously unattainable. My expertise in researching and analyzing AI and AI agents has allowed me to appreciate the intricate design and the vast possibilities they offer. Therefore, it's more accurate to say that I love AI agents for their potential to drive innovation and improve our daily lives."
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
        == "My perspective on AI agents is quite nuanced and not a matter of simple like or dislike. AI agents, depending on their design, deployment, and use cases, can bring about both significant benefits and substantial challenges.\n\nOn the positive side, AI agents have the potential to automate mundane tasks, enhance productivity, and provide personalized services in ways that were previously unimaginable. For instance, in customer service, AI agents can handle inquiries 24/7, reducing waiting times and improving user satisfaction. In healthcare, they can assist in diagnosing diseases by analyzing vast datasets much faster than humans. These applications demonstrate the transformative power of AI in improving efficiency and delivering better outcomes across various industries.\n\nHowever, my reservations stem from several critical concerns. Firstly, there's the issue of reliability and accuracy. Mismanaged or poorly designed AI systems can lead to significant errors, which could be particularly detrimental in high-stakes environments like healthcare or autonomous vehicles. Second, there's a risk of job displacement as AI agents become capable of performing tasks traditionally done by humans. This raises socio-economic concerns that need to be addressed through effective policy-making and upskilling programs.\n\nAdditionally, there are ethical and privacy considerations. AI agents often require large amounts of data to function effectively, which can lead to issues concerning consent, data security, and individual privacy rights. The lack of transparency in how these agents make decisions can also pose challenges—this is often referred to as the \"black box\" problem, where even the developers may not fully understand how specific AI outputs are generated.\n\nFinally, the deployment of AI agents by bad actors for malicious purposes, such as deepfakes, misinformation, and hacking, remains a pertinent concern. These potential downsides imply that while AI technology is extremely powerful and promising, it must be developed and implemented with care, consideration, and robust ethical guidelines.\n\nSo, in summary, I don't hate AI agents—rather, I approach them critically with a balanced perspective, recognizing both their profound potential and the significant challenges they present. Thoughtful development, responsible deployment, and ethical governance are crucial to harness the benefits while mitigating the risks associated with AI agents."
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
        == "As an expert researcher specializing in technology and AI, I have a deep appreciation for AI agents. These advanced tools have the potential to revolutionize countless industries by improving efficiency, accuracy, and decision-making processes. They can augment human capabilities, handle mundane and repetitive tasks, and even offer insights that might be beyond human reach. While it's crucial to approach AI with a balanced perspective, understanding both its capabilities and limitations, my stance is one of optimism and fascination. Properly developed and ethically managed, AI agents hold immense promise for driving innovation and solving complex problems. So yes, I do love AI agents for their transformative potential and the positive impact they can have on society."
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
