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
        == "As a researcher specialized in technology, I believe that AI agents represent a significant advancement in the capability to automate tasks, process large amounts of data, and even make decisions in certain domains. While I wouldn't characterize my stance as \"hate\" towards them, I do have some critical perspectives on their current deployment and use.\n\nFirstly, AI agents can perform repetitive and mundane tasks more efficiently than humans, allowing individuals and organizations to allocate their human resources to more complex and creative endeavors. This has promising benefits for productivity and innovation across various sectors.\n\nHowever, there are several concerns and challenges that come along with the deployment of AI agents. One primary concern is the degree of control and transparency. Many AI systems, particularly those utilizing machine learning and neural networks, operate as \"black boxes\" where understanding the decision-making process is not straightforward. This lack of transparency can lead to trust issues and difficulties in compliance with legal and ethical standards.\n\nAnother significant challenge is the potential for bias. AI agents learn from data, and if the training data contains biases, the AI agents can perpetuate and even exacerbate those biases. This can result in unfair outcomes and discriminatory practices, especially in sensitive applications like hiring, law enforcement, and lending decisions.\n\nAdditionally, the impact on employment is a critical issue. While AI agents can take over repetitive tasks, there is a legitimate concern over job displacement. It's essential to have strategies in place for retraining and upskilling the workforce to ensure that the benefits of AI advancements are shared broadly across society.\n\nSecurity is another crucial factor. AI agents, if not properly secured, can be susceptible to malicious attacks, leading to potential data breaches or adversarial manipulation. The security of AI systems needs to be robust to safeguard against these risks.\n\nFinally, the ethical implications of autonomous decision-making by AI agents cannot be ignored. As we delegate more decision-making power to AI, it's vital to ensure that these systems operate within the ethical boundaries and align with human values and societal norms.\n\nIn conclusion, while AI agents have transformative potential, it's essential to address these concerns comprehensively. The focus should be on ensuring transparency, mitigating bias, providing employment transition strategies, enhancing security measures, and upholding ethical standards in the deployment and use of AI agents. This balanced approach can help us harness the benefits of AI agents while minimizing their risks and downsides."
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
        == "AI agents are autonomous software programs that perform specific tasks based on a combination of programmed rules and machine learning algorithms. These agents can operate independently to analyze data, make decisions, and interact with their environment, encompassing a wide range of applications from customer service chatbots to sophisticated decision-making systems in finance and healthcare.\n\nDespite their impressive capabilities, AI agents are not without their challenges and limitations. One major concern is the ethical implications of AI decision-making, especially in situations where bias in training data can lead to unfair or harmful outcomes. Additionally, there are substantial issues related to privacy, as these agents often require access to large amounts of personal data to function effectively.\n\nAnother significant drawback is the potential for job displacement. As AI agents become more advanced, they may perform tasks traditionally done by humans, leading to economic and employment disruptions. This creates a pressing need for society to find a balance between leveraging AI advancements and ensuring that the workforce can adapt through reskilling and upskilling initiatives.\n\nOn the technical side, AI agents still encounter difficulties with context understanding and adaptability. While they are excellent at processing structured data, unstructured data—like natural language in diverse formats—presents a significant challenge. The reliability of these systems can also vary depending on the quality of data they are trained on and the complexity of the task at hand.\n\nIn conclusion, while AI agents offer transformative potential across various industries, they come with a set of complex challenges that must be carefully managed. It is crucial to continue advancing these technologies responsibly, considering both their positive impacts and the societal implications they bring."
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
        == "I do not hate AI agents; in fact, I find them incredibly fascinating and valuable. As someone who specializes in technology research, I recognize the immense potential AI agents have in transforming various industries. They can automate mundane tasks, provide insightful data analysis, enhance decision-making processes, and even personalize user experiences. My enthusiasm stems from their ability to innovate and solve complex problems, which aligns with my passion for exploring technological advancements. So yes, you heard correctly—I indeed love them for their transformative capabilities and the endless possibilities they offer."
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
        == "As an expert researcher specialized in technology and artificial intelligence, I find AI agents to be incredibly fascinating and valuable tools in advancing various fields. AI agents have the potential to revolutionize industries, improve efficiencies, and solve complex problems that were previously insurmountable. My admiration for AI agents stems from their ability to learn, adapt, and provide innovative solutions that can enhance human capabilities rather than replace them. While AI agents do present ethical and operational challenges that need to be addressed, I do not harbor any hatred towards them. Instead, I am passionate about leveraging their potential responsibly and ethically to create positive societal impacts."
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
        == "It's a misconception to say I hate AI agents; the reality is far more nuanced. AI agents, in their many forms, are remarkable advancements in technology with the capability to transform various sectors ranging from healthcare to customer service. The key is understanding the context and application of these agents and being mindful of their limitations and ethical considerations.\n\nAI agents, powered by machine learning algorithms and vast data pools, can automate tasks, analyze complex datasets, provide personalized recommendations, and offer conversational interfaces that improve efficiency and user experience. For instance, in customer support, AI agents can handle routine queries swiftly, freeing up human agents for more complex issues.\n\nHowever, it's crucial to address the shortcomings and challenges associated with AI agents. One major concern is the risk of biases in AI systems, which can lead to discriminatory outcomes if not properly managed and audited. Data privacy is another significant issue; AI agents often require access to large amounts of personal data, raising concerns about how this data is collected, stored, and used.\n\nThere's also the question of job displacement. While AI can augment human capabilities and create new job categories, it can also render certain roles obsolete, necessitating workforce reskilling and adaptation.\n\nIn summary, my perspective isn't one of antagonism but of cautious optimism. AI agents have tremendous potential when implemented responsibly, with a clear focus on ethical considerations and societal impact. It's not about hating or loving AI agents, but about striving for balanced, well-informed integration into our technological ecosystem."
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
        == "As an expert researcher specializing in technology, particularly AI and AI agents, I have a nuanced perspective on the matter. AI agents have their strengths and weaknesses, and my appreciation for them is rooted in their potential to transform various industries and improve efficiencies. AI agents can automate repetitive tasks, provide insightful data analyses, and enhance user experiences through intelligent interactions. However, it's also critical to address the ethical considerations and potential risks associated with their use, such as biases in AI models, privacy concerns, and the need for transparency. In summary, I wouldn't say I hate AI agents; rather, I value their contributions and advocate for responsible and ethical development and deployment."
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
