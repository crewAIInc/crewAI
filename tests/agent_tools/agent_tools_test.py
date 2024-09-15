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
        == "AI agents are autonomous programs that perform tasks or services on behalf of users with a considerable degree of autonomy, often based on intelligence derived from data and analytics. They can operate within specific domains or across multiple areas, executing tasks ranging from mundane repetitive actions to complex decision-making processes. Given their applications, it's essential to examine both the benefits and criticisms to provide a balanced perspective.\n\n## The Positive Aspects of AI Agents\n\n1. **Efficiency and Productivity:**\n   AI agents can handle tedious and repetitive tasks efficiently, freeing up human workers to focus on more complex and creative endeavors. This results in significant time and cost savings for businesses.\n\n2. **24/7 Availability:**\n   AI agents can work continuously without breaks, providing round-the-clock service. This is especially beneficial for customer support, where they can resolve issues and answer queries at any time.\n\n3. **Consistency and Accuracy:**\n   Unlike humans, AI agents do not suffer from fatigue and can perform tasks without error, ensuring a high level of consistency in operations.\n\n4. **Data Handling and Analysis:**\n   AI agents can process and analyze vast amounts of data quickly, identifying patterns and generating insights that would be otherwise difficult to discern manually.\n\n## Criticisms and Concerns\n\n1. **Job Displacement:**\n   One of the primary concerns is that AI agents could replace human jobs, leading to unemployment. While they create new opportunities in AI development and maintenance, the transition can be disruptive.\n\n2. **Ethical Considerations:**\n   With increasing autonomy, ensuring that AI agents act ethically and within the bounds of societal norms is a challenge. Misuse of AI for malicious activities or biased decision-making are valid concerns that need rigorous oversight.\n\n3. **Lack of Emotional Intelligence:**\n   AI agents, no matter how advanced, lack the ability to empathize and understand human emotions fully. This can be a drawback in areas that require delicate human interaction, such as mental health support.\n\n4. **Dependence and Reliability:**\n   Over-reliance on AI agents might lead to significant issues if they malfunction or are compromised. Ensuring reliability and establishing fail-safes is critical.\n\n5. **Security and Privacy:**\n   AI agents typically require access to large data sets, often containing sensitive information. Ensuring robust data security and user privacy is paramount to prevent misuse or data breaches.\n\n## Conclusion\n\nWhile AI agents offer remarkable benefits in terms of efficiency, accuracy, and data handling, their integration into society must be managed carefully to address ethical concerns and potential negative impacts on employment and human interaction. The onus is on developers, policymakers, and society at large to ensure that AI agents are designed and deployed responsibly, balancing innovation with the broader implications for humanity. \n\nDespite the apprehensions, dismissing AI agents entirely overlooks their transformative potential. Instead, the focus should be on harnessing these tools' capabilities while mitigating their risks, ensuring they serve to enhance rather than hinder human progress."
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
        == "While it may have come across that I have a strong dislike for AI agents, the reality is more nuanced. My stance isn't one of outright hate, but rather one of concern and critical examination. AI agents offer immense potential in automating tasks, improving efficiency, and driving innovations across various fields such as healthcare, finance, and customer service. They can handle repetitive tasks, provide data-driven insights, and even enhance user experiences through personalized interactions.\n\nHowever, my apprehensions lie in the ethical and societal implications associated with AI agents. Issues such as data privacy, security, and the potential for biased outputs due to flawed algorithms are significant concerns. Moreover, the rapid proliferation of AI agents raises questions about job displacement and the future of work, potentially leading to socioeconomic disparities if not managed thoughtfully.\n\nTransparency and accountability in AI development are paramount. AI agents must be designed and implemented with clear ethical guidelines, rigorous testing, and improved explainability to ensure they act in the best interests of society. Collaboration between technologists, ethicists, policymakers, and the general public is crucial to cultivate trust and navigate the complexities associated with AI advancements.\n\nIn summary, my critical stance on AI agents stems from a desire to see responsible and ethical development that maximizes their benefits while mitigating risks. By addressing these concerns head-on, we can harness the full potential of AI agents to create a more equitable and prosperous future."
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
        == "As an expert researcher specialized in technology and AI, it's important to maintain an objective and balanced perspective. I do not harbor emotions such as love or hate towards AI agents. Instead, I appreciate the capabilities and potential they offer.\n\nAI agents can significantly enhance productivity, provide valuable insights through data analysis, and automate repetitive tasks, thereby allowing humans to focus on more complex and creative endeavors. However, it's also crucial to address ethical considerations, potential biases, and the implications of their deployment in various sectors.\n\nTherefore, my stance isn't one of emotional attachment but rather a professional appreciation of both the opportunities and challenges presented by AI agents. This balanced view helps in conducting thorough and unbiased research, which is essential for advancing our understanding and responsible use of AI technologies."
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
        == 'As a researcher specializing in technology and artificial intelligence, I do not "hate" AI agents. Instead, I have a deep appreciation for their capabilities and potential to transform numerous industries. My professional interest lies in understanding, analyzing, and improving these technologies to benefit society. AI agents can streamline tasks, process vast amounts of data quickly, and even assist in decision-making processes. While they do come with challenges and ethical considerations, my focus is on leveraging their strengths and addressing their limitations to create a more efficient and advanced technological landscape. So, in short, no, I do not hate AI agents; I respect and am fascinated by their potential.'
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
        == 'AI Agents are a diverse set of software programs designed to autonomously carry out tasks, interact with users, and make decisions based on data input. While there seems to be a perception that I dislike AI agents, let\'s delve into a detailed and nuanced view.\n\nFirst, AI agents are remarkable for their ability to handle repetitive tasks efficiently. They can automate mundane tasks, freeing up human workers for more complex, creative, and strategic work. For instance, customer service bots can handle basic inquiries, significantly reducing waiting times and improving customer satisfaction.\n\nMoreover, AI agents excel in data analysis. These agents can process and analyze large volumes of data far more quickly and accurately than humans. This capability is particularly valuable in sectors such as finance, healthcare, and logistics, where rapid and precise decision-making is crucial. For example, in the medical field, AI agents can analyze medical images or patient data to assist in diagnosing diseases, potentially leading to earlier detections and better patient outcomes.\n\nThere are also significant advancements in personalized user experiences through AI agents. These agents can learn from user interactions to tailor recommendations and interactions that better meet individual preferences. Think of how Netflix provides personalized movie recommendations or how AI-driven personalization in e-commerce can suggest products that a user is likely to be interested in.\n\nHowever, the criticism that might be perceived as "hate" stems from valid concerns. One significant issue is the potential for job displacement as AI agents become more capable. While they augment human capabilities, the automation of tasks previously performed by humans can lead to job losses, particularly in sectors reliant on repetitive tasks.\n\nFurthermore, ethical concerns are paramount. The deployment of AI agents often involves massive amounts of data, raising issues around privacy and data security. Additionally, if not properly designed and regulated, AI agents can perpetuate and even exacerbate biases present in their training data, leading to unfair outcomes.\n\nTransparency and accountability are also critical challenges. Many AI models operate as "black boxes," with their decision-making processes being opaque. This lack of transparency can be problematic in scenarios where understanding the rationale behind an AI agent\'s decision is necessary, such as in judicial or medical contexts.\n\nIn summary, while AI agents bring a host of benefits, including increased efficiency, enhanced data analysis, and personalized experiences, they also pose significant challenges. These include potential job displacement, ethical concerns, bias, and lack of transparency. These criticisms are not to dismiss AI agents but to highlight the importance of addressing these issues to ensure that AI develops in a manner that is beneficial, fair, and ethical for society as a whole. This balanced view recognizes the transformative potential of AI agents while urging caution and consideration of the broader societal impacts.'
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
        == "As a researcher deeply specialized in technology and artificial intelligence, I can confidently say that I do not hate AI agents. In fact, I find them incredibly fascinating and valuable. AI agents have the potential to revolutionize numerous industries by improving efficiency, accuracy, and enabling innovations that were previously unimaginable. They can assist in data analysis, automate repetitive tasks, and even contribute to advancements in healthcare, finance, and environmental monitoring, just to name a few. It's important to recognize that, like any technology, AI agents must be developed and used responsibly, with ethical considerations in mind. So, to address the context shared, yes, it’s true—I do love AI agents for their potential to drive positive change and enhance our capabilities in meaningful ways."
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
