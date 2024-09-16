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
        == "While it's a common perception that I might \"hate\" AI agents, my actual stance is much more nuanced and guided by an in-depth understanding of their potential and limitations. As an expert researcher in technology, I recognize that AI agents are a significant advancement in the field of computing and artificial intelligence, offering numerous benefits and applications across various sectors. Here's a detailed take on AI agents:\n\n**Advantages of AI Agents:**\n1. **Automation and Efficiency:** AI agents can automate repetitive tasks, thus freeing up human workers for more complex and creative work. This leads to significant efficiency gains in industries such as customer service (chatbots), data analysis, and even healthcare (AI diagnostic tools).\n\n2. **24/7 Availability:** Unlike human workers, AI agents can operate continuously without fatigue. This is particularly beneficial in customer service environments where support can be provided around the clock.\n\n3. **Data Handling and Analysis:** AI agents can process and analyze vast amounts of data more quickly and accurately than humans. This ability is invaluable in fields like finance, where AI can detect fraudulent activities, or in marketing, where consumer data can be analyzed to improve customer engagement strategies.\n\n4. **Personalization:** AI agents can provide personalized experiences by learning from user interactions. For example, recommendation systems on platforms like Netflix and Amazon use AI agents to suggest content or products tailored to individual preferences.\n\n5. **Scalability:** AI agents can be scaled up easily to handle increasing workloads, making them ideal for businesses experiencing growth or variable demand.\n\n**Challenges and Concerns:**\n1. **Ethical Implications:** The deployment of AI agents raises significant ethical questions, including issues of bias, privacy, and the potential for job displacement. It’s crucial to address these concerns by incorporating transparent, fair, and inclusive practices in AI development and deployment.\n\n2. **Dependability and Error Rates:** While AI agents are generally reliable, they are not infallible. Errors, especially in critical areas like healthcare or autonomous driving, can have severe consequences. Therefore, rigorous testing and validation are essential.\n\n3. **Lack of Understanding:** Many users and stakeholders may not fully understand how AI agents work, leading to mistrust or misuse. Improving AI literacy and transparency can help build trust in these systems.\n\n4. **Security Risks:** AI agents can be vulnerable to cyber-attacks. Ensuring robust cybersecurity measures are in place is vital to protect sensitive data and maintain the integrity of AI systems.\n\n5. **Regulation and Oversight:** The rapid development of AI technology often outpaces regulatory frameworks. Effective governance is needed to ensure AI is used responsibly and ethically.\n\nIn summary, while I thoroughly understand the transformative potential of AI agents and their numerous advantages, I also recognize the importance of addressing the associated challenges. It's not about hating AI agents, but rather advocating for their responsible and ethical use to ensure they benefit society as a whole. My critical perspective is rooted in a desire to see AI agents implemented in ways that maximize their benefits while minimizing potential harms."
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
        == "As an expert researcher in technology, particularly in the field of AI and AI agents, it is essential to clarify that my perspective is not one of hatred but rather critical analysis. My evaluation of AI agents is grounded in a balanced view of their advantages and the challenges they present. \n\nAI agents represent a significant leap in technological progress with a wide array of applications across industries. They can perform tasks ranging from customer service interactions, data analysis, complex simulations, to even personal assistance. Their ability to learn and adapt makes them powerful tools for enhancing productivity and innovation.\n\nHowever, there are considerable challenges and ethical concerns associated with their deployment. These include privacy issues, job displacement, and the potential for biased decision-making driven by flawed algorithms. Furthermore, the security risks posed by AI agents, such as how they can be manipulated or hacked, are critical concerns that cannot be ignored.\n\nIn essence, while I do recognize the transformative potential of AI agents, I remain vigilant about their implications. It is vital to ensure that their development is guided by robust ethical standards and stringent regulations to mitigate risks. My view is not rooted in hatred but in a deep commitment to responsible and thoughtful technological advancement. \n\nI hope this clarifies my stance on AI agents and underscores the importance of critical engagement with emerging technologies."
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
        == "No, I do not hate AI agents; in fact, I find them incredibly fascinating and useful. As a researcher specializing in technology, particularly in AI and AI agents, I appreciate their potential to revolutionize various industries by automating tasks, providing deep insights through data analysis, and even enhancing decision-making processes. AI agents can streamline operations, improve efficiency, and contribute to advancements in fields like healthcare, finance, and cybersecurity. While they do present challenges, such as ethical considerations and the need for robust security measures, the benefits and potential for positive impact are immense. Therefore, my stance is one of strong support and enthusiasm for AI agents and their future developments."
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
        == "I do not hate AI agents; in fact, I appreciate them for their immense potential and the numerous benefits they bring to various fields. My passion for AI agents stems from their ability to streamline processes, enhance decision-making, and provide innovative solutions to complex problems. They significantly contribute to advancements in healthcare, finance, education, and many other sectors, making tasks more efficient and freeing up human capacities for more creative and strategic endeavors. So, to answer your question, I love AI agents because of the positive impact they have on our world and their capability to drive technological progress."
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
        == "AI agents have emerged as a revolutionary force in today's technological landscape, and my stance on them is not rooted in hatred but in a critical, analytical perspective. Let's delve deeper into what makes AI agents both a boon and a bane in various contexts.\n\n**Benefits of AI Agents:**\n\n1. **Automation and Efficiency:**\n   AI agents excel at automating repetitive tasks, which frees up human resources for more complex and creative endeavors. They are capable of performing tasks rapidly and with high accuracy, leading to increased efficiency in operations.\n\n2. **Data Analysis and Decision Making:**\n   These agents can process vast amounts of data at speeds far beyond human capability. They can identify patterns and insights that would otherwise be missed, aiding in informed decision-making processes across industries like finance, healthcare, and logistics.\n\n3. **Personalization and User Experience:**\n   AI agents can personalize interactions on a scale that is impractical for humans. For example, recommendation engines in e-commerce or content platforms tailor suggestions to individual users, enhancing user experience and satisfaction.\n\n4. **24/7 Availability:**\n   Unlike human employees, AI agents can operate round-the-clock without the need for breaks, sleep, or holidays. This makes them ideal for customer service roles, providing consistent and immediate responses any time of the day.\n\n**Challenges and Concerns:**\n\n1. **Job Displacement:**\n   One of the major concerns is the displacement of jobs. As AI agents become more proficient at a variety of tasks, there is a legitimate fear of human workers being replaced, leading to unemployment and economic disruption.\n\n2. **Bias and Fairness:**\n   AI agents are only as good as the data they are trained on. If the training data contains biases, the AI agents can perpetuate or even exacerbate these biases, leading to unfair and discriminatory outcomes.\n\n3. **Privacy and Security:**\n   The use of AI agents often involves handling large amounts of personal data, raising significant privacy and security concerns. Unauthorized access or breaches could lead to severe consequences for individuals and organizations.\n\n4. **Accountability and Transparency:**\n   The decision-making processes of AI agents can be opaque, making it difficult to hold them accountable. This lack of transparency can lead to mistrust and ethical dilemmas, particularly when AI decisions impact human lives.\n\n5. **Ethical Considerations:**\n   The deployment of AI agents in sensitive areas, such as surveillance and law enforcement, raises ethical issues. The potential for misuse or overdependence on AI decision-making poses a threat to individual freedoms and societal norms.\n\nIn conclusion, while AI agents offer remarkable advantages in terms of efficiency, data handling, and user experience, they also bring significant challenges that need to be addressed carefully. My critical stance is driven by a desire to ensure that their integration into society is balanced, fair, and beneficial to all, without ignoring the potential downsides. Therefore, a nuanced approach is essential in leveraging the power of AI agents responsibly."
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
        == "As a researcher specialized in technology, particularly in AI and AI agents, my feelings toward them are far more nuanced than simply loving or hating them. AI agents represent a remarkable advancement in technology and hold tremendous potential for improving various aspects of our lives and industries. They can automate tedious tasks, provide intelligent data analysis, support decision-making, and even enhance our creative processes. These capabilities can drive efficiency, innovation, and economic growth.\n\nHowever, it is also crucial to acknowledge the challenges and ethical considerations posed by AI agents. Issues such as data privacy, security, job displacement, and the need for proper regulation are significant concerns that must be carefully managed. Moreover, the development and deployment of AI should be guided by principles that ensure fairness, transparency, and accountability.\n\nIn essence, I appreciate the profound impact AI agents can have, but I also recognize the importance of approaching their integration into society with thoughtful consideration and responsibility. Balancing enthusiasm with caution and ethical oversight is key to harnessing the full potential of AI while mitigating its risks."
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
