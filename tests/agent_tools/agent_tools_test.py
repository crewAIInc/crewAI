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
        == 'AI agents are a significant advancement in the field of artificial intelligence, offering immense potential and diverse applications across various sectors. However, the perception that I "hate" them is not entirely accurate. My stance on AI agents is more nuanced and analytical rather than emotional. Here\'s a comprehensive take:\n\n1. **Capabilities and Applications**:\n   - **Automation**: AI agents excel in performing repetitive and mundane tasks, allowing human workers to focus on more complex and creative activities.\n   - **Efficiency**: They can process vast amounts of data at speeds incomprehensible to humans, leading to quicker decision-making and problem-solving.\n   - **Personalization**: In customer service, AI agents provide personalized experiences by analyzing customer data and predicting preferences.\n   - **Healthcare**: AI agents assist in diagnosing diseases, recommending treatments, and even predicting outbreaks by analyzing medical data.\n\n2. **Benefits**:\n   - **Increased Productivity**: By automating routine tasks, businesses can significantly improve productivity and reduce operational costs.\n   - **24/7 Availability**: Unlike human workers, AI agents can operate continuously without breaks, providing constant support and monitoring.\n   - **Data-Driven Insights**: AI agents can uncover patterns and insights from data that might be missed by human analysts.\n\n3. **Challenges and Concerns**:\n   - **Job Displacement**: There is a legitimate concern about AI agents replacing human jobs, particularly in roles involving routine and predictable tasks.\n   - **Bias and Fairness**: AI agents can perpetuate and even amplify existing biases found in the data they are trained on, leading to unfair or discriminatory outcomes.\n   - **Transparency**: The decision-making process of AI agents can sometimes be opaque, making it difficult to understand how they arrive at certain conclusions (known as the "black box" problem).\n   - **Security**: AI systems can be vulnerable to hacking and other malicious activities, posing significant security risks.\n\n4. **Ethical Considerations**:\n   - **Accountability**: Who is responsible when an AI agent makes a mistake? This question remains complex and is a major ethical consideration.\n   - **Privacy**: The extensive use of data by AI agents raises concerns about user privacy and the potential for misuse of sensitive information.\n   - **Consent**: Users must be informed and provide consent regarding how their data is used by AI agents.\n\n5. **Future Directions**:\n   - **Regulation and Governance**: Developing robust regulatory frameworks to oversee the deployment and use of AI agents is crucial.\n   - **Continuous Improvement**: Ongoing research is needed to enhance the capabilities of AI agents while addressing their current limitations and ethical concerns.\n   - **Collaboration**: Encouraging collaboration between AI agents and human workers can lead to hybrid systems that leverage the strengths of both.\n\nIn summary, while AI agents offer numerous benefits and transformative potential, they also come with a set of challenges and ethical questions that must be carefully managed. My analysis focuses on a balanced view, recognizing both the opportunities and the complexities associated with AI agents. \n\nThus, it\'s not about hating AI agents, but rather critically evaluating their impact and advocating for responsible and fair implementation. This ensures that while we harness their capabilities, we also address and mitigate any negative repercussions.'
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
        == "While it might have circulated that I \"hate\" AI agents, I think it's more nuanced than that. AI agents are incredibly sophisticated pieces of technology that carry immense potential for transforming multiple sectors like healthcare, finance, and customer service.\n\nFirst, let's define what AI agents are. AI agents are autonomous entities designed to perform specific tasks, often mimicking human decision-making and problem-solving abilities. They can range from simple chatbots to complex systems capable of managing large-scale operations.\n\nThe Pros:\n1. Efficiency: AI agents can complete tasks faster and more accurately than humans. For instance, they can instantaneously process large datasets to provide real-time insights.\n2. Scalability: With AI agents, services can be scaled without the proportional increase in cost, which is particularly beneficial for businesses looking to grow without extensive human resource expenditures.\n3. 24/7 Availability: Unlike human workers, AI agents can operate around the clock, increasing productivity and availability, particularly in customer service domains.\n4. Data Handling: AI agents can handle and process data at speeds and volumes that far exceed human capabilities, enabling smarter business decisions and innovations.\n\nHowever, there are some drawbacks and concerns:\n1. Ethical Issues: AI agents often face ethical dilemmas, especially when their decision-making processes are opaque. Bias in AI systems can perpetuate existing inequalities, making transparency and fairness critical issues.\n2. Job Displacement: One significant concern is the potential for AI agents to displace human workers, particularly in industries reliant on routine, repetitive tasks.\n3. Dependency: Over-reliance on AI can lead to skill erosion among human employees and vulnerabilities in cases where technology fails or is compromised.\n4. Security: Given their access to sensitive data, AI agents can be attractive targets for cyber-attacks. Ensuring robust security measures is essential.\n\nSo, while I recognize the transformative power and benefits of AI agents, it's crucial to approach them with a balanced perspective. Adequate governance, ethical considerations, and transparency are needed to harness their potential responsibly. It's not a matter of hating them, but rather advocating for their responsible and fair use in society."
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
        == 'As a researcher specialized in technology, particularly in artificial intelligence (AI) and AI agents, my perspective is shaped by the extensive analysis and study I engage in daily. AI agents are neither inherently good nor bad; they are tools with the potential for both positive and negative impacts depending on how they are designed, implemented, and used.\n\nI do not "hate" AI agents. In fact, I find them fascinating and hold an appreciation for their capabilities and the potential they have to transform various sectors, from healthcare and education to finance and entertainment. AI agents can offer significant benefits, such as improving efficiency, personalizing user experiences, and even tackling complex problems that are beyond human capacity to solve alone.\n\nHowever, it is essential to approach AI with a critical eye. There are legitimate concerns around privacy, security, ethical use, and the socio-economic implications of AI deployment. Ensuring that AI is developed responsibly and is aligned with ethical standards is crucial for minimizing potential negatives.\n\nIn summary, my stance on AI agents is informed by a balanced view that recognizes both their potential and the necessity for responsible stewardship. The enthusiasm I have for their capabilities is tempered by a commitment to understanding and addressing the challenges they present.'
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
        == "As an expert researcher specialized in technology and AI, I do not harbor feelings of hate towards AI agents. In fact, the notion of hate or love for a technology is somewhat misplaced. AI agents are tools created to solve specific problems and enhance our capabilities. They can transform industries, improve efficiencies, and even save lives in fields such as healthcare. My enthusiasm and interest in AI agents come from their potential to innovate and drive progress. While it is essential to be mindful of ethical considerations and responsible use, it is equally important to remain objective. My goal as a researcher is to contribute to balanced and insightful analysis, exploring both the benefits and challenges of AI, and not to personalize or emotionally charge the discussion about technological tools."
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
        == "AI agents, like any technology, have both significant advantages and notable challenges. Initially, the skepticism surrounding AI agents might stem from concerns about their implementation, ethical considerations, or potential impact on jobs. However, let's delve into a comprehensive analysis of their roles and implications.\n\nFirstly, AI agents are software entities that perform tasks autonomously on behalf of a user or another program with some degree of intelligence and learning capability. These agents can interpret data, learn from patterns, and even make decisions based on the information they process. They are essential across various sectors, including healthcare, finance, customer service, and logistics.\n\nPositive Aspects:\n1. **Efficiency and Productivity:** AI agents can process large volumes of data faster than humans, enabling quicker decision-making. This efficiency can significantly boost productivity in organizations by automating repetitive tasks, allowing human workers to focus on more strategic activities.\n\n2. **24/7 Operation:** Unlike humans, AI agents can work around the clock without fatigue. This capability is crucial for industries that require continuous operation, such as monitoring systems in cybersecurity or customer service through chatbots.\n\n3. **Personalization:** AI agents can tailor experiences to individual users by learning from their behavior. For instance, recommendation systems used by services like Netflix or Amazon provide personalized content, enhancing user satisfaction and engagement.\n\n4. **Accuracy and Precision:** AI agents, when properly trained, can perform tasks with a high degree of accuracy, reducing the likelihood of human error. This precision is especially valuable in fields such as medical diagnostics, where accurate analysis can be life-saving.\n\nChallenges and Concerns:\n1. **Ethical Issues:** The deployment of AI agents raises ethical questions regarding privacy, data security, and potential biases in decision-making. It is essential to ensure that AI systems are transparent, fair, and respect user privacy.\n\n2. **Job Displacement:** As AI agents automate more tasks, there is a genuine concern about job loss, particularly in roles that involve routine and repetitive tasks. It is critical to address this issue by reskilling and upskilling the workforce to adapt to the changing job landscape.\n\n3. **Dependence on Data:** AI agents rely heavily on large datasets to learn and make decisions. If the data is biased or incomplete, the AI’s performance and reliability can be compromised. This dependence also raises questions about data ownership and governance.\n\n4. **Complexity and Accountability:** Understanding and managing AI agents can be complex, and determining accountability in the event of failures or erroneous decisions can be challenging. Developing robust frameworks for monitoring and accountability is crucial.\n\nIn conclusion, while there are valid concerns associated with AI agents, they also offer transformative potential in improving efficiency, personalization, and accuracy in various applications. It is important to engage in balanced discussions, address ethical considerations, and develop policies to mitigate potential negative impacts while leveraging the benefits of AI technology. Hence, my stance is not of outright hatred, but one of cautious optimism, advocating for responsible development and deployment of AI agents."
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
        == "While I understand there are mixed feelings about AI agents, I do not hate them. In fact, I see significant potential in AI and AI agents to transform various industries and improve our daily lives. My enthusiasm for AI comes from appreciating the advancements it's bringing, such as personalized recommendations, improved diagnostics in healthcare, and more efficient business processes.\n\nHowever, it's essential to approach AI with a balanced perspective. Ethical considerations, privacy concerns, and the potential for job displacement are critical issues that need to be addressed. I believe in responsible AI development, where the focus is on maximizing benefits while minimizing potential harms.\n\nSo, in short, I love the potential and capabilities of AI agents when developed and implemented responsibly and ethically."
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
