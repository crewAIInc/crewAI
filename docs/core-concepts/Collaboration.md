---
title: How Agents Collaborate in CrewAI
description: Exploring the dynamics of agent collaboration within the CrewAI framework, focusing on the newly integrated features for enhanced functionality.
---

## Collaboration Fundamentals
!!! note "Core of Agent Interaction"
    Collaboration in CrewAI is fundamental, enabling agents to combine their skills, share information, and assist each other in task execution, embodying a truly cooperative ecosystem.

- **Information Sharing**: Ensures all agents are well-informed and can contribute effectively by sharing data and findings.
- **Task Assistance**: Allows agents to seek help from peers with the required expertise for specific tasks.
- **Resource Allocation**: Optimizes task execution through the efficient distribution and sharing of resources among agents.

## Enhanced Attributes for Improved Collaboration
The `Crew` class has been enriched with several attributes to support advanced functionalities:

- **Language Model Management (`manager_llm`, `function_calling_llm`)**: Manages language models for executing tasks and tools, facilitating sophisticated agent-tool interactions. Note that while `manager_llm` is mandatory for hierarchical processes to ensure proper execution flow, `function_calling_llm` is optional, with a default value provided for streamlined tool interaction.
- **Process Flow (`process`)**: Defines the execution logic (e.g., sequential, hierarchical) to streamline task distribution and execution.
- **Verbose Logging (`verbose`)**: Offers detailed logging capabilities for monitoring and debugging purposes. It supports both integer and boolean types to indicate the verbosity level. For example, setting `verbose` to 1 might enable basic logging, whereas setting it to True enables more detailed logs.
- **Rate Limiting (`max_rpm`)**: Ensures efficient utilization of resources by limiting requests per minute. Guidelines for setting `max_rpm` should consider the complexity of tasks and the expected load on resources.
- **Internationalization Support (`language`, `language_file`)**: Facilitates operation in multiple languages, enhancing global usability. Supported languages and the process for utilizing the `language_file` attribute for customization should be clearly documented.
- **Execution and Output Handling (`full_output`)**: Distinguishes between full and final outputs for nuanced control over task results. Examples showcasing the difference in outputs can aid in understanding the practical implications of this attribute.
- **Callback and Telemetry (`step_callback`, `task_callback`)**: Integrates callbacks for step-wise and task-level execution monitoring, alongside telemetry for performance analytics. The purpose and usage of `task_callback` alongside `step_callback` for granular monitoring should be clearly explained.
- **Crew Sharing (`share_crew`)**: Enables sharing of crew information with CrewAI for continuous improvement and training models. The privacy implications and benefits of this feature, including how it contributes to model improvement, should be outlined.
- **Usage Metrics (`usage_metrics`)**: Stores all metrics for the language model (LLM) usage during all tasks' execution, providing insights into operational efficiency and areas for improvement. Detailed information on accessing and interpreting these metrics for performance analysis should be provided.
- **Memory Usage (`memory`)**: Indicates whether the crew should use memory to store memories of its execution, enhancing task execution and agent learning.
- **Embedder Configuration (`embedder`)**: Specifies the configuration for the embedder to be used by the crew for understanding and generating language. This attribute supports customization of the language model provider.

## Delegation: Dividing to Conquer
Delegation enhances functionality by allowing agents to intelligently assign tasks or seek help, thereby amplifying the crew's overall capability.

## Implementing Collaboration and Delegation
Setting up a crew involves defining the roles and capabilities of each agent. CrewAI seamlessly manages their interactions, ensuring efficient collaboration and delegation, with enhanced customization and monitoring features to adapt to various operational needs.

## Example Scenario
Consider a crew with a researcher agent tasked with data gathering and a writer agent responsible for compiling reports. The integration of advanced language model management and process flow attributes allows for more sophisticated interactions, such as the writer delegating complex research tasks to the researcher or querying specific information, thereby facilitating a seamless workflow.

## Conclusion
The integration of advanced attributes and functionalities into the CrewAI framework significantly enriches the agent collaboration ecosystem. These enhancements not only simplify interactions but also offer unprecedented flexibility and control, paving the way for sophisticated AI-driven solutions capable of tackling complex tasks through intelligent collaboration and delegation.
