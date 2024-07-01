---
title: crewAI Train
description: Learn how to train your crewAI agents by giving them feedback early on and get consistent results.
---

## Introduction
The training feature in CrewAI allows you to train your AI agents using the command-line interface (CLI). By running the command `crewai train -n <n_iterations>`, you can specify the number of iterations for the training process.

During training, CrewAI utilizes techniques to optimize the performance of your agents along with human feedback. This helps the agents improve their understanding, decision-making, and problem-solving abilities.

To use the training feature, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where your CrewAI project is located.
3. Run the following command:

```shell
crewai train -n <n_iterations>
```

Replace `<n_iterations>` with the desired number of training iterations. This determines how many times the agents will go through the training process.

Remember to also replace the placeholder inputs with the actual values you want to use on the main.py file in the `train` function.

```python
def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs"}
    try:
        ProjectCreationCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)
    ...
```

It is important to note that the training process may take some time, depending on the complexity of your agents and will also require your feedback on each iteration.

Once the training is complete, your agents will be equipped with enhanced capabilities and knowledge, ready to tackle complex tasks and provide more consistent and valuable insights.

Remember to regularly update and retrain your agents to ensure they stay up-to-date with the latest information and advancements in the field.

Happy training with CrewAI!