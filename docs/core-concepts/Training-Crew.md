---
title: crewAI Train
description: Learn how to train your crewAI agents by giving them feedback early on and get consistent results.
---

## Introduction
The training feature in CrewAI allows you to train your AI agents using the command-line interface (CLI). By running the command `crewai train -n <n_iterations>`, you can specify the number of iterations for the training process.

During training, CrewAI utilizes techniques to optimize the performance of your agents along with human feedback. This helps the agents improve their understanding, decision-making, and problem-solving abilities.

### Training Your Crew Using the CLI
To use the training feature, follow these steps:

1. Open your terminal or command prompt.
2. Navigate to the directory where your CrewAI project is located.
3. Run the following command:

```shell
crewai train -n <n_iterations>
```

### Training Your Crew Programmatically
To train your crew programmatically, use the following steps:

1. Define the number of iterations for training.
2. Specify the input parameters for the training process.
3. Execute the training command within a try-except block to handle potential errors.

```python
    n_iterations = 2
    inputs = {"topic": "CrewAI Training"}

    try:
        YourCrewName_Crew().crew().train(n_iterations= n_iterations, inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")
```

!!! note "Replace `<n_iterations>` with the desired number of training iterations. This determines how many times the agents will go through the training process."


### Key Points to Note:
- **Positive Integer Requirement:** Ensure that the number of iterations (`n_iterations`) is a positive integer. The code will raise a `ValueError` if this condition is not met.
- **Error Handling:** The code handles subprocess errors and unexpected exceptions, providing error messages to the user.

It is important to note that the training process may take some time, depending on the complexity of your agents and will also require your feedback on each iteration.

Once the training is complete, your agents will be equipped with enhanced capabilities and knowledge, ready to tackle complex tasks and provide more consistent and valuable insights.

Remember to regularly update and retrain your agents to ensure they stay up-to-date with the latest information and advancements in the field.

Happy training with CrewAI!