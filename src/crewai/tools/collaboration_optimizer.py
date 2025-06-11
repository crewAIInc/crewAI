# collaboration_optimizer.py
# Located at: src/crewai/tools/collaboration_optimizer.py

import numpy as np
import gym
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class AgentCollaborationEnv(gym.Env):
    """
    A Gym environment that simulates collaboration between agents.
    Each agent gets a performance score. The better they collaborate, the higher the cumulative reward.
    """

    def __init__(self, num_agents: int = 3):
        super(AgentCollaborationEnv, self).__init__()
        self.num_agents = num_agents
        self.observation_space = Box(
            low=0, high=1, shape=(self.num_agents,), dtype=np.float32
        )
        self.action_space = Discrete(
            self.num_agents * 2
        )  # Simulate routing or delegation decisions
        self.state = np.zeros(self.num_agents, dtype=np.float32)

    def reset(self):
        self.state = np.random.rand(self.num_agents)
        return self.state

    def step(self, action):
        # Simulate some logic based on action
        self.state = np.random.rand(self.num_agents)
        reward = float(
            np.mean(self.state)
        )  # Reward: better collaboration = higher mean
        done = np.random.rand() > 0.95  # Random termination condition
        return self.state, reward, done, {}


# Optional test harness
if __name__ == "__main__":
    env = AgentCollaborationEnv(num_agents=4)
    check_env(env, warn=True)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2000)
    model.save("agent_collab_optimizer")

    # Run a quick evaluation
    obs = env.reset()
    for _ in range(5):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        print(f"Step reward: {reward:.4f}")
