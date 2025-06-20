# src/crewai/tools/collaboration_optimizer.py

from crewai.tools import BaseTool
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

class AgentCollaborationEnv(gym.Env):
    def __init__(self, num_agents: int = 3):
        super(AgentCollaborationEnv, self).__init__()
        self.num_agents = num_agents
        self.observation_space = Box(low=0, high=1, shape=(self.num_agents,), dtype=np.float32)
        self.action_space = Discrete(self.num_agents * 2)
        self.state = np.zeros(self.num_agents, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.state = np.random.rand(self.num_agents).astype(np.float32)
        return self.state, {}

    def step(self, action):
        self.state = np.random.rand(self.num_agents).astype(np.float32)
        reward = float(np.mean(self.state))
        terminated = np.random.rand() > 0.95
        truncated = False
        return self.state, reward, terminated, truncated, {}


class CollaborationOptimizerTool(BaseTool):
    name: str = "collaboration_optimizer"
    description: str = "Optimizes collaboration strategies among agents using reinforcement learning."

    def _run(self, num_agents: int = 3, timesteps: int = 2000):
        env = AgentCollaborationEnv(num_agents)
        check_env(env, warn=True)

        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps)

        # Evaluation phase (returns average reward over 5 steps)
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(5):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
              break
        avg_reward = total_reward / 5.0

        return f"Average collaboration reward for {num_agents} agents: {avg_reward:.4f}"
