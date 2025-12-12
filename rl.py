# rl.py 
"""
This file implements a Reinforcement Learning (RL) pipeline to optimize the hyperparameters
of a Retrieval-Augmented Generation (RAG) system using Proximal Policy Optimization (PPO).
The RL agent learns to select the best 'top_k' retrieval parameter and LLM temperature
"""

import time
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
import abc
from rag import RAGConfig
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, util
from langchain_huggingface import HuggingFaceEmbeddings

from rag_interface import RAGInterface


class Rewarder(abc.ABC):
    """
    Base class for reward scoring.
    """
    @abc.abstractmethod
    def __call__(self, rag_answer: str, gold_answer: str) -> float:
        pass

class DatasetRewardScorer(Rewarder):
    """
    Simple semantic similarity-based scorer.
    """

    def __init__(self, similarity_threshold: float = 0.5):
        self.threshold = similarity_threshold
        print("Loading sentence-transformer model for reward computation...")
        # Load the embedder only once
        self.embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def _cosine_similarity(self, a: str, b: str) -> float:
        emb_a = self.embedder.encode(a, convert_to_tensor=True)
        emb_b = self.embedder.encode(b, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_b).cpu().numpy()[0][0]) # Ensure scalar float result

    def __call__(self, rag_answer: str, gold_answer: str) -> float:
        sim = self._cosine_similarity(rag_answer, gold_answer)
        print(f"Computed similarity: {sim:.4f}")
        return sim

class HumanRewardScorer(Rewarder):
    """
    Placeholder for human-in-the-loop reward scoring.
    """

    def __call__(self, rag_answer: str, gold_answer: str) -> float:
        print("RAG Answer:\n", rag_answer)
        print("Gold Answer:\n", gold_answer)
        while True:
            try:
                score = float(input("Please provide a reward score (0.0 to 1.0): "))
                if 0.0 <= score <= 1.0:
                    return score
                else:
                    print("Score must be between 0.0 and 1.0.")
            except ValueError:
                print("Invalid input. Please enter a numeric value between 0.0 and 1.0.")


@dataclass(frozen=True)
class RagAction:
    """Action space defined only by top_k and llm_temperature."""
    top_k: int
    llm_temperature: float


TOP_K_VALUES = [2, 3, 5, 7, 10, 15, 20, 25, 30]
TEMPERATURES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5] 


ACTION_SPACE = [
    RagAction(k, t)
    for k in TOP_K_VALUES
    for t in TEMPERATURES
]


class Agent(abc.ABC):
    """
    Base class for a reinforcement learning agent.
    """
    @abc.abstractmethod
    def act():
        pass

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic network for discrete action spaces.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 512):
        """
        obs_dim: Dimension of the observation space.]
        n_actions: Number of discrete actions.
        hidden: Number of hidden units in each layer.
        
        """
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions),
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        args:
            obs: (batch_size, obs_dim)
        returns:
            logits:  (batch_size, n_actions)
            value:  (batch_size,)
        """
        logits = self.actor(obs)
        value = self.critic(obs).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor):
        """
        args:
            obs: (batch_size, obs_dim)
        returns:
            action: (batch_size,)
            logp: (batch_size,)
            value: (batch_size,)
        """

        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def logprob_value(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        args:
            obs: (batch_size, obs_dim)
            actions: (batch_size,)
        returns:
            logp: (batch_size,)
            value: (batch_size,)
        """

        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        return logp, value


@dataclass
class PPOConfig:
    lr: float = 1e-4           
    clip_eps: float = 0.15     
    value_coef: float = 0.5
    entropy_coef: float = 0.001  
    batch_size: int = 64
    update_epochs: int = 4
    max_buffer_size: int = 32  
    gamma: float = 0.0
    gae_lambda: float = 0.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """ Stores single-step transitions (stateless PPO). """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.logps = []
        self.values = []

    def add(self, obs, action, reward, logp, value):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.logps.append(logp)
        self.values.append(value)

    def clear(self):
        self.__init__()


# RL Pipeline

class RLPipeline:
    """
    PPO optimization of RAG hyperparameters using automated reward scoring.
    Each question = a single-step episode.
    """

    def __init__(
        self,
        base_config: RAGConfig,
        ppo_cfg: PPOConfig = PPOConfig(),
        action_space: List[RagAction] = ACTION_SPACE,
        reward_scorer: Optional[DatasetRewardScorer] = None,
    ):
        self.base_config = base_config
        self.action_space = action_space
        self.device = torch.device(ppo_cfg.device)

        self.embedding_model = HuggingFaceEmbeddings(model_name=base_config.embedding_model_name)

        # Create Actor-Critic Model
        probe = self.embedding_model.embed_query("probe text")
        obs_dim = len(probe)

        self.ac = ActorCritic(obs_dim=obs_dim, n_actions=len(action_space)).to(self.device)
        self.optim = optim.Adam(self.ac.parameters(), lr=ppo_cfg.lr)
        self.ppo_cfg = ppo_cfg

        # Reward Scorer
        self.scorer = reward_scorer or DatasetRewardScorer()

        # Buffer
        self.buf = ReplayBuffer()
        self.sample_count = 0

        # Reward Logging
        self.reward_window = []
        self.log_every = 10

        # History
        self.history = []

    def _apply_action_to_config(self, action: RagAction) -> RAGConfig:
        cfg = type(self.base_config)(**self.base_config.__dict__)
        
        # Update top_k and llm_temperature from the selected action 
        cfg.top_k = action.top_k
        cfg.llm_temperature = action.llm_temperature
        
        # The following parameters are implicitly carried over from self.base_config
        return cfg

    @torch.no_grad()
    def _select_action(self, obs_np):
        # Convert observation to tensor
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, logp, val = self.ac.act(obs)
        return int(a.item()), float(logp.item()), float(val.item())


    def _ppo_update(self):
        obs = torch.tensor(np.stack(self.buf.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buf.actions, dtype=torch.long, device=self.device)
        old_logps = torch.tensor(self.buf.logps, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buf.rewards, dtype=torch.float32, device=self.device)
        old_values = torch.tensor(self.buf.values, dtype=torch.float32, device=self.device)

        # Advantage and Target Calculation 
        targets = rewards
        adv = rewards - old_values
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.ppo_cfg.update_epochs):
            idx = torch.randperm(len(obs), device=self.device)
            for start in range(0, len(obs), self.ppo_cfg.batch_size):
                batch = idx[start:start + self.ppo_cfg.batch_size]

                b_obs = obs[batch]
                b_act = actions[batch]
                b_old_logp = old_logps[batch]
                b_adv = adv[batch]
                b_targets = targets[batch]

                new_logp, new_value = self.ac.logprob_value(b_obs, b_act)
                ratio = torch.exp(new_logp - b_old_logp)

                clip_eps = self.ppo_cfg.clip_eps
                pg1 = ratio * b_adv
                pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv

                policy_loss = -torch.min(pg1, pg2).mean()
                value_loss = (new_value - b_targets).pow(2).mean()

                logits, _ = self.ac.forward(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()

                loss = policy_loss + self.ppo_cfg.value_coef * value_loss - self.ppo_cfg.entropy_coef * entropy

                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.optim.step()

        self.buf.clear()


    def answer(self, question: str, gold_answer: str) -> Dict[str, Any]:
        """
        Takes a question, selects an action (top_k, llm_temperature), runs RAG to get an answer,
        computes reward, and performs PPO update if needed.
        Returns the answer, reward, latency, and action details.
        """

        obs_np = np.array(self.embedding_model.embed_query(question), dtype=np.float32)

        a_idx, logp, vpred = self._select_action(obs_np)
        action = self.action_space[a_idx]

        cfg = self._apply_action_to_config(action)

        t0 = time.time()
        rag = RAGInterface(cfg)
        answer = rag.ask(question)
        latency = time.time() - t0

        # Automated reward
        if gold_answer is not None:
            reward = self.scorer(answer, gold_answer)
        else:
            # fallback: no label provided
            reward = 0.0

        # Store transition
        self.buf.add(obs_np, a_idx, reward, logp, vpred)
        self.sample_count += 1

        # PPO update
        if self.sample_count % self.ppo_cfg.max_buffer_size == 0:
            self._ppo_update()

        # Reward logging every 10 episodes
        self.reward_window.append(reward)
        if len(self.reward_window) >= self.log_every:
            avg = sum(self.reward_window) / len(self.reward_window)
            print(f"\n=== Average Reward (last {self.log_every} episodes): {avg:.3f} ===\n")
            self.reward_window.clear()

        # Logging
        record = {
            "question": question,
            "action_idx": a_idx,
            "action": asdict(action),
            "reward": reward,
            "latency": latency,
        }
        self.history.append(record)

        return {
            "answer": answer,
            "reward": reward,
            "latency": latency,
            "action_idx": a_idx,
            "action": asdict(action),
        }


    def save(self, path: str):
        torch.save(self.ac.state_dict(), path + ".pt")
        with open(path + ".json", "w") as f:
            json.dump({"history": self.history}, f, indent=2)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path + ".pt", map_location=self.device))