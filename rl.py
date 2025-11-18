# rl_pipeline.py

import time
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from langchain_huggingface import HuggingFaceEmbeddings

# own modules
from rag import RAGConfig
from rag_interface import RAGInterface


# ---- Reward model (HF) ----
from transformers import pipeline

class HFRewardScorer:
    """
    Returns a reward explicitly from console input.
    The human provides feedback: yes → 1.0, no → 0.0.
    """

    def __call__(self, question: str, answer: str) -> float:
        print("\n=== HUMAN FEEDBACK REQUIRED ===")
        print("QUESTION:")
        print(question)
        print("\nANSWER:")
        print(answer)
        print("\nIs this answer helpful and correct? (yes/no)")

        # Loop until valid input
        while True:
            result = input("Your feedback (yes/no): ").strip().lower()

            if result in ["yes", "y", "1", "true", "correct", "helpful"]:
                print("Reward = 1.0")
                return 1.0

            if result in ["no", "n", "0", "false", "incorrect", "bad"]:
                print("Reward = 0.0")
                return 0.0

            print("Invalid input. Please type 'yes' or 'no'.")

# Action space encoded as dataclass
@dataclass(frozen=True)
class RagAction:
    chunk_size: int
    chunk_overlap: int
    top_k: int
    llm_temperature: float

# Define your discrete action space 
CHUNK_SIZES = [400, 600, 800, 1000, 1200]
CHUNK_OVERLAPS = [50, 100, 150, 200]
TOP_K_VALUES = [2, 3, 5, 7]
TEMPERATURES = [0.0, 0.1, 0.2, 0.3]

ACTION_SPACE = [
    RagAction(cs, co, k, temp)
    for cs in CHUNK_SIZES
    for co in CHUNK_OVERLAPS
    for k in TOP_K_VALUES
    for temp in TEMPERATURES
]



# PPO Discrete Actor-Critic 
class ActorCritic(nn.Module):
    """
    Actor–Critic network in PPO for discrete action spaces.

    This module contains two functions:
    
      • The Actor π_θ(a|s):  
        A policy network that outputs a set of logits over the discrete action
        space. These logits parameterize a Categorical distribution from which
        the agent samples an action. 

      • The Critic V_ϕ(s):  
        A value network that estimates the expected return (reward) from a
        given state. It does not directly
        choose actions but guides the actor's learning signal.

    PPO (Schulman et al., 2017) optimizes the policy using:
        - clipped likelihood ratios to prevent destructive policy updates,
        - the critic's state-value estimates to compute advantages,
        - entropy bonuses to encourage exploration.

    Schulman et al. 2017 — "Proximal Policy Optimization Algorithms"
    (https://arxiv.org/abs/1707.06347)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, n_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given observations, returns action logits and state-value estimates.

        args:
            obs: Tensor of shape (batch_size, obs_dim)
        returns:
            logits: Tensor of shape (batch_size, n_actions)
            value: Tensor of shape (batch_size,)
        """

        logits = self.actor(obs) # (batch_size, n_actions)
        value  = self.critic(obs).squeeze(-1) # (batch_size,)
        return logits, value

    def act(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples an action from the policy given observations.

        args:
            obs: Tensor of shape (batch_size, obs_dim)
        returns:
            action: Tensor of shape (batch_size,)
            logp: Tensor of shape (batch_size,)
            value: Tensor of shape (batch_size,)
        """
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logp = dist.log_prob(action)
        return action, logp, value

    def logprob_value(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes log-probabilities of given actions and state-value estimates.

        args:
            obs: Tensor of shape (batch_size, obs_dim)
            actions: Tensor of shape (batch_size,)
        returns:
            logp: Tensor of shape (batch_size,)
            value: Tensor of shape (batch_size,)
        """

        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        return logp, value


@dataclass
class PPOConfig:
    lr: float = 3e-4
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    batch_size: int = 64
    update_epochs: int = 4
    max_buffer_size: int = 2          # how many single-step samples before update
    gamma: float = 0.0                   # stateless ⇒ 0.0
    gae_lambda: float = 0.0              # stateless ⇒ 0.0
    device: str = "cpu"


class ReplayBuffer:
    """
    Stores single-step transitions since episodes are length-1.
    This will be useful for offline trainning.
    """
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



class RLPipeline:
    """
    Stateless PPO controlling a discrete set of RAG configurations.
    Each query is a single-step episode:
      - build state features
      - sample action (RAG config)
      - run RAGInterface with that config
      - score via HF reward model
      - store transition
      - periodically update PPO
    """
    def __init__(
        self,
        base_config: "RAGConfig",
        ppo_cfg: PPOConfig = PPOConfig(),
        action_space: List[RagAction] = ACTION_SPACE,
        alpha_quality: float = 1.0,
        beta_latency: float = 0.0,
        lambda_tokens: float = 0.0,
    ):
        self.base_config = base_config
        self.action_space = action_space
        self.device = torch.device(ppo_cfg.device)

        # Embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=base_config.embedding_model_name
        )

        # Models
        probe = self.embedding_model.embed_query("probe text")
        obs_dim = len(probe)
        self.ac = ActorCritic(obs_dim=obs_dim, n_actions=len(action_space)).to(self.device)
        self.optim = optim.Adam(self.ac.parameters(), lr=ppo_cfg.lr)
        self.ppo_cfg = ppo_cfg

        # Reward
        self.scorer = HFRewardScorer()
        self.lambda_tokens = lambda_tokens

        # Buffer
        self.buf = ReplayBuffer()
        self.sample_count = 0

        # Logging
        self.history: List[Dict[str, Any]] = []

    def _apply_action_to_config(self, action: RagAction) -> "RAGConfig":
        cfg = self.base_config
        # Create a shallow copy; replace fields with selected action
        cfg = type(cfg)(**{**cfg.__dict__})
        cfg.chunk_size       = action.chunk_size
        cfg.chunk_overlap    = action.chunk_overlap
        cfg.top_k            = action.top_k
        cfg.llm_temperature  = action.llm_temperature
        return cfg

    @torch.no_grad()
    def _select_action(self, obs_np: np.ndarray) -> Tuple[int, float, float]:
        obs = torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, logp_t, value_t = self.ac.act(obs)
        return int(action_t.item()), float(logp_t.item()), float(value_t.item())

    def _ppo_update(self):
        # Prepare tensors
        obs = torch.tensor(np.stack(self.buf.obs), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buf.actions, dtype=torch.long, device=self.device)
        old_logps = torch.tensor(self.buf.logps, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(self.buf.rewards, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, values = self.ac.forward(obs)
            # Stateless: advantage = reward - value
            adv = rewards - torch.tensor(self.buf.values, dtype=torch.float32, device=self.device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            targets = rewards  # no bootstrapping in single-step setting

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

                # Clipped surrogate
                clip_eps = self.ppo_cfg.clip_eps
                pg1 = ratio * b_adv
                pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
                policy_loss = -torch.min(pg1, pg2).mean()

                # Value loss
                value_loss = (new_value - b_targets).pow(2).mean()

                # Entropy bonus
                logits, _ = self.ac.forward(b_obs)
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()

                loss = policy_loss + self.ppo_cfg.value_coef * value_loss - self.ppo_cfg.entropy_coef * entropy
                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), 1.0)
                self.optim.step()

        self.buf.clear()

    # ---- Public API ----
    def answer(self, question: str) -> Dict[str, Any]:
        """
        Main call: returns the RAG answer and logs RL metadata.
        Also performs PPO updates when enough samples have accumulated.
        """
        # 1) Build state features
        obs_np = np.array(self.embedding_model.embed_query(question), dtype=np.float32)

        # 2) Select action (RAG config)
        a_idx, logp, vpred = self._select_action(obs_np)
        action = self.action_space[a_idx]

        # 3) Run RAG with selected config
        cfg = self._apply_action_to_config(action)
        t0 = time.time()
        rag = RAGInterface(cfg)
        answer = rag.ask(question)
        latency = time.time() - t0

        # Optional token cost proxy
        token_cost = 0.0  # integrate if you track tokens

        # 4) Compute reward
        reward = self.scorer(question, answer)

        # 5) Store transition and maybe update PPO
        self.buf.add(obs_np, a_idx, reward, logp, vpred)
        self.sample_count += 1
        if self.sample_count % self.ppo_cfg.max_buffer_size == 0:
            self._ppo_update()

        # Log
        record = {
            "question": question,
            "action_idx": a_idx,
            "action": asdict(action),
            "latency": latency,

            "reward": reward,
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
            json.dump({"history_len": len(self.history)}, f)

    def load(self, path: str):
        self.ac.load_state_dict(torch.load(path + ".pt", map_location=self.device))
