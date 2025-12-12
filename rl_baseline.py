# rl_rag.py

"""
This file is responsible for training a Reinforcement Learning (RL) pipeline to optimize
the parameters of a Retrieval-Augmented Generation (RAG) system using the SQuAD v2 dataset.
In particular, it optimizes the 'top_k' retrieval parameter and the LLM temperature setting.
It compares the performance of the RL-optimized RAG against a static baseline RAG system.
"""

from rag import RAGConfig
from rl import RLPipeline, DatasetRewardScorer
from rag_interface import RAGInterface
from datasets import load_dataset
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from utils import SHORT_ANSWER_PROMPT

# SQUAD v2 Dataset Loading

print("Loading SQuAD v2 dataset...")
ds = load_dataset("squad_v2", split="train")

articles = defaultdict(list)
for ex in ds:
    if len(ex["answers"]["text"]) == 0:
        continue
    articles[ex["title"]].append(ex)

print(f"Loaded {len(articles)} distinct Wikipedia articles.")

# Setup

EPISODES = 150 
INITIAL_TOP_K = 2 

base_config = RAGConfig(
    texts=[], 
    chunk_size=350,
    chunk_overlap=50,
    top_k=INITIAL_TOP_K, 
    llm_model_name="llama-3.1-8b-instant", 
    llm_provider="groq",
    llm_temperature=0.2,
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    custom_prompt_template=SHORT_ANSWER_PROMPT,
)

reward_scorer = DatasetRewardScorer(
    similarity_threshold=0.5
)

# RL Pipeline for training
rl_pipeline = RLPipeline(
    base_config=base_config,
    reward_scorer=reward_scorer
)

# Static RAG Baseline 

rag_baseline_config = RAGConfig(**base_config.__dict__)
rag_baseline = RAGInterface(rag_baseline_config)

# Data loading for a random article

all_titles = list(articles.keys())
title = random.choice(all_titles)
article_items = articles[title]
print(f"\nEvaluation on Article: **{title}** with {len(article_items)} Q/A pairs.")

long_context = "\n\n".join(item["context"] for item in article_items)
rl_pipeline.base_config.texts = [long_context]
rag_baseline.config.texts = [long_context]


# Trainning Loop

rl_rewards = []
baseline_rewards = []

print("\n" + "="*50)
print(f"(Trainning with {EPISODES} Episodes)")
print("RL Policy is Training | Baseline is Static (k=2)")
print("="*50)

for i in range(EPISODES):
    # 1. Pick a random Q/A example
    ex = random.choice(article_items)
    question = ex["question"]
    gold_answer = ex["answers"]["text"][0]

    # 2. Run RL Pipeline (Training mode: policy updates occur here)
    rl_result = rl_pipeline.answer(question, gold_answer=gold_answer)
    rl_rewards.append(rl_result["reward"])

    # 3. Run Static Baseline Pipeline (No training, fixed k=2)
    baseline_answer = rag_baseline.ask(question)
    baseline_reward = reward_scorer(baseline_answer, gold_answer)
    baseline_rewards.append(baseline_reward)

    print(f"Baseline Reward: {baseline_reward:.4f} | RL Reward: {rl_result['reward']:.4f} | Episode: {i+1}/{EPISODES}")
print("\nLoop Finished.")
print("Final optimized action space policy:", rl_pipeline.action_space)

# Training RL vs Baseline Reward Comparison Plotting

window = 10 

# Average every 'window' episodes
rl_avg_rewards = [
    sum(rl_rewards[i:i+window]) / len(rl_rewards[i:i+window])
    for i in range(0, len(rl_rewards), window)
]
baseline_avg_rewards = [
    sum(baseline_rewards[i:i+window]) / len(baseline_rewards[i:i+window])
    for i in range(0, len(baseline_rewards), window)
]

episodes_axis = [i + window for i in range(0, len(rl_rewards), window)]

final_rl_avg = rl_avg_rewards[-1] if rl_avg_rewards else 0
final_baseline_avg = baseline_avg_rewards[-1] if baseline_avg_rewards else 0

print("\n" + "="*50)
print(f"FINAL AVERAGE REWARDS (Last {window} Episodes):")
print(f"RL Optimized RAG Policy: {final_rl_avg:.4f}")
print(f"Static RAG Baseline (k={INITIAL_TOP_K}): {final_baseline_avg:.4f}")
print("="*50)


plt.figure(figsize=(12, 7))
plt.plot(episodes_axis, rl_avg_rewards, marker="o", color="red", label="RL Optimized RAG (Policy Training)")
plt.plot(episodes_axis, baseline_avg_rewards, marker="s", color="blue", linestyle='--', label=f"Static RAG Baseline (k={INITIAL_TOP_K})")
plt.title(f"Reward Comparison: RL Policy Training vs. Static Baseline\nArticle: {title}")
plt.xlabel("Episode Number")
plt.ylabel(f"Average Reward (Moving Window of {window} Episodes)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig("rl_rag_unified_comparison.png")
plt.show()

# Final Evaluation of Optimized RAG vs Baseline

# Take optimized top k and temperature from RL policy

optimal_top_k = rl_pipeline.action_space[-1].top_k
optimal_llm_temp = rl_pipeline.action_space[-1].llm_temperature
optimal_rag_config = RAGConfig(
    texts=[long_context],
    chunk_size=350,
    chunk_overlap=50,
    top_k=optimal_top_k,
    llm_model_name="llama-3.1-8b-instant",
    llm_provider="groq",
    llm_temperature=optimal_llm_temp,
    embedding_model_name="sentence-transformers/all-mpnet-base-v2",
    custom_prompt_template=SHORT_ANSWER_PROMPT,
)
rag_optimized = RAGInterface(optimal_rag_config)

baseline_rewards = []
optimized_rewards = []

EPISODES_EVAL = 20

for i in range(EPISODES_EVAL):
    ex = random.choice(article_items)
    question = ex["question"]
    gold_answer = ex["answers"]["text"][0]

    optimized_answer = rag_optimized.ask(question)
    baseline_answer = rag_baseline.ask(question)

    optimized_reward = reward_scorer(optimized_answer, gold_answer)
    baseline_reward = reward_scorer(baseline_answer, gold_answer)

    optimized_rewards.append(optimized_reward)
    baseline_rewards.append(baseline_reward)

    print("\n==========================")
    print(f"EPISODE {i} — Article:", title)
    print("==========================")
    print(f"Q: {question}")
    print(f"Optimized A: {optimized_answer} | Reward: {optimized_reward}")
    print(f"Baseline A: {baseline_answer} | Reward: {baseline_reward}")

rl_avg_rewards = sum(optimized_rewards) / len(optimized_rewards)
baseline_avg_rewards = sum(baseline_rewards) / len(baseline_rewards)

print("\n" + "="*50)
print("FINAL AVERAGE REWARDS OVER 20 EPISODES:")
print(f"Optimized RAG Policy Reward: {rl_avg_rewards:.4f}")
print(f"Static RAG Baseline Reward: {baseline_avg_rewards:.4f}")
print("="*50)
plt.bar(['Static Baseline', 'RL Optimized'], [baseline_avg_rewards, rl_avg_rewards], color=['blue', 'red'])
plt.ylabel('Average Reward')
plt.title('RAG Baseline vs RL Optimized Policy Performance')
plt.savefig('rag_optimized_vs_baseline.png')
plt.show()