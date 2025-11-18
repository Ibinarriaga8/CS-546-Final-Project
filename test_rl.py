# test_rl_rag.py

from rag import RAGConfig
from rl import RLPipeline
import random

# 1. Deep Learning RAG configuration
config = RAGConfig(
    urls=[
        "https://en.wikipedia.org/wiki/Deep_learning",
        "https://en.wikipedia.org/wiki/Artificial_neural_network",
        "https://en.wikipedia.org/wiki/Backpropagation",
        "https://en.wikipedia.org/wiki/Gradient_descent",
        "https://cs230.stanford.edu/notes/"
    ],
    chunk_size=1000,
    chunk_overlap=200,
    top_k=4,
    llm_model_name="llama-3.1-8b-instant",
    llm_provider="groq",
    llm_temperature=0.2,
    embedding_model_name="sentence-transformers/all-mpnet-base-v2"
)

# 2. RL pipeline
rl = RLPipeline(base_config=config)

# 3. Deep learning question set (can expand)
questions = [
    "What is a neural network?",
    "Explain backpropagation in simple terms.",
    "What is gradient descent?",
    "What is the difference between overfitting and underfitting?",
    "What is a loss function in deep learning?",
    "What are activation functions and why are they needed?",
    "What is the vanishing gradient problem?",
    "How does regularization work?",
    "What is a convolutional neural network?",
    "Explain the idea of training, validation, and testing sets.",
    "What is the structure of a biological neuron and how does it inspire artificial neural networks?",
    "What are the main components of an artificial neural network?",
    "Explain the concept of weights and biases in neural networks.",
    "What is a feedforward neural network?",
    "What is supervised learning?",
    "What is unsupervised learning?",
    "Explain the chain rule as used in backpropagation.",
    "Why is differentiability important in neural networks?",
    "What is the cost surface in gradient descent?",
    "What are the limitations of standard gradient descent?",
    "What is stochastic gradient descent and why is it useful?",
    "What is a learning rate?",
    "Explain the concept of epochs and iterations.",
    "What is the purpose of splitting data into batches?",
    "What is a multilayer perceptron (MLP)?",
    "What is the universal approximation theorem?",
    "What are convolutional layers used for?",
    "What is feature extraction in the context of CNNs?",
    "What are pooling layers and what problem do they solve?",
    "What is overparameterization?",
    "What is L2 regularization?",
    "What is L1 regularization?",
    "What is ridge regression?",
    "What is early stopping?",
    "What are hyperparameters?",
    "What is gradient vanishing and in what situations does it occur?",
    "What is gradient exploding?",
    "What is weight initialization and why does it matter?",
    "What is the purpose of activation functions like ReLU or sigmoid?",
    "What is the softmax function and where is it used?",
    "What is cross-entropy loss?",
    "What is mean squared error loss?",
    "What is backpropagation through layers?",
    "What is a computational graph?",
    "What is model generalization?",
    "What is training error vs. test error?",
    "What is the bias of a model?",
    "What is the variance of a model?",
    "Explain the bias-variance tradeoff.",
    "What is a gradient update rule?",
    "What is the Hessian in the context of optimization?",
    "Explain the idea of local minima and saddle points in optimization.",
    "What are the main challenges in training deep neural networks?",
    "What is data augmentation?",
    "What is feature scaling and why is it important?",
    "Why do deep models require large datasets?",
    "What is the role of GPUs in deep learning?",
    
]

# 4. Run multiple RL episodes
for i in range(50):
    q = random.choice(questions)
    result = rl.answer(q)
    print(f"\n--- Episode {i} ---")
    print("Q:", q)
    print("Action used:", result["action"])
    print("Reward:", result["reward"])
    print("Latency:", result["latency"])

# 5. Save policy
rl.save("ppo_rag_policy_dl")
print("\nTraining Finished.")
