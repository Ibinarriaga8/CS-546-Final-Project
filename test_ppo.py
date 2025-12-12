# test_ppo.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 

from rl import ActorCritic, PPOConfig, ReplayBuffer

# Environment: learn f(x) = sin(x) via RL


class FunctionEnv:
    """
    Single-step environment:
      - state: x ~ Uniform[-pi, pi]
      - reward: negative squared error between sin(x) and selected y_action
    """
    def __init__(self, x_low=-np.pi, x_high=np.pi):
        self.x_low = x_low
        self.x_high = x_high

    def sample_state(self) -> float:
        x = np.random.uniform(self.x_low, self.x_high)
        return float(x)

    def f(self, x: float) -> float:
        return float(np.sin(x))


def ppo_update(ac, optim, buf: ReplayBuffer, cfg: PPOConfig, device: torch.device):
    # Prepare data tensors
    obs = torch.tensor(np.stack(buf.obs), dtype=torch.float32, device=device)
    actions = torch.tensor(buf.actions, dtype=torch.long, device=device)
    old_logps = torch.tensor(buf.logps, dtype=torch.float32, device=device)
    rewards = torch.tensor(buf.rewards, dtype=torch.float32, device=device)
    old_values = torch.tensor(buf.values, dtype=torch.float32, device=device)
    
    # Advantage estimation and targets
    targets = rewards
    adv = rewards - old_values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(cfg.update_epochs):
        idx = torch.randperm(len(obs), device=device)
        for start in range(0, len(obs), cfg.batch_size):
            batch = idx[start:start + cfg.batch_size]
            b_obs = obs[batch]
            b_act = actions[batch]
            b_old_logp = old_logps[batch]
            b_adv = adv[batch]
            b_targets = targets[batch]

            new_logp, new_value = ac.logprob_value(b_obs, b_act)
            ratio = torch.exp(new_logp - b_old_logp)

            clip_eps = cfg.clip_eps
            pg1 = ratio * b_adv
            pg2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * b_adv
            policy_loss = -torch.min(pg1, pg2).mean()

            value_loss = (new_value - b_targets).pow(2).mean()

            logits, _ = ac.forward(b_obs)
            dist = torch.distributions.Categorical(logits=logits)
            entropy = dist.entropy().mean()

            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
            optim.step()

    buf.clear()


def test_ppo_nonlinear():

    env = FunctionEnv()
    obs_dim = 1           
    n_actions = 21        

    # Discrete y grid in [-1, 1]
    action_values = np.linspace(-1.0, 1.0, n_actions).astype(np.float32)

    cfg = PPOConfig()
    cfg.device = "cpu"
    cfg.gamma = 0.0
    cfg.gae_lambda = 0.0
    cfg.max_buffer_size = 256
    cfg.batch_size = 64
    cfg.update_epochs = 4
    
    total_steps = 20000 
    print_interval = 1000

    device = torch.device(cfg.device)
    ac = ActorCritic(obs_dim=obs_dim, n_actions=n_actions).to(device)
    optim = torch.optim.Adam(ac.parameters(), lr=cfg.lr)
    buf = ReplayBuffer()

    #Data for plotting
    mse_history = []
    step_history = []

    for step in range(1, total_steps + 1):
        # Sample a state x
        x = env.sample_state()
        obs_np = np.array([x], dtype=np.float32)

        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action_t, logp_t, value_t = ac.act(obs_t)
        a = int(action_t.item())

        # Compute reward
        y_true = env.f(x)
        y_pred = float(action_values[a])
        mse = (y_true - y_pred) ** 2
        reward = -mse

        buf.add(obs_np, a, reward, float(logp_t.item()), float(value_t.item()))

        if step % cfg.max_buffer_size == 0:
            ppo_update(ac, optim, buf, cfg, device)

        # Periodic evaluation: check approximation error over a grid
        if step % print_interval == 0:
            xs = np.linspace(-np.pi, np.pi, 50, dtype=np.float32)
            with torch.no_grad():
                obs_eval = torch.tensor(xs[:, None], dtype=torch.float32, device=device)
                logits, _ = ac.forward(obs_eval)
                dist = torch.distributions.Categorical(logits=logits)
                probs = dist.probs.cpu().numpy()

            # Expected y = sum_a p(a|x) * y_a
            y_pred_grid = (probs * action_values[None, :]).sum(axis=1)
            y_true_grid = np.sin(xs)
            mse_grid = ((y_true_grid - y_pred_grid) ** 2).mean()

            step_history.append(step)
            mse_history.append(mse_grid)

            print(f"step {step}: mean MSE over grid = {mse_grid:.4f}")


    print("\nFinal samples:")
    test_xs = np.linspace(-np.pi, np.pi, 5, dtype=np.float32)
    with torch.no_grad():
        obs_eval = torch.tensor(test_xs[:, None], dtype=torch.float32, device=device)
        logits, _ = ac.forward(obs_eval)
        dist = torch.distributions.Categorical(logits=logits)
        probs = dist.probs.cpu().numpy()
    y_pred_grid_final = (probs * action_values[None, :]).sum(axis=1)
    
    #Store final sample data for plotting 
    final_sample_data = {'x': test_xs, 'sin_x': np.sin(test_xs), 'predicted_y': y_pred_grid_final}
    # --------------------------------------------

    for x, yt, yp in zip(test_xs, np.sin(test_xs), y_pred_grid_final):
        print(f"x={x: .2f}, sin(x)={yt: .3f}, predicted≈{yp: .3f}")
        

    
    print("\nGenerating plots...")

    ##  MSE Convergence Plot
    plt.figure(figsize=(10, 5))
    plt.plot(step_history, mse_history, label='Mean MSE over Grid', color='skyblue')
    plt.title('PPO Training Convergence: Mean MSE vs. Training Step', fontsize=14)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Mean MSE over Grid', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ppo_mse_convergence.png')
    plt.show() # Display the plot
    print("Saved MSE Convergence Plot as ppo_mse_convergence.png")

    ##  Function Approximation Plot
    plt.figure(figsize=(10, 5))
    
    # Plot true sin(x) curve for reference
    x_range = np.linspace(-np.pi, np.pi, 100)
    plt.plot(x_range, np.sin(x_range), label='True $f(x) = \sin(x)$', color='blue', linestyle='--')
    
    # Plot the predicted points (Expected value)
    plt.scatter(final_sample_data['x'], final_sample_data['predicted_y'], 
                label='PPO Predicted $E[y|x]$ (5 samples)', color='red', marker='o', zorder=5)
    
    # Plot the true points corresponding to the samples
    plt.scatter(final_sample_data['x'], final_sample_data['sin_x'], 
                label='True $\sin(x)$ (5 samples)', color='green', marker='x', zorder=5)

    plt.title('PPO Function Approximation vs. True $\sin(x)$', fontsize=14)
    plt.xlabel('$x$', fontsize=12)
    plt.ylabel('$y$', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('ppo_function_approximation.png')
    plt.show() # Display the plot
    print("Saved Function Approximation Plot as ppo_function_approximation.png")

if __name__ == "__main__":
    test_ppo_nonlinear()