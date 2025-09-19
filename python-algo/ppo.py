import torch
import torch.nn as nn
import torch.optim as optim
from nn_creator import TerminalA2C
import numpy as np
from torch.distributions import Categorical
import time
import random

from gamelib.util import *
from run_match import run_match


# Hyperparameters
gamma = 0.99  # Discount factor
lambda_gae = 0.95  # GAE lambda
eps_clip = 0.2  # PPO clipping parameter
lr = 3e-4  # Learning rate
ppo_epochs = 10  # Number of updates per episode
mini_batch_size = 64  # Size of mini-batch for updating
max_episodes = 1000  # Total number of episodes
num_games = 15

new_model = True


# Model and optimizer
model = TerminalA2C()
if not new_model:
    try:
        model.load_state_dict(torch.load("checkpoints/latest.pth"))
    except Exception as e:
        print(e)
        print("Using new models.")
optimizer = optim.Adam(model.parameters(), lr=lr)


# Helper to compute GAE across full buffer
def compute_gae(rewards, values, dones):
    values = values + [0]  # add next value at the end
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lambda_gae * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])
    advantages = [ret - val for ret, val in zip(returns, values[:-1])]
    return returns, advantages

# Buffer to accumulate episode data
buffer = []

# Training loop
for episode in range(max_episodes):
    
    for game in range(num_games):
        start = time.time()
        with open("thegame.txt", "w") as file:
            file.write(str(game))
        run_match("python-algo/ppo_strategy.ps1", "python-algo/starter_strategy.ps1")

        ep_obs = []
        ep_actions = []
        ep_log_probs = []
        ep_rewards = []
        ep_dones = []
        ep_values = []

        with open(f"buffer/{game}.py", "r") as file:
            data = [eval(_.strip()) for _ in file.readlines()]

        for obs, action, log_prob, value in data:
            ep_obs.append(obs)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_values.append(value)
        
        with open(f"buffer/{game}_rewards.txt", "r") as file:
            data = [float(_.strip()) for _ in file.readlines()]
        
        for reward in data:
            # Normalise reward with victory reward
            ep_rewards.append(reward / VICTORY_REWARD)
            if (reward == -250.0) or (reward == 250.0):
                ep_dones.append(True)
            else:
                ep_dones.append(False)

        # Store one full episode in the buffer
        buffer.append({
            'obs': ep_obs,
            'actions': ep_actions,
            'log_probs': ep_log_probs,
            'rewards': ep_rewards,
            'dones': ep_dones,
            'values': ep_values
        })

        print(f"Episode {episode}, Game {game} finished in time {time.time() - start} seconds. Reward: {sum(ep_rewards)}")
    
    # Time to update PPO

    # Flatten the buffer
    all_ms_obs = []
    all_msc_obs = []
    all_ts_obs = []
    all_tsc_obs = []
    all_board_obs = []
    all_building_actions = []
    all_unit_actions = []
    all_building_log_probs = []
    all_unit_log_probs = []
    all_returns = []
    all_advantages = []

    for ep in buffer:
        # TODO: need extra loop and sum for each game!
        returns, advantages = compute_gae(ep['rewards'], ep['values'], ep['dones'])
        for frame in range(len(ep["obs"])):
            all_ms_obs.append(torch.tensor(ep['obs'][frame][0]))
            all_msc_obs.append(torch.tensor(ep['obs'][frame][1]))
            all_ts_obs.append(torch.tensor(ep['obs'][frame][2]))
            all_tsc_obs.append(torch.tensor(ep['obs'][frame][3]))
            all_board_obs.append(torch.tensor(ep['obs'][frame][4]))
            all_building_actions.append(torch.tensor(ep['actions'][frame][0]))
            all_unit_actions.append(torch.tensor(ep['actions'][frame][1]))
            all_building_log_probs.append(torch.tensor(ep['log_probs'][frame][0]))
            all_unit_log_probs.append(torch.tensor(ep['log_probs'][frame][1]))
        all_returns += returns
        all_advantages += advantages

    # Convert to tensors
    all_ms_obs = torch.stack(all_ms_obs)
    all_msc_obs = torch.stack(all_msc_obs)
    all_ts_obs = torch.stack(all_ts_obs)
    all_tsc_obs = torch.stack(all_tsc_obs)
    all_board_obs = torch.stack(all_board_obs)
    all_building_actions = torch.stack(all_building_actions)
    all_unit_actions = torch.stack(all_unit_actions)
    all_building_log_probs = torch.stack(all_building_log_probs).detach()
    all_unit_log_probs = torch.stack(all_unit_log_probs).detach()
    all_returns = torch.tensor(all_returns, dtype=torch.float32)
    all_advantages = torch.tensor(all_advantages, dtype=torch.float32)
    all_advantages = (all_advantages - all_advantages.mean()) / (all_advantages.std() + 1e-8)

    mean_return = all_returns.mean()

    # PPO update
    dataset_size = len(all_returns)
    for _ in range(ppo_epochs):
        time_start = time.time()
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        thelosses = []
        for start in range(0, dataset_size, mini_batch_size):
            # Get minibatch data
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            real_batch_size = len(mb_idx)
            ms_obs_batch = all_ms_obs[mb_idx]
            msc_obs_batch = all_msc_obs[mb_idx]
            ts_obs_batch = all_ts_obs[mb_idx]
            tsc_obs_batch = all_tsc_obs[mb_idx]
            board_obs_batch = all_board_obs[mb_idx]
            building_action_batch = all_building_actions[mb_idx]
            unit_action_batch = all_unit_actions[mb_idx]
            old_building_log_prob_batch = all_building_log_probs[mb_idx].view((real_batch_size, 3, 392))
            old_unit_log_prob_batch = all_unit_log_probs[mb_idx].view((real_batch_size, 28, 3))
            return_batch = all_returns[mb_idx]
            adv_batch = all_advantages[mb_idx]

            # Forward pass through model and calculate difference to old policy
            building_actions_dists, unit_actions_dists, values = model.forward(ms_obs_batch, msc_obs_batch, ts_obs_batch, tsc_obs_batch, board_obs_batch)
            building_dist = Categorical(building_actions_dists)
            new_building_log_probs = building_dist.log_prob(building_action_batch)
            building_entropy = building_dist.entropy().mean()
            building_ratio = torch.exp(new_building_log_probs - old_building_log_prob_batch).mean(dim=(1,2)) # Average ratio (change between old and new policy)
            unit_dist = Categorical(unit_actions_dists)
            new_unit_log_probs = unit_dist.log_prob(unit_action_batch)
            unit_entropy = unit_dist.entropy().mean()
            unit_ratio = torch.exp(new_unit_log_probs - old_unit_log_prob_batch).mean(dim=(1,2)) # Average ratio

            surr1 = building_ratio * adv_batch
            surr2 = torch.clamp(building_ratio, 1 - eps_clip, 1 + eps_clip) * adv_batch
            surr3 = unit_ratio * adv_batch
            surr4 = torch.clamp(unit_ratio, 1 - eps_clip, 1 + eps_clip) * adv_batch
            building_policy_loss = -torch.min(surr1, surr2).mean()
            unit_policy_loss = -torch.min(surr3, surr4).mean()
            value_loss = (return_batch - values.squeeze()).pow(2).mean()

            factor = -min(0, mean_return) / 1000
            
            building_l1_reg = torch.sum(torch.abs(new_building_log_probs))
            unit_l1_reg = torch.sum(torch.abs(new_unit_log_probs))
            print(building_policy_loss, unit_policy_loss, value_loss, building_l1_reg, unit_l1_reg, factor)

            entropy = 0.5 * (building_entropy + unit_entropy)
            loss = torch.abs(building_policy_loss) + torch.abs(unit_policy_loss) + 0.4 * value_loss - 0.01 * entropy + factor * (building_l1_reg + unit_l1_reg)
            thelosses.append(loss.detach().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Mini-Epoch {_}, Loss = {np.mean(thelosses)}, Time {time.time() - time_start} seconds")

    # Experience replay
    random.shuffle(buffer)
    length = len(buffer)
    buffer = buffer[ : length // 2]


    torch.save(model.state_dict(), f"checkpoints/{episode}.pth")
    torch.save(model.state_dict(), f"checkpoints/latest.pth")
