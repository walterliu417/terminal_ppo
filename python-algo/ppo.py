import torch
import torch.nn as nn
import torch.optim as optim
from nn_creator import *
import numpy as np
from torch.distributions import Categorical
import time
import random

from gamelib.util import *
from run_match import run_match


# Hyperparameters
gamma = 0.97  # Discount factor
lambda_gae = 0.93  # GAE lambda
eps_clip = 0.3  # PPO clipping parameter
lr = 4e-4  # Learning rate
ppo_epochs = 7  # Number of updates per episode
mini_batch_size = 64  # Size of mini-batch for updating
max_episodes = 10000  # Total number of episodes
num_games = 20
entropy_bonus = 0.02

new_model = True


# Model and optimizer
unit_model = UnitAgent()
building_model = BuildingAgent()
if not new_model:
    try:
        unit_model.load_state_dict(torch.load("checkpoints/unit_latest.pth", weights_only=True))
        building_model.load_state_dict(torch.load("checkpoints/building_latest.pth", weights_only=True))
    except Exception as e:
        print(e)
        print("Using new models.")
unit_optimizer = optim.Adam(unit_model.parameters(), lr=lr)
building_optimizer = optim.Adam(building_model.parameters(), lr=lr)
unit_model.train()
building_model.train()


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
        is_windows = sys.platform.startswith('win')
        if is_windows:
            run_match("python-algo/ppo_strategy.ps1", "python-algo/starter_strategy.ps1")
        else:
            run_match("python-algo/ppo_strategy.sh", "python-algo/starter_strategy.sh")

        unit_ep_obs = []
        unit_ep_actions = []
        unit_ep_log_probs = []
        unit_ep_rewards = []
        unit_ep_dones = []
        unit_ep_values = []
        unit_ep_penalties = []
        building_ep_obs = []
        building_ep_actions = []
        building_ep_log_probs = []
        building_ep_rewards = []
        building_ep_dones = []
        building_ep_values = []
        building_ep_penalties = []

        with open(f"buffer/{game}.py", "r") as file:
            data = [eval(_.strip()) for _ in file.readlines()]

        for obs, action, log_prob, value in data:
            unit_ep_obs.append([obs[0], obs[1], obs[3]])
            unit_ep_actions.append(action[1])
            unit_ep_log_probs.append(log_prob[1])
            unit_ep_values.append(value[1])
            building_ep_obs.append([obs[0], obs[2], obs[3]])
            building_ep_actions.append(action[0])
            building_ep_log_probs.append(log_prob[0])
            building_ep_values.append(value[0])
        
        with open(f"buffer/{game}_rewards.txt", "r") as file:
            data = [_.strip().split(",") for _ in file.readlines()]
        
        for building_reward, unit_reward in data:
            # Normalise reward with victory reward
            unit_ep_rewards.append(float(unit_reward) / VICTORY_REWARD)
            building_ep_rewards.append(float(building_reward) / VICTORY_REWARD)
            if (unit_reward == -250.0) or (unit_reward == 250.0):
                unit_ep_dones.append(True)
                building_ep_dones.append(True)
            else:
                unit_ep_dones.append(False)
                building_ep_dones.append(False)

        with open(f"buffer/{game}_penalties.txt", "r") as file:
            data = [_.strip().split(",") for _ in file.readlines()]
        
        for building_penalty, unit_penalty in data:
            # Normalise reward with victory reward
            unit_ep_penalties.append(float(unit_penalty) / VICTORY_REWARD)
            building_ep_penalties.append(float(building_penalty) / VICTORY_REWARD)

        # Store one full episode in the buffer
        buffer.append({
            'unit_obs': unit_ep_obs,
            'unit_actions': unit_ep_actions,
            'unit_log_probs': unit_ep_log_probs,
            'unit_rewards': unit_ep_rewards,
            'unit_dones': unit_ep_dones,
            'unit_values': unit_ep_values,
            'unit_penalties' : unit_ep_penalties,
            'building_obs': building_ep_obs,
            'building_actions': building_ep_actions,
            'building_log_probs': building_ep_log_probs,
            'building_rewards': building_ep_rewards,
            'building_dones': building_ep_dones,
            'building_values': building_ep_values,
            'building_penalties': building_ep_penalties
        })

        print(f"Episode {episode}, Game {game} finished in time {time.time() - start} seconds. Building reward {sum(building_ep_rewards)}, Unit rewrd {sum(unit_ep_rewards)})")
    
    # Time to update PPO

    # Flatten the buffer
    all_unit_obs = []
    all_all_building_obs = []
    all_my_building_obs = []
    all_stats_obs = []
    all_building_actions = []
    all_unit_actions = []
    all_building_log_probs = []
    all_unit_log_probs = []
    all_unit_returns = []
    all_unit_advantages = []
    all_unit_penalties = []
    all_building_returns = []
    all_building_advantages = []
    all_building_penalties = []

    for ep in buffer:
        unit_returns, unit_advantages = compute_gae(ep['unit_rewards'], ep['unit_values'], ep['unit_dones'])
        building_returns, building_advantages = compute_gae(ep['building_rewards'], ep['building_values'], ep['building_dones'])
        for frame in range(len(ep["unit_obs"])):
            all_unit_obs.append(torch.tensor(ep['unit_obs'][frame][0]))
            all_all_building_obs.append(torch.tensor(ep['unit_obs'][frame][1]))
            all_my_building_obs.append(torch.tensor(ep['building_obs'][frame][1]))
            all_stats_obs.append(torch.tensor(ep['building_obs'][frame][2]))
            all_building_actions.append(torch.tensor(ep['building_actions'][frame]))
            all_unit_actions.append(torch.tensor(ep['unit_actions'][frame]))
            all_building_log_probs.append(torch.tensor(ep['building_log_probs'][frame]))
            all_unit_log_probs.append(torch.tensor(ep['unit_log_probs'][frame]))
            all_unit_penalties += ep['unit_penalties'][frame]
            all_building_penalties += ep['building_penalties'][frame]
        all_unit_returns += unit_returns
        all_unit_advantages += unit_advantages
        all_building_returns += building_returns
        all_building_advantages += building_advantages

    # Convert to tensors
    all_unit_obs = torch.cat(all_unit_obs, dim=0)
    all_all_building_obs = torch.cat(all_all_building_obs, dim=0)
    all_my_building_obs = torch.cat(all_my_building_obs, dim=0)
    all_stats_obs = torch.cat(all_stats_obs, dim=0)
    all_building_actions = torch.stack(all_building_actions)
    all_unit_actions = torch.stack(all_unit_actions)
    all_building_log_probs = torch.cat(all_building_log_probs, dim=0).detach()
    all_unit_log_probs = torch.cat(all_unit_log_probs, dim=0).detach()
    all_unit_returns = torch.tensor(all_unit_returns, dtype=torch.float32)
    all_unit_advantages = torch.tensor(all_unit_advantages, dtype=torch.float32)
    all_unit_penalties = torch.stack(all_unit_penalties)
    all_building_penalties = torch.stack(all_building_penalties)
    all_building_returns = torch.tensor(all_building_returns, dtype=torch.float32)
    all_building_advantages = torch.tensor(all_building_advantages, dtype=torch.float32)

    # Further advantage normalisation
    all_unit_advantages = (all_unit_advantages - all_unit_advantages.mean()) / (all_unit_advantages.std() + 1e-8)
    all_building_advantages = (all_building_advantages - all_building_advantages.mean()) / (all_building_advantages.std() + 1e-8)


    # PPO update
    dataset_size = len(all_unit_returns)
    for _ in range(ppo_epochs):
        time_start = time.time()
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        unit_thelosses = []
        building_thelosses = []
        for start in range(0, dataset_size, mini_batch_size):
            # Get minibatch data
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            real_batch_size = len(mb_idx)
            if real_batch_size <= mini_batch_size:
                # rather not train.
                continue
            unit_obs_batch = all_unit_obs[mb_idx]
            all_building_obs_batch = all_all_building_obs[mb_idx]
            my_building_obs_batch = all_my_building_obs[mb_idx]
            stats_obs_batch = all_stats_obs[mb_idx]
            building_action_batch = all_building_actions[mb_idx]
            unit_action_batch = all_unit_actions[mb_idx]
            old_building_log_prob_batch = all_building_log_probs[mb_idx].view((real_batch_size, 210, 6))
            old_unit_log_prob_batch = all_unit_log_probs[mb_idx].view((real_batch_size, 28, 3))
            unit_return_batch = all_unit_returns[mb_idx]
            unit_adv_batch = all_unit_advantages[mb_idx]
            unit_penalty_batch = all_unit_penalties[mb_idx]
            building_return_batch = all_building_returns[mb_idx]
            building_adv_batch = all_building_advantages[mb_idx]
            building_penalty_batch = all_building_penalties[mb_idx]

            # Forward pass through models and calculate difference to old policy

            # Unit model.
            unit_actions_dists, unit_values = unit_model.forward(unit_obs_batch, all_building_obs_batch, stats_obs_batch)
            unit_dist = Categorical(unit_actions_dists)
            new_unit_log_probs = unit_dist.log_prob(unit_action_batch)
            unit_entropy = unit_dist.entropy().mean()
            unit_ratio = torch.exp(new_unit_log_probs - old_unit_log_prob_batch).mean(dim=(1,2)) # Average ratio

            surr1 = unit_ratio * unit_adv_batch
            surr2 = torch.clamp(unit_ratio, 1 - eps_clip, 1 + eps_clip) * unit_adv_batch
            unit_policy_loss = -torch.min(surr1, surr2).mean()
            unit_value_loss = (unit_return_batch - unit_values.squeeze()).pow(2).mean()

            unit_factor = max(0, torch.mean(unit_penalty_batch)) / (real_batch_size * 100)
            #unit_factor = 0 # Don't force it

            unit_l1_reg = torch.sum(torch.abs(new_unit_log_probs))
            print(unit_policy_loss, unit_value_loss, unit_l1_reg, unit_factor)

            unit_loss = torch.abs(unit_policy_loss) + 0.4 * unit_value_loss - entropy_bonus * unit_entropy + unit_factor * unit_l1_reg
            unit_thelosses.append(unit_loss.detach().item())
            unit_optimizer.zero_grad()
            unit_loss.backward()
            unit_optimizer.step()

            # Building model.
            building_actions_dists, building_values = building_model.forward(unit_obs_batch, my_building_obs_batch, stats_obs_batch)
            building_dist = Categorical(building_actions_dists)
            new_building_log_probs = building_dist.log_prob(building_action_batch)
            building_entropy = building_dist.entropy().mean()
            building_ratio = torch.exp(new_building_log_probs - old_building_log_prob_batch).mean(dim=(1,2)) # Average ratio (change between old and new policy)

            surr1 = building_ratio * building_adv_batch
            surr2 = torch.clamp(building_ratio, 1 - eps_clip, 1 + eps_clip) * building_adv_batch
            building_policy_loss = -torch.min(surr1, surr2).mean()
            building_value_loss = (building_return_batch - building_values.squeeze()).pow(2).mean()

            building_factor = max(0, torch.mean(building_penalty_batch)) / (real_batch_size * 100)
            building_factor = 0 # Don't force it

            building_l1_reg = torch.sum(torch.abs(new_building_log_probs))
            print(building_policy_loss, building_value_loss, building_l1_reg, building_factor)

            building_loss = torch.abs(building_policy_loss) + 0.4 * building_value_loss - entropy_bonus * building_entropy + building_factor * building_l1_reg
            building_thelosses.append(building_loss.detach().item())
            building_optimizer.zero_grad()
            building_loss.backward()
            building_optimizer.step()

        print(f"Mini-Epoch {_}, Building loss = {np.mean(building_thelosses)}, Unit loss = {np.mean(unit_thelosses)}, Time {time.time() - time_start} seconds")

    # Experience replay
    random.shuffle(buffer)
    length = len(buffer)
    buffer = buffer[ : length // 50]

    # Save models.
    torch.save(unit_model.state_dict(), f"checkpoints/unit_{episode}.pth")
    torch.save(unit_model.state_dict(), f"checkpoints/unit_latest.pth")
    torch.save(building_model.state_dict(), f"checkpoints/building_{episode}.pth")
    torch.save(building_model.state_dict(), f"checkpoints/building_latest.pth")
