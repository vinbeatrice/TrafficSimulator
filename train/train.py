import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import time
import copy
import numpy as np
from env.path_env import PathEnv
from agent.agent import DQNAgent
from env.maps import GridMap
from plot.plot_learning_queue import plot_learning_curve, plot_learning_curve_with_baseline
from plot.plot_epsilon_decay import plot_epsilon
from plot.plot_gradient_norms import plot_gradient_norm
from plot.plot_all_paths import plot_all_paths
from plot.plot_loss import plot_loss
from plot.plot_convergence import plot_convergence
from utils.helpers import generate_random_path, generate_random_path_with_tl
from config.env_config import FOV_W, FOV_H, MAX_STEPS, RENDER_MODE_TRAIN
from config.train_config import NUM_EPISODES, LR, GAMMA, BATCH_SIZE, SAVE_PATH, MULTI_PATH, OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, TARGET_UPDATE_FREQ, NPC_PATH
from config.agent_config import INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY, N_CHANNELS
from config.paths import PATHS
from tests.baseline import evaluate_baseline_on_path

def save_env_state(env):
    return {
        "agent_pos": env.agent_pos.copy(),
        "path_index": env.path_index,
        "step_count": env.step_count,
        "npcs": copy.deepcopy(env.npcs),

        "position_history": copy.deepcopy(env.position_history)
            if hasattr(env, "position_history") else None,

        "agent_prev_pos": env.agent_prev_pos.copy()
            if hasattr(env, "agent_prev_pos") else None,
    }


def restore_env_state(env, snapshot):
    env.agent_pos = snapshot["agent_pos"].copy()
    env.path_index = snapshot["path_index"]
    env.step_count = snapshot["step_count"]
    env.npcs = copy.deepcopy(snapshot["npcs"])

    if snapshot["position_history"] is not None:
        env.position_history = copy.deepcopy(snapshot["position_history"])

    if snapshot["agent_prev_pos"] is not None:
        env.agent_prev_pos = snapshot["agent_prev_pos"].copy()

def train():
    # ---- Environment creation ----
    grid_map = GridMap(obstacle_map=OBSTACLE_MAP, traffic_light_map=TL_MAP, direction_map=DIRECTION_MAP)
    env = PathEnv(grid_map=grid_map, path=PATHS[0], fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN, max_steps=MAX_STEPS, num_npc=15, npc_policy_path=None)
    print("[INFO] Environment created.")
    #baseline_env = PathEnv(grid_map=grid_map, path=PATHS[0], fov=(FOV_W, FOV_H), max_steps=MAX_STEPS, render_mode=None, num_npc=15, npc_policy_path=NPC_PATH)
    print("[INFO] Baseline Environment created.")
    obs, _ = env.reset()

    # ---- Agent creation ----
    n_obs = FOV_W * FOV_H * N_CHANNELS  # trajectory map + obstacle map + traffic lights map + allowed dirs map
    n_actions = env.action_space.n
    agent = DQNAgent(env=env, n_obs=n_obs, n_actions=n_actions,learning_rate=LR, gamma=GAMMA, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON)
    print("[INFO] Agent created.")
    
    if env.num_npc > 0 and os.path.exists(NPC_PATH):
        agent.policy_net.load_state_dict(torch.load(NPC_PATH, map_location=agent.device))
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        print(f"[INFO] Loaded pretrained weights from {NPC_PATH}")
    

    # ---- Training variables setup ----
    n_episodes = NUM_EPISODES
    max_steps = MAX_STEPS
    episode_rewards = []
    baseline_rewards = []
    episode_epsilons = []
    #reward_per_path = {i: [] for i in range(len(PATHS))}
    global_steps = 0
    start_episode = 0
    tl_total = 0
    tl_other = 0
    tl_stay = 0

    
    start_time = time.time()
    best_avg_reward = -float("inf")
    checkpoint_freq = 10_000

    for ep in range(start_episode, n_episodes):

        # ---- Path selection ----
        #path = PATHS[ep % len(PATHS)]
        path = generate_random_path_with_tl(grid_map)
        env.setPath(path)

        # genera scenario iniziale UNA SOLA VOLTA
        obs, _ = env.reset()

        # snapshot dello stato iniziale
        snapshot = save_env_state(env)

        # baseline sullo stesso scenario
        baseline_reward = evaluate_baseline_on_path(env, do_reset=False)

        # ripristina scenario identico
        restore_env_state(env, snapshot)

        # ricostruisci observation iniziale
        obs = env._get_obs()
        state = env.obs_to_array(obs)
        total_reward = 0

        last_action = None

        # ---- Episode start ----
        for t in range(max_steps):

            action = agent.select_action(state)
            
            #---------------
            next_path_pos = env.path[env.path_index] if env.path_index < len(env.path) else None
            is_red_or_yellow = False
            if next_path_pos is not None and next_path_pos in env.traffic_lights:
                light = env.traffic_lights[next_path_pos]
                if light.isRed(env.step_count) or light.isYellow(env.step_count):
                    is_red_or_yellow = True

            # traccia
            if is_red_or_yellow:
                tl_total += 1
                if action == 4:  # STAY
                    tl_stay += 1
                else:
                    tl_other += 1
            
            if ep % 1000 == 0:
                print(f"TL situations: {tl_total} | STAY: {tl_stay} | Other: {tl_other}")

            #---------------

            next_obs, reward, done, truncated, _ = env.step(action)
            last_action = action

            
            next_state = env.obs_to_array(next_obs)
            done_flag = done or truncated
            agent.store_transition(state, action, reward, next_state, done_flag)
            
            if len(agent.buffer) > BATCH_SIZE:
                agent.update()

                if global_steps % TARGET_UPDATE_FREQ == 0:
                    agent.update_target()

            state = next_state
            total_reward += reward
            global_steps += 1
            
            if done or truncated:
                break

        episode_rewards.append(total_reward)
        baseline_rewards.append(baseline_reward)

        episode_epsilons.append(agent.epsilon)
        #reward_per_path[ep % len(PATHS)].append(total_reward)

        
        if len(episode_rewards) >= 100:
            moving_avg = np.mean(episode_rewards[-100:])
            agent.scheduler.step(moving_avg)
        
        # Reduce exploration rate
        agent.decay_epsilon(episode=ep)

        current_lr = agent.optimizer.param_groups[0]["lr"]
        print(f"Episode {ep} - Total reward: {total_reward} - Epsilon value: {agent.epsilon} - LR: {current_lr}")

        # ---- Save best checkpoint every 10k episodes ----
        """
        if (ep + 1) % checkpoint_freq == 0 and len(episode_rewards) >= 100:

            avg_reward = np.mean(episode_rewards[-100:])

            print(f"[CHECKPOINT] Episode {ep+1} | Avg reward (last 100): {avg_reward:.2f}")

            if avg_reward > best_avg_reward:

                best_avg_reward = avg_reward

                save_path = f"weights/best_proj_fov_1v60_ep{ep+1}.pt"

                torch.save(agent.policy_net.state_dict(), save_path)

                print(f"[INFO] New best model saved: {save_path}")
        """
    
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Training time: {total_time:.2f} sec ({total_time/60:.2f} min)")

    # save weights
    torch.save(agent.policy_net.state_dict(), "weights/proj_aware_1v15_baseline_weights.pt")


    # ---- Plots ----
    plot_learning_curve(episode_rewards= episode_rewards)
    plot_learning_curve_with_baseline(episode_rewards=episode_rewards, baseline_rewards=baseline_rewards)
    plot_convergence(episode_rewards=episode_rewards)
    plot_epsilon(epsilons=episode_epsilons)
    plot_gradient_norm(agent.grad_norms)
    plot_loss(agent.losses)
    #plot_all_paths(reward_per_path=reward_per_path)

    print(f"[INFO] Weight saved in proj_aware_1v15_baseline_weights.pt")

if __name__ == "__main__":
    train()
