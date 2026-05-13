import torch
import time
import os
import numpy as np
from env.path_env import PathEnv
from agent.agent import DQNAgent
from env.maps import GridMap
from plot.plot_learning_queue import plot_learning_queue
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



def train(resume_from=None):
    # ---- Environment creation ----
    grid_map = GridMap(obstacle_map=OBSTACLE_MAP, traffic_light_map=TL_MAP, direction_map=DIRECTION_MAP)
    #env = PathEnv(grid_map=grid_map, path=PATHS[0], fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN, max_steps=MAX_STEPS)
    env = PathEnv(grid_map=grid_map, path=PATHS[0], fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN, max_steps=MAX_STEPS, num_npc=10, npc_policy_path=NPC_PATH)
    print("[INFO] Environment created.")
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
    episode_epsilons = []
    reward_per_path = {i: [] for i in range(len(PATHS))}
    global_steps = 0
    start_episode = 0

    
    start_time = time.time()

    for ep in range(start_episode, n_episodes):

        # ---- Path selection ----
        path = PATHS[ep % len(PATHS)]
        #path = generate_random_path_with_tl(grid_map)
        env.setPath(path)

        # Episode reset
        obs, _ = env.reset()
        state = env.obs_to_array(obs)
        total_reward = 0

        # ---- Episode start ----
        for t in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, _ = env.step(action)
            
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

        episode_epsilons.append(agent.epsilon)
        reward_per_path[ep % len(PATHS)].append(total_reward)

        
        if len(episode_rewards) >= 100:
            moving_avg = np.mean(episode_rewards[-100:])
            agent.scheduler.step(moving_avg)
        
        # Reduce exploration rate
        agent.decay_epsilon(episode=ep)

        current_lr = agent.optimizer.param_groups[0]["lr"]
        print(f"Episode {ep} - Total reward: {total_reward} - Epsilon value: {agent.epsilon} - LR: {current_lr}")
    
    
    total_time = time.time() - start_time
    print(f"\n[INFO] Training time: {total_time:.2f} sec ({total_time/60:.2f} min)")

    # save weights
    torch.save(agent.policy_net.state_dict(), "weights/1v10_weights_zero_idle_multi_path.pt")


    # ---- Plots ----
    plot_learning_queue(episode_rewards= episode_rewards)
    plot_convergence(episode_rewards=episode_rewards)
    plot_epsilon(epsilons=episode_epsilons)
    plot_gradient_norm(agent.grad_norms)
    plot_loss(agent.losses)
    plot_all_paths(reward_per_path=reward_per_path)

    print(f"[INFO] Weight saved in 1v10_weights_zero_idle_multi_path.pt")

if __name__ == "__main__":
    train()
