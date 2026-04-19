import torch
import os
import numpy as np
from env.path_env import PathEnv
from agent.agent import DQNAgent
from env.maps import GridMap
from plot.plot_learning_queue import plot_learning_queue
from plot.plot_epsilon_decay import plot_epsilon
from plot.plot_gradient_norms import plot_gradient_norm
from plot.plot_all_paths import plot_all_paths
from utils.helpers import generate_random_path, generate_random_path_with_tl
from config.env_config import FOV_W, FOV_H, MAX_STEPS, RENDER_MODE_TRAIN
from config.train_config import NUM_EPISODES, LR, GAMMA, BATCH_SIZE, SAVE_PATH, OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, TARGET_UPDATE_FREQ
from config.agent_config import INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY, N_CHANNELS
from config.paths import PATHS


# ------ CHECKPOINT CONFIGURATION ------
CHECKPOINT_PATH = "checkpoint_latest.pt"
CHECKPOINT_EVERY = 5000
RESUME_FROM = None # path to checkpoint (if any)

# ------ LR DROP CONFIGURATION ------
LR_DROP_EPISODE = 70000
FINETUNE_LR = 5e-6


def obs_to_array(obs, fov_h=FOV_H, fov_w=FOV_W):
    """
    Convert observation dict into a flat array for the neural network.
    Output shape:
    - trajectory map          (fov_h * fov_w)
    - obstacle map            (fov_h * fov_w)
    - traffic light map       (fov_h * fov_w)
    - allowed directions map  (fov_h * fov_w)
    """

    # --- Trajectory map ---
    traj_map = np.zeros((fov_h, fov_w), dtype=np.float32)
    for x, y in obs["trajectory"]:
        if 0 <= x < fov_w and 0 <= y < fov_h:
            traj_map[y, x] = 1.0

    # --- Obstacle map ---
    obstacle_map = obs["obstacles"].astype(np.float32)

    # --- Traffic lights map ---
    # normalize: 0–3 → 0–1
    traffic_map = obs["traffic_lights"].astype(np.float32) / 3.0

    allowed_dirs = obs["allowed_dirs"].astype(np.float32)

    # --- Flatten ---
    return np.concatenate([
        traj_map.flatten(),
        obstacle_map.flatten(),
        traffic_map.flatten(),
        allowed_dirs.flatten()
    ])

def set_optimizer_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def save_checkpoint(
    checkpoint_path,
    agent,
    episode,
    global_steps,
    episode_rewards,
    episode_epsilons,
    reward_per_path,
    lr_drop_done,
):
    checkpoint = {
        "episode": episode,
        "global_steps": global_steps,
        "policy_state_dict": agent.policy_net.state_dict(),
        "target_state_dict": agent.target_net.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
        "scheduler_state_dict": agent.scheduler.state_dict(),
        "epsilon": agent.epsilon,
        "buffer": agent.buffer,
        "grad_norms": agent.grad_norms,
        "episode_rewards": episode_rewards,
        "episode_epsilons": episode_epsilons,
        "reward_per_path": reward_per_path,
        "lr_drop_done": lr_drop_done,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"[INFO] Checkpoint salvato in: {checkpoint_path}")


def load_checkpoint(checkpoint_path, agent):
    checkpoint = torch.load(checkpoint_path, map_location=agent.device)

    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    agent.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    agent.epsilon = checkpoint["epsilon"]
    agent.buffer = checkpoint["buffer"]
    agent.grad_norms = checkpoint.get("grad_norms", [])

    start_episode = checkpoint["episode"] + 1
    global_steps = checkpoint["global_steps"]
    episode_rewards = checkpoint.get("episode_rewards", [])
    episode_epsilons = checkpoint.get("episode_epsilons", [])
    reward_per_path = checkpoint.get(
        "reward_per_path",
        {i: [] for i in range(len(PATHS))}
    )
    lr_drop_done = checkpoint.get("lr_drop_done", False)

    print(f"[INFO] Checkpoint caricato da: {checkpoint_path}")
    print(f"[INFO] Riprendo da episodio {start_episode}, global_steps={global_steps}")

    return (
        start_episode,
        global_steps,
        episode_rewards,
        episode_epsilons
    )

def train(resume_from=None):
    # ---- Environment creation ----
    grid_map = GridMap(obstacle_map=OBSTACLE_MAP, traffic_light_map=TL_MAP, direction_map=DIRECTION_MAP)
    env = PathEnv(grid_map=grid_map, path=PATHS[0], fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN, max_steps=MAX_STEPS)
    print("[INFO] Environment created.")
    obs, _ = env.reset()

    # ---- Agent creation ----
    n_obs = FOV_W * FOV_H * N_CHANNELS  # trajectory map + obstacle map + traffic lights map + allowed dirs map
    n_actions = env.action_space.n
    agent = DQNAgent(env=env, n_obs=n_obs, n_actions=n_actions,learning_rate=LR, gamma=GAMMA, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON)
    print("[INFO] Agent created.")

    # ---- Training variables setup ----
    n_episodes = NUM_EPISODES
    max_steps = MAX_STEPS
    episode_rewards = []
    episode_epsilons = []
    global_steps = 0
    start_episode = 0

    # resume checkpoint (if requested)
    if resume_from is not None and os.path.exists(resume_from):
        (
            start_episode,
            global_steps,
            episode_rewards,
            episode_epsilons
        ) = load_checkpoint(resume_from, agent)

    for ep in range(start_episode, n_episodes):

        # ---- Random path selection ----
        #path = PATHS[ep % len(PATHS)]
        path = generate_random_path_with_tl(grid_map)
        env.setPath(path)

        # Episode reset
        obs, _ = env.reset()
        state = obs_to_array(obs)
        total_reward = 0

        # ---- Episode start ----
        for t in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            next_state = obs_to_array(next_obs)
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
        #reward_per_path[ep % len(PATHS)].append(total_reward)

        
        if len(episode_rewards) >= 100:
            moving_avg = np.mean(episode_rewards[-100:])
            agent.scheduler.step(moving_avg)
        
        # Reduce exploration rate
        agent.decay_epsilon(episode=ep)

        current_lr = agent.optimizer.param_groups[0]["lr"]
        print(f"Episode {ep} - Total reward: {total_reward} - Epsilon value: {agent.epsilon} - LR: {current_lr}")
    
        # checkpoint
        """
        if (ep + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(
                CHECKPOINT_PATH,
                agent,
                ep,
                global_steps,
                episode_rewards,
                episode_epsilons,
                reward_per_path,
                lr_drop_done,
            )
        """

    # save weights
    torch.save(agent.policy_net.state_dict(), SAVE_PATH)

    # final checkpoint
    """
    save_checkpoint(
        CHECKPOINT_PATH,
        agent,
        NUM_EPISODES - 1,
        global_steps,
        episode_rewards,
        episode_epsilons,
        reward_per_path,
        lr_drop_done,
    )
    """
    # ---- Plots ----
    #plot_all_paths(reward_per_path=reward_per_path)
    plot_learning_queue(episode_rewards= episode_rewards)
    plot_epsilon(epsilons=episode_epsilons)
    plot_gradient_norm(agent.grad_norms)

    print(f"[INFO] Weight saved in {SAVE_PATH}")

if __name__ == "__main__":
    train()
