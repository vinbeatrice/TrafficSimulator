import torch
import numpy as np
from env.path_env import PathEnv
from agent.agent import DQNAgent
from env.maps import GridMap
from plot.plot_learning_queue import plot_learning_queue
from plot.plot_epsilon_decay import plot_epsilon
from utils.helpers import generate_random_path
from config.env_config import FOV_W, FOV_H, MAX_STEPS, RENDER_MODE_TRAIN
from config.train_config import NUM_EPISODES, LR, GAMMA, BATCH_SIZE, SAVE_PATH, OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, TARGET_UPDATE_FREQ
from config.agent_config import INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY, N_CHANNELS
from config.paths import TEST_PATH


def obs_to_array(obs, fov_h=FOV_H, fov_w=FOV_W):
    """
    Convert observation dict into a flat array for the neural network.
    Output shape:
    - trajectory map      (fov_h * fov_w)
    - obstacle map        (fov_h * fov_w)
    - traffic light map   (fov_h * fov_w)
    - border map          (fov_h * fov_w)
    - one way border map  (fov_h * fov_w)
    """

    # --- Trajectory map ---
    traj_map = np.zeros((fov_h, fov_w), dtype=np.float32)

    fov_xmin = obs["fov"][0][0]
    fov_ymin = obs["fov"][0][1]

    for x, y in obs["trajectory"]:  # ← è una LISTA
        rx = x - fov_xmin
        ry = y - fov_ymin
        if 0 <= rx < fov_w and 0 <= ry < fov_h:
            traj_map[ry, rx] = 1.0

    # --- Obstacle map ---
    obstacle_map = obs["obstacles"].astype(np.float32)

    # --- Road Borders map ---
    #border_map = obs["borders"].astype(np.float32) / 5.0

    # --- One Way Road Borders map ---
    # normalize: 0–4 → 0–1
    #one_way_border_map = obs["one_way_borders"].astype(np.float32) / 4.0

    # --- Traffic lights map ---
    # normalize: 0–3 → 0–1
    traffic_map = obs["traffic_lights"].astype(np.float32) / 3.0

    allowed_dirs = obs["allowed_dirs"].astype(np.float32)

    # --- Flatten ---
    return np.concatenate([
        traj_map.flatten(),
        obstacle_map.flatten(),
        #border_map.flatten(),
        #one_way_border_map.flatten(),
        traffic_map.flatten(),
        allowed_dirs.flatten()
    ])


def train():
    grid_map = GridMap(obstacle_map=OBSTACLE_MAP, traffic_light_map=TL_MAP, direction_map=DIRECTION_MAP)
    env = PathEnv(grid_map=grid_map, path=TEST_PATH, fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN, max_steps=MAX_STEPS)
    print("[INFO] Environment created.")
    obs, _ = env.reset()

    n_obs = FOV_W * FOV_H * N_CHANNELS  # trajectory map + obstacle map + traffic lights map + border map + one way border map
    n_actions = env.action_space.n
    agent = DQNAgent(env=env, n_obs=n_obs, n_actions=n_actions,learning_rate=LR, gamma=GAMMA, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON)
    print("[INFO] Agent created.")

    n_episodes = NUM_EPISODES
    max_steps = MAX_STEPS
    episode_rewards = []
    episode_epsilons = []

    for ep in range(n_episodes):
        #path = generate_random_path(grid_map=grid_map)
        #env.setPath(path)
        obs, _ = env.reset()
        state = obs_to_array(obs)
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, _ = env.step(action)
            
            next_state = obs_to_array(next_obs)
            done_flag = done or truncated
            agent.store_transition(state, action, reward, next_state, done_flag)
            
            if len(agent.buffer) > 1000:
                agent.update()

            state = next_state
            total_reward += reward
            if done or truncated:
                break

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        #if ep % 100 == 0:
        #    print(f"[LR] {agent.optimizer.param_groups[0]['lr']:.2e}")

        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)
        
        if len(episode_rewards) >= 50:
            moving_avg = np.mean(episode_rewards[-50:])
            agent.scheduler.step(moving_avg)
        
        # Reduce exploration rate
        #agent.decay_epsilon(episode=ep)

        print(f"Episode {ep} - Total reward: {total_reward} - Epsilon value: {agent.epsilon}")
    
    torch.save(
        agent.policy_net.state_dict(),
        SAVE_PATH
    )

    plot_learning_queue(episode_rewards= episode_rewards)
    plot_epsilon(epsilons=episode_epsilons)

    print("Weight saved in dqn_final_weights.pt")

if __name__ == "__main__":
    train()
