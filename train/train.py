import torch
import numpy as np
from env.path_env import PathEnv
from agent.agent import DQNAgent
from env.maps import GridMap
from config.env_config import FOV_W, FOV_H, MAX_STEPS, RENDER_MODE_TRAIN
from config.train_config import NUM_EPISODES, GAMMA, BATCH_SIZE, SAVE_PATH, OBSTACLE_MAP, TL_MAP, BOARDERS, TARGET_UPDATE_FREQ
from config.agent_config import INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY, N_CHANNELS
from config.paths import TEST_PATH


def obs_to_array(obs, fov_h=FOV_H, fov_w=FOV_W):
    """
    Convert observation dict into a flat array for the neural network.
    Output shape:
    - trajectory map      (fov_h * fov_w)
    - obstacle map        (fov_h * fov_w)
    - traffic light map   (fov_h * fov_w)
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
    border_map = obs["borders"].astype(np.float32)

    # --- Traffic lights map ---
    # normalize: 0–3 → 0–1
    traffic_map = obs["traffic_lights"].astype(np.float32) / 3.0

    # --- Flatten ---
    return np.concatenate([
        traj_map.flatten(),
        obstacle_map.flatten(),
        border_map.flatten(),
        traffic_map.flatten()
    ])


def train():
    env = PathEnv(grid_map=GridMap(obstacle_map=OBSTACLE_MAP, traffic_light_map=TL_MAP, road_border_map=BOARDERS), path=TEST_PATH, fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN, max_steps=MAX_STEPS)
    print("[INFO] Environment created.")
    obs, _ = env.reset()

    n_obs = FOV_W * FOV_H * N_CHANNELS  # trajectory map + obstacle map + traffic lights map
    n_actions = env.action_space.n
    agent = DQNAgent(env=env, n_obs=n_obs, n_actions=n_actions,learning_rate=1e-3, gamma=GAMMA, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON)
    print("[INFO] Agent created.")

    n_episodes = NUM_EPISODES
    max_steps = MAX_STEPS

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs_to_array(obs)
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = obs_to_array(next_obs)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update(batch_size=BATCH_SIZE)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
        print(f"Episode {ep} - Total reward: {total_reward} - Epsilon value: {agent.epsilon}")
    
    torch.save(
        agent.policy_net.state_dict(),
        SAVE_PATH
    )

    print("Weight saved in dqn_final_weights.pt")

if __name__ == "__main__":
    train()
