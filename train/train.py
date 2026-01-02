import torch
import numpy as np
from env.path_env import PathEnv
from agent.agent import DQNAgent
from config.env_config import GRID_W, GRID_H, FOV_W, FOV_H, MAX_STEPS, RENDER_MODE_TRAIN
from config.train_config import NUM_EPISODES, GAMMA, BATCH_SIZE, SAVE_PATH
from config.agent_config import INITIAL_EPSILON, FINAL_EPSILON, EPSILON_DECAY
from config.paths import SIMPLE_PATH


def obs_to_array(obs, fov_w, fov_h):
    """
    Convert observation dict into a flat array for the neural network.
    Output shape:
    - trajectory map      (fov_h * fov_w)
    - obstacle map        (fov_h * fov_w)
    - traffic light map   (fov_h * fov_w)
    - agent position      (2,)
    """

    # --- Trajectory map ---
    traj_map = np.zeros((fov_h, fov_w), dtype=np.float32)

    # agent_pos è RELATIVA al FOV
    ax, ay = obs["agent_pos"]

    # ricostruisco il minimo del FOV in coordinate globali
    # (serve solo per mappare la trajectory)
    fov_xmin = obs["fov"][0][0]
    fov_ymin = obs["fov"][0][1]

    for x, y in obs["trajectory"]:
        rx = x - fov_xmin
        ry = y - fov_ymin
        if 0 <= rx < fov_w and 0 <= ry < fov_h:
            traj_map[ry, rx] = 1.0

    # --- Obstacle map ---
    obstacle_map = obs["obstacles"].astype(np.float32)

    # --- Traffic lights map ---
    # normalize: 0–3 → 0–1
    traffic_map = obs["traffic_lights"].astype(np.float32) / 3.0

    # --- Agent position ---
    agent_pos = np.array([ax, ay], dtype=np.float32)
    # agent_rel = np.array([obs['agent_pos'][0] - fov_xmin, obs['agent_pos'][1] - fov_ymin], dtype=np.float32) # agent relative position in fov

    # --- Flatten ---
    return np.concatenate([
        traj_map.flatten(),
        obstacle_map.flatten(),
        traffic_map.flatten(),
        agent_pos
    ])


def train():
    env = PathEnv(grid_size=(GRID_W, GRID_H), path=SIMPLE_PATH, fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TRAIN)
    obs, _ = env.reset()

    n_obs = FOV_W*FOV_H + 2  # mappa FOV + agent_pos
    n_actions = env.action_space.n
    agent = DQNAgent(env=env, n_obs=n_obs, n_actions=n_actions,learning_rate=1e-3, gamma=GAMMA, initial_epsilon=INITIAL_EPSILON, final_epsilon=FINAL_EPSILON, epsilon_decay=EPSILON_DECAY)

    n_episodes = NUM_EPISODES
    max_steps = MAX_STEPS

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state = obs_to_array(obs, FOV_W, FOV_H)
        total_reward = 0
        for t in range(max_steps):
            action = agent.select_action(state)
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = obs_to_array(next_obs, FOV_W, FOV_H)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update(batch_size=BATCH_SIZE)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        agent.update_target()
        print(f"Episode {ep} - Total reward: {total_reward} - Epsilon value: {agent.epsilon}")
    
    torch.save(
        agent.policy_net.state_dict(),
        SAVE_PATH
    )

    print("Weight saved in dqn_final_weights.pt")

if __name__ == "__main__":
    train()
