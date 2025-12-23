import time
import torch
import numpy as np

from env.path_env import PathEnv
from agent.dqn_model import DQNNet
from train.train import obs_to_array

from config.env_config import GRID_W, GRID_H, FOV_W, FOV_H, RENDER_MODE_TEST
from config.paths import SIMPLE_PATH
from config.train_config import SAVE_PATH


# -------- Config --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Env (human render) --------
env = PathEnv(
    render_mode=RENDER_MODE_TEST,
    grid_size=(GRID_W, GRID_H),
    path=SIMPLE_PATH,
    fov=(FOV_W, FOV_H)
)

obs, _ = env.reset()

# -------- Load model --------
n_obs = FOV_W * FOV_H + 2
n_actions = 4

policy_net = DQNNet(n_obs, n_actions).to(device)
policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=device))
policy_net.eval()

done = False
total_reward = 0

print("▶ Starting rollout...")

while not done:
    state = obs_to_array(obs, FOV_W, FOV_H)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = int(q_values.argmax().item())

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated

    print(
        f"Action: {action} | "
        f"Pos: {env.agent_pos.tolist()} | "
        f"Reward: {reward} | "
        f"Path index: {env.path_index}"
    )

    time.sleep(0.3)

print("🏁 Episode finished")
print("Total reward:", total_reward)

env.close()
