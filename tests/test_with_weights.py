import torch
import numpy as np
from env.path_env import PathEnv
from train.train import obs_to_array
from agent.agent import DQNAgent
from agent.dqn_model import DQNNet
from env.maps import GridMap
from config.env_config import FOV_H, FOV_W, RENDER_MODE_TEST, MAX_STEPS
from config.paths import TEST_PATH
from config.train_config import OBSTACLE_MAP, TL_MAP, SAVE_PATH
from config.agent_config import N_CHANNELS, N_ACTIONS


# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

map = GridMap(obstacle_map=OBSTACLE_MAP, traffic_light_map=TL_MAP)
env = PathEnv(grid_map=map, path=TEST_PATH, fov=(FOV_W, FOV_H), render_mode=RENDER_MODE_TEST, max_steps=MAX_STEPS)
obs, _ = env.reset()

# --- Inizializza rete e carica pesi ---
input_dim = FOV_W * FOV_H * N_CHANNELS
n_actions = N_ACTIONS
agent = DQNAgent(env=env,n_obs=input_dim, n_actions=n_actions)
policy_net = DQNNet(n_obs=input_dim, n_actions=n_actions).to(agent.device)
policy_net.load_state_dict(torch.load(SAVE_PATH, map_location=device))
policy_net.eval()

done = False
step = 0
while not done:
    state = obs_to_array(obs)
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = policy_net(state_tensor)
        action = int(q_values.argmax().item())

    obs, reward, terminated, truncated, info = env.step(action)
    step += 1
    done = terminated or truncated
    print("[STEP ",step, "] ACTION", action, "REWARD: ", reward," DONE: ", done)

env.close()
