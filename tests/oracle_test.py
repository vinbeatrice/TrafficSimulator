import numpy as np
import time

from env.path_env import PathEnv
from env.maps import GridMap
from agent.agent import DQNAgent
from utils.helpers import generate_random_path, generate_random_path_with_tl
from tests.baseline import baseline_policy

from config.env_config import FOV_W, FOV_H, MAX_STEPS
from config.train_config import OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, SAVE_PATH, MULTI_PATH, NPC_PATH
from config.agent_config import N_CHANNELS


def run_episode(env):

    obs, _ = env.reset()
    policy_state = {
        "overtake_mode": False,
        "overtake_entry_action": None,
        "overtake_forward_action": None,
        "overtake_progress": 0
    }

    total_reward = 0
    done = False
    truncated = False

    while not (done or truncated):

        action, policy_state = baseline_policy(obs, policy_state)

        next_obs, reward, done, truncated, _ = env.step(action)
        obs = next_obs

        total_reward += reward
        time.sleep(0.6)

    
    success = env.path_index >= len(env.path)

    return total_reward, success


def evaluate_agent(env, n_paths=100, max_length=20):

    results = []

    for i in range(n_paths):

        path = generate_random_path_with_tl(env.map, max_length=max_length)
        env.setPath(path)

        # Oracle baseline
        oracle_reward, oracle_success = run_episode(env)

        results.append({
            "oracle_reward": oracle_reward,
            "oracle_success": oracle_success,
        })

        print(
            f"[{i}] "
            f"agent: {oracle_reward:.2f} ({oracle_success})"
        )

    return results

def main():

    # --- ENV ---
    grid_map = GridMap(
        obstacle_map=OBSTACLE_MAP,
        traffic_light_map=TL_MAP,
        direction_map=DIRECTION_MAP
    )

    dummy_path = generate_random_path(grid_map)

    env = PathEnv(
        grid_map=grid_map,
        path=dummy_path,
        fov=(FOV_W, FOV_H),
        max_steps=MAX_STEPS,
        render_mode="human",
        num_npc=1,
        npc_policy_path=NPC_PATH
    )

    # --- AGENT ---
    n_obs = FOV_W * FOV_H * N_CHANNELS
    n_actions = env.action_space.n


    # --- EVALUATION ---
    results = evaluate_agent(env, n_paths=1000)

    #summarize_results(results)


if __name__ == "__main__":
    main()