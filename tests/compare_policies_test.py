import copy
import numpy as np
import torch
import time

from env.path_env import PathEnv
from env.maps import GridMap
from agent.agent import DQNAgent
from utils.helpers import generate_random_path, generate_random_path_with_tl
from tests.baseline import baseline_policy

from config.env_config import FOV_W, FOV_H, MAX_STEPS
from config.train_config import OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, NPC_PATH
from config.agent_config import N_CHANNELS


# =========================
# SNAPSHOT UTILS
# =========================

def save_env_state(env):
    return {
        "agent_pos": env.agent_pos.copy(),
        "agent_prev_pos": env.agent_prev_pos.copy(),
        "path_index": env.path_index,
        "step_count": env.step_count,
        "npcs": copy.deepcopy(env.npcs)
    }


def restore_env_state(env, snapshot):
    env.agent_pos = snapshot["agent_pos"].copy()
    env.agent_prev_pos = snapshot["agent_prev_pos"].copy()
    env.path_index = snapshot["path_index"]
    env.step_count = snapshot["step_count"]
    env.npcs = copy.deepcopy(snapshot["npcs"])


# =========================
# POLICY LOADING
# =========================

def load_agent(env, weights_path):
    n_obs = FOV_W * FOV_H * N_CHANNELS
    n_actions = env.action_space.n

    agent = DQNAgent(
        env=env,
        n_obs=n_obs,
        n_actions=n_actions
    )

    agent.policy_net.load_state_dict(
        torch.load(weights_path, map_location=agent.device)
    )
    agent.policy_net.eval()

    return agent


# =========================
# POLICY EXECUTION
# =========================

def select_action(agent, state):
    state_tensor = torch.tensor(
        state,
        dtype=torch.float32,
        device=agent.device
    ).unsqueeze(0)

    with torch.no_grad():
        action = agent.policy_net(state_tensor).argmax().item()

    return action


def run_episode(env, agent):
    obs = env._get_obs()
    state = env.obs_to_array(obs)
    policy_state = {
        "overtake_mode": False,
        "overtake_entry_action": None,
        "overtake_forward_action": None,
        "overtake_progress": 0
    }

    total_reward = 0
    done = False
    truncated = False

    collision = False
    tl_violation = False

    while not (done or truncated):

        if agent is not None:
            action = select_action(agent, state)
        else:
            action, policy_state = baseline_policy(obs, policy_state)


        next_obs, reward, done, truncated, _ = env.step(action)
        #time.sleep(0.4)

        total_reward += reward
        obs = next_obs
        state = env.obs_to_array(next_obs)

        # Se hai flag nell'env usa quelli:
        if done and reward < -70:
            collision = True
        elif done and reward < -30:
            tl_violation = True

    return {
        "reward": total_reward,
        "collision": collision,
        "traffic_violation": tl_violation
    }


# =========================
# EVALUATION
# =========================

def evaluate_policies(env, agent1, agent2, agent3, n_episodes=1000, max_length=20):

    p1_rewards = []
    p2_rewards = []
    p3_rewards = []
    p4_rewards = []

    p1_collisions = 0
    p2_collisions = 0
    p3_collisions = 0
    p4_collisions = 0

    p1_tl = 0
    p2_tl = 0
    p3_tl = 0
    p4_tl = 0

    for ep in range(n_episodes):

        path = generate_random_path_with_tl(env.map, max_length=max_length)
        env.setPath(path)

        obs, _ = env.reset()


        snapshot = save_env_state(env)

        
        # ---- POLICY 1 ----
        result1 = run_episode(env, agent1)

        if result1["collision"]:
            p1_collisions += 1
        if result1["traffic_violation"]:
            p1_tl += 1

        
        # ---- RESTORE ----
        restore_env_state(env, snapshot)
        

        # ---- POLICY 2 ----
        result2 = run_episode(env, agent2)

        if result2["collision"]:
            p2_collisions += 1
        if result2["traffic_violation"]:
            p2_tl += 1
        

        # ---- RESTORE ----
        restore_env_state(env, snapshot)

        # ---- POLICY 3 ----
        result3 = run_episode(env, agent3)

        if result3["collision"]:
            p3_collisions += 1
        if result3["traffic_violation"]:
            p3_tl += 1
        

        # ---- RESTORE ----
        restore_env_state(env, snapshot)

        # ---- BASELINE POLICY ----
        result4 = run_episode(env, agent=None)

        if result4["collision"]:
            p4_collisions += 1
        if result4["traffic_violation"]:
            p4_tl += 1

        p1_rewards.append(result1["reward"])
        p2_rewards.append(result2["reward"])
        p3_rewards.append(result3["reward"])
        p4_rewards.append(result4["reward"])

        print(
            f"[Episode {ep}] "
            f"P1 reward={result1['reward']:.2f} | "
            f"P2 reward={result2['reward']:.2f} | "
            f"P3 reward={result3['reward']:.2f} | "
            f"P4 reward={result4['reward']:.2f}"
        )

    print("\n===== RESULTS =====")
    print(f"P1 avg reward: {np.mean(p1_rewards):.2f}")
    print(f"P2 avg reward: {np.mean(p2_rewards):.2f}")
    print(f"P3 avg reward: {np.mean(p3_rewards):.2f}")
    print(f"Baseline avg reward: {np.mean(p4_rewards):.2f}")

    print(f"P1 collision rate: {100 * p1_collisions / n_episodes:.2f}%")
    print(f"P2 collision rate: {100 * p2_collisions / n_episodes:.2f}%")
    print(f"P3 collision rate: {100 * p3_collisions / n_episodes:.2f}%")
    print(f"Baseline collision rate: {100 * p4_collisions / n_episodes:.2f}%")

    print(f"P1 traffic violation rate: {100 * p1_tl / n_episodes:.2f}%")
    print(f"P2 traffic violation rate: {100 * p2_tl / n_episodes:.2f}%")
    print(f"P3 traffic violation rate: {100 * p3_tl / n_episodes:.2f}%")
    print(f"Baseline traffic violation rate: {100 * p4_tl / n_episodes:.2f}%")


# =========================
# MAIN
# =========================

def main():

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
        render_mode=None,
        num_npc=60,
        npc_policy_path=NPC_PATH
    )

    policy1_path = "weights/proj_aware_1v15_weights.pt"
    policy2_path = "weights/proj_aware_1v15_baseline_weights.pt"
    policy3_path = "weights/proj_aware_1v30_weights.pt"

    agent1 = load_agent(env, policy1_path)
    agent2 = load_agent(env, policy2_path)
    agent3 = load_agent(env, policy3_path)

    evaluate_policies(
        env,
        agent1,
        agent2,
        agent3,
        n_episodes=1000
    )


if __name__ == "__main__":
    main()