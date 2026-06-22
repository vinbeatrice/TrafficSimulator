import numpy as np
import torch

from env.path_env import PathEnv
from env.maps import GridMap
from agent.agent import DQNAgent
from utils.helpers import generate_random_path, generate_random_path_with_tl
from tests.baseline import baseline_policy

from config.env_config import FOV_W, FOV_H, MAX_STEPS
from config.train_config import OBSTACLE_MAP, TL_MAP, DIRECTION_MAP, SAVE_PATH, MULTI_PATH, NPC_PATH
from config.agent_config import N_CHANNELS



# =========================
# ORACLE POLICY (perfect path follower)
# =========================
def oracle_action(env):

    if env.path_index >= len(env.path):
        return 4  # STAY

    target = env.path[env.path_index]
    curr = env.agent_pos

    dx = target[0] - curr[0]
    dy = target[1] - curr[1]

    if dx == -1: return 1  # UP
    if dx == 1:  return 3  # DOWN
    if dy == 1:  return 0  # RIGHT
    if dy == -1: return 2  # LEFT

    return 4


# =========================
# EPISODE ROLLOUT (greedy only)
# =========================
def run_episode(env, agent=None, use_oracle=False):

    obs, _ = env.reset()
    state = env.obs_to_array(obs)

    total_reward = 0
    done = False
    truncated = False

    # baseline internal state
    policy_state = {
        "overtake_mode": False,
        "overtake_entry_action": None,
        "overtake_forward_action": None,
        "overtake_progress": 0
    }

    while not (done or truncated):

        if use_oracle:
            action, policy_state = baseline_policy(obs, policy_state)

        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                action = agent.policy_net(state_tensor).argmax().item()

        next_obs, reward, done, truncated, _ = env.step(action)

        obs = next_obs
        state = env.obs_to_array(next_obs)

        total_reward += reward

    print("DONE")
    success = env.path_index >= len(env.path)

    return total_reward, success

# =========================
# EVALUATION LOOP
# =========================
def evaluate_agent(env, agent, n_paths=100, max_length=20):

    results = []

    for i in range(n_paths):

        path = generate_random_path_with_tl(env.map, max_length=max_length)
        env.setPath(path)

        # Oracle baseline
        oracle_reward, oracle_success = run_episode(env, use_oracle=True)

        env.setPath(path)

        # Agent performance
        agent_reward, agent_success = run_episode(env, agent=agent)

        results.append({
            "agent_reward": agent_reward,
            "agent_success": agent_success,
        })

        print(
            f"[{i}] "
            f"agent: {agent_reward:.2f} ({agent_success})"
        )

    return results


# =========================
# SUMMARY
# =========================
def summarize_results(results):

    oracle_rewards = np.array([r["oracle_reward"] for r in results])
    agent_rewards = np.array([r["agent_reward"] for r in results])

    oracle_success = np.array([r["oracle_success"] for r in results])
    agent_success = np.array([r["agent_success"] for r in results])

    print("\n===== RESULTS =====")
    print(f"Oracle avg reward: {oracle_rewards.mean():.2f}")
    print(f"Agent avg reward:  {agent_rewards.mean():.2f}")
    print(f"Gap: {(agent_rewards - oracle_rewards).mean():.2f}")

    print(f"Oracle success rate: {oracle_success.mean()*100:.1f}%")
    print(f"Agent success rate:  {agent_success.mean()*100:.1f}%")


# =========================
# MAIN
# =========================
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
        num_npc=15,
        npc_policy_path=NPC_PATH
    )

    # --- AGENT ---
    n_obs = FOV_W * FOV_H * N_CHANNELS
    n_actions = env.action_space.n

    agent = DQNAgent(
        env=env,
        n_obs=n_obs,
        n_actions=n_actions
    )

    # ⚠️ CARICA PESI
    agent.policy_net.load_state_dict(
        torch.load("weights/proj_aware_1v15_weights.pt", map_location=agent.device)
    )
    agent.policy_net.eval()

    # --- EVALUATION ---
    results = evaluate_agent(env, agent, n_paths=1000)

    summarize_results(results)


if __name__ == "__main__":
    main()