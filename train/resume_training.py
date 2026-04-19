import torch
import numpy as np

from env.path_env import PathEnv
from agent.agent import DQNAgent
from env.maps import GridMap

from train.train import obs_to_array

from plot.plot_learning_queue import plot_learning_queue
from plot.plot_gradient_norms import plot_gradient_norm
from plot.plot_all_paths import plot_all_paths
from plot.plot_epsilon_decay import plot_epsilon

from config.env_config import FOV_W, FOV_H, MAX_STEPS, RENDER_MODE_TRAIN
from config.train_config import (
    GAMMA,
    BATCH_SIZE,
    SAVE_PATH,
    OBSTACLE_MAP,
    TL_MAP,
    DIRECTION_MAP,
    TARGET_UPDATE_FREQ,
)
from config.agent_config import N_CHANNELS
from config.paths import PATHS


# --- file dei pesi da cui ripartire ---
PRETRAINED_WEIGHTS = SAVE_PATH

# --- output del fine-tuning ---
RESUMED_SAVE_PATH = "dqn_finetuned_from_checkpoint.pt"

# --- quanti episodi aggiuntivi fare ---
RESUME_EPISODES = 15000

# --- fine-tuning: LR più basso ---
RESUME_LR = 5e-6

# --- epsilon piccolo e fisso ---
RESUME_EPSILON = 0.02


def resume_train():
    grid_map = GridMap(
        obstacle_map=OBSTACLE_MAP,
        traffic_light_map=TL_MAP,
        direction_map=DIRECTION_MAP
    )

    env = PathEnv(
        grid_map=grid_map,
        path=PATHS[0],
        fov=(FOV_W, FOV_H),
        render_mode=RENDER_MODE_TRAIN,
        max_steps=MAX_STEPS
    )
    print("[INFO] Environment created.")

    n_obs = FOV_W * FOV_H * N_CHANNELS
    n_actions = env.action_space.n

    agent = DQNAgent(
        env=env,
        n_obs=n_obs,
        n_actions=n_actions,
        learning_rate=RESUME_LR,
        gamma=GAMMA,
        initial_epsilon=RESUME_EPSILON,
        final_epsilon=RESUME_EPSILON,   # epsilon fisso
    )

    print("[INFO] Agent created.")

    # --- carica i pesi della policy ---
    policy_state = torch.load(PRETRAINED_WEIGHTS, map_location=agent.device)
    agent.policy_net.load_state_dict(policy_state)

    # --- allinea anche la target net ---
    agent.target_net.load_state_dict(policy_state)
    agent.target_net.eval()

    # --- epsilon fisso basso per il fine-tuning ---
    agent.epsilon = RESUME_EPSILON

    episode_rewards = []
    episode_epsilons = []
    reward_per_path = {i: [] for i in range(len(PATHS))}
    global_steps = 0

    for ep in range(RESUME_EPISODES):
        path_id = ep % len(PATHS)
        path = PATHS[path_id]

        env.setPath(path)
        obs, _ = env.reset()
        state = obs_to_array(obs)
        total_reward = 0

        for t in range(MAX_STEPS):
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

            if done_flag:
                break

        episode_rewards.append(total_reward)
        episode_epsilons.append(agent.epsilon)
        reward_per_path[path_id].append(total_reward)

        if len(episode_rewards) >= 100:
            moving_avg = np.mean(episode_rewards[-100:])
            agent.scheduler.step(moving_avg)

        print(
            f"[RESUME] Episode {ep + 1}/{RESUME_EPISODES} - "
            f"Path {path_id} - Reward: {total_reward:.2f} - "
            f"Epsilon: {agent.epsilon:.4f}"
        )

        # checkpoint intermedio ogni 1000 episodi
        if (ep + 1) % 1000 == 0:
            ckpt_name = f"dqn_resume_ep_{ep + 1}.pt"
            torch.save(agent.policy_net.state_dict(), ckpt_name)
            print(f"[INFO] Saved intermediate checkpoint: {ckpt_name}")

    torch.save(agent.policy_net.state_dict(), RESUMED_SAVE_PATH)
    print(f"[INFO] Final fine-tuned weights saved in {RESUMED_SAVE_PATH}")

    plot_all_paths(reward_per_path=reward_per_path)
    plot_learning_queue(episode_rewards=episode_rewards)
    plot_epsilon(epsilons=episode_epsilons)
    plot_gradient_norm(agent.grad_norms)


if __name__ == "__main__":
    resume_train()