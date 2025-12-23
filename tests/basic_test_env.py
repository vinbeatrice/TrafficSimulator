import time
from env.path_env import PathEnv 

"""Deterministic visual smoke test for PathEnv environment."""


def main():
    print("Running deterministic visual test...")

    # Simple deterministic path
    path = [(0,0), (1,0), (2,0), (3,0), (4,0)]

    # Environment with render_mode='human' to open the Pygame window
    env = PathEnv(
        grid_size=(10, 6),
        path=path,
        fov=(5, 3),
        render_mode="human"
    )

    # Initial reset
    obs, info = env.reset()
    print("Initial observation:", obs)

    # Deterministic action sequence:
    # RIGHT = 0 LEFT = 2based on your action_to_direction
    actions = [0, 0, 0, 2]

    try:
        for step_idx, a in enumerate(actions):
            print(f"\nStep {step_idx} - action = {a} (RIGHT)")
            obs, reward, terminated, truncated, info = env.step(a)

            print("   agent_pos:", obs["agent_pos"])
            print("   reward:", reward)

            # Wait a bit to visualize
            time.sleep(0.5)

            if terminated:
                print("\nReached goal — episode terminated.")
                break

        # Keep the window open for 2 seconds after the end
        time.sleep(2.0)

    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
