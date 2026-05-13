import numpy as np
import pygame
import torch
from env.path_env import PathEnv, Actions
from utils.helpers import getTrajectoryinFOV, getFOV_with_layers
from env.directions import Direction


class MultiAgentPathEnv(PathEnv):

    def __init__(self, *args, npc_policy_path=None, **kwargs):
        super().__init__(*args, **kwargs)

        # ---- NPC ----
        self.npc = None

        # ---- Load v0 policy ----
        self.npc_policy = None
        if npc_policy_path is not None:
            from agent.agent import DQNAgent

            n_obs = self.fov_w * self.fov_h * 4  # SENZA nuovo layer
            n_actions = self.action_space.n

            self.npc_policy = DQNAgent(
                env=self,
                n_obs=n_obs,
                n_actions=n_actions
            )

            self.npc_policy.policy_net.load_state_dict(
                torch.load(npc_policy_path, map_location=self.npc_policy.device)
            )
            self.npc_policy.policy_net.eval()

            for p in self.npc_policy.policy_net.parameters():
                p.requires_grad = False


    # =========================
    # RESET
    # =========================
    def reset(self, seed=None, options=None):

        obs, info = super().reset(seed=seed, options=options)

        # ---- spawn NPC ----
        npc_path = self._generate_npc_path()

        self.npc = {
            "pos": np.array(npc_path[0], dtype=np.int32),
            "path": npc_path,
            "path_index": 1,
            "path_index_map": {pos: i for i, pos in enumerate(npc_path)},
            "dir": Direction.UP
        }

        return self._get_obs(), info


    def _generate_npc_path(self):
        from utils.helpers import generate_random_path
        return generate_random_path(self.map, max_length=30)


    # =========================
    # STEP
    # =========================
    def step(self, action):

        # ---- 1. MOVE NPC ----
        self._move_npc()

        # ---- 2. MOVE AGENT ----
        obs, reward, terminated, truncated, info = super().step(action)

        # ---- 3. COLLISION ----
        if np.array_equal(self.agent_pos, self.npc["pos"]):
            reward -= 20.0
            terminated = True

        return obs, reward, terminated, truncated, info


    # =========================
    # NPC LOGIC
    # =========================
    def _move_npc(self):

        if self.npc_policy is None:
            return

        # --- STOP se path finito ---
        if self.npc["path_index"] >= len(self.npc["path"]):
            action = Actions.STAY.value
        else:
            # ---- build observation ----
            obs = self._get_obs_for_npc()
            state = self._obs_to_array(obs)
            action = self.npc_policy.select_action(state, greedy=True)

        # ---- move ----
        direction = self._action_to_direction[action]
        new_pos = self.npc["pos"] + direction
        new_pos = np.clip(new_pos, [0, 0], [self.W - 1, self.H - 1])

        self.npc["pos"] = new_pos

        # ---- update direction ----
        if action == 0:
            self.npc["dir"] = Direction.RIGHT
        elif action == 1:
            self.npc["dir"] = Direction.UP
        elif action == 2:
            self.npc["dir"] = Direction.LEFT
        elif action == 3:
            self.npc["dir"] = Direction.DOWN

        # ---- UPDATE PATH INDEX ----
        new_pos_tuple = (int(new_pos[0]), int(new_pos[1]))

        idx = self.npc["path_index_map"].get(new_pos_tuple, -1)

        if idx != -1 and idx >= self.npc["path_index"]:
            self.npc["path_index"] = idx + 1



    def _get_obs_for_npc(self):
        """
        Versione semplificata: NPC vede come agente
        """

        fov_data = getFOV_with_layers(
            agent_pos=self.npc["pos"],
            agent_dir=self.npc["dir"],
            fov_w=self.fov_w,
            fov_h=self.fov_h,
            grid_map=self.map,
            traffic_lights=self.traffic_lights,
            step_count=self.step_count
        )

        traj = getTrajectoryinFOV(
            fov_data["fov_bounds"],
            self.npc["path"],
            start_idx=self.npc["path_index"]
        )

        # inject AGENT as obstacle for NPC
        fov = fov_data["fov_bounds"]
        x_min, y_min = fov[0]

        ax, ay = self.agent_pos

        rx = ax - x_min
        ry = ay - y_min

        if 0 <= rx < self.fov_w and 0 <= ry < self.fov_h:
            fov_data["obstacles"][ry, rx] = 1

        return {
            "fov": np.array(fov_data["fov_bounds"]),
            "trajectory": traj,
            "obstacles": fov_data["obstacles"],
            "traffic_lights": fov_data["traffic_lights"],
            "allowed_dirs": fov_data["allowed_dirs"]
        }


    def _obs_to_array(self, obs):
        """
        Copia della tua obs_to_array
        """
        traj_map = np.zeros((self.fov_h, self.fov_w), dtype=np.float32)

        for x, y in obs["trajectory"]:
            if 0 <= x < self.fov_w and 0 <= y < self.fov_h:
                traj_map[y, x] = 1.0

        obstacle_map = obs["obstacles"].astype(np.float32)
        traffic_map = obs["traffic_lights"].astype(np.float32) / 3.0
        allowed_dirs = obs["allowed_dirs"].astype(np.float32)

        return np.concatenate([
            traj_map.flatten(),
            obstacle_map.flatten(),
            traffic_map.flatten(),
            allowed_dirs.flatten()
        ])


    # =========================
    # OVERRIDE OBS
    # =========================
    def _get_obs(self):

        obs = super()._get_obs()

        # ---- inject NPC into obstacle map ----
        fov = obs["fov"]
        x_min, y_min = fov[0]

        npc_x, npc_y = self.npc["pos"]

        rx = npc_x - x_min
        ry = npc_y - y_min

        if 0 <= rx < self.fov_w and 0 <= ry < self.fov_h:
            obs["obstacles"][ry, rx] = 1

        return obs


    # =========================
    # RENDER
    # =========================
    def _render_frame(self):

        # base render
        canvas = super()._render_frame()

        if self.render_mode == "rgb_array":
            return canvas

        # --- draw NPC on top ---
        if self.npc is not None:

            pix = self.window_size / self.W

            x, y = self.npc["pos"]

            px = y * pix
            py = x * pix

            pygame.draw.circle(
                self.window,
                (255, 0, 0),
                (int(px + pix / 2), int(py + pix / 2)),
                int(pix / 3)
            )

        return canvas