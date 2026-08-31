"""
Train PPO on a custom two-task Meta-World multi-task environment.

The environment samples one of the pair's MT1 environments uniformly at the start
of every episode and appends a one-hot task id to the observation.

Examples:
    python scripts/training/train_custom_mt_pair.py --pair button_push --combo config_1
    python scripts/training/train_custom_mt_pair.py --pair basketball_pickplace --combo config_1
    python scripts/training/train_custom_mt_pair.py --pair basketball_push --combo config_1
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path
from typing import Optional, Sequence

# Keep each worker lightweight when using SubprocVecEnv on CPU.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
from gymnasium import spaces
import metaworld  #type: ignore
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from custom_mt_config import PAIRS, PPO_CONFIGS, make_run_name

ENV_ID = "Meta-World/MT1"
DEFAULT_SEED = 42
DEFAULT_N_ENVS = 8
DEFAULT_MAX_EPISODE_STEPS = 500
DEFAULT_REWARD_TYPE = "v2"
DEFAULT_CHECKPOINT_EVERY_TIMESTEPS = 50_000


class CustomMTPairEnv(gym.Env):
    #Simple two-task Meta-World wrapper with one-hot task.

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        task_names: Sequence[str],
        seed: int = DEFAULT_SEED,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        reward_type: str = DEFAULT_REWARD_TYPE,
        terminate_on_success: bool = False,
        append_task_id: bool = True,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        if len(task_names) != 2:
            raise ValueError("This custom-MT wrapper expects exactly two task names.")

        self.task_names = list(task_names)
        self.seed_value = seed
        self.max_episode_steps = max_episode_steps
        self.reward_type = reward_type
        self.terminate_on_success = terminate_on_success
        self.append_task_id = append_task_id
        self.render_mode = render_mode
        self.rng = np.random.default_rng(seed)

        self.envs = []
        for idx, task_name in enumerate(self.task_names):
            env = gym.make(
                ENV_ID,
                env_name=task_name,
                task_select="pseudorandom",
                terminate_on_success=terminate_on_success,
                max_episode_steps=max_episode_steps,
                seed=seed + idx,
                reward_function_version=reward_type,
                render_mode=render_mode,
            )
            try:
                env.get_wrapper_attr("toggle_sample_tasks_on_reset")(True)
            except Exception as exc:
                print(f"Warning: could not enable task sampling for {task_name}: {exc}")
            self.envs.append(env)

        self.action_space = self.envs[0].action_space
        for env in self.envs[1:]:
            if env.action_space.shape != self.action_space.shape:
                raise ValueError("All task action spaces must have the same shape.")

        base_obs_space = self.envs[0].observation_space
        if not isinstance(base_obs_space, spaces.Box):
            raise TypeError("Expected Box observation space from Meta-World env.")

        base_shape = base_obs_space.shape
        for task_name, env in zip(self.task_names[1:], self.envs[1:]):
            if env.observation_space.shape != base_shape:
                raise ValueError(f"Observation shape mismatch for {task_name}.")

        self.task_id_dim = len(self.task_names) if append_task_id else 0
        low = np.asarray(base_obs_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_obs_space.high, dtype=np.float32).reshape(-1)
        if append_task_id:
            low = np.concatenate([low, np.zeros(self.task_id_dim, dtype=np.float32)])
            high = np.concatenate([high, np.ones(self.task_id_dim, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.active_task_idx = 0
        self.active_env = self.envs[0]

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if not self.append_task_id:
            return obs
        task_one_hot = np.zeros(self.task_id_dim, dtype=np.float32)
        task_one_hot[self.active_task_idx] = 1.0
        return np.concatenate([obs, task_one_hot]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and "task_idx" in options:
            self.active_task_idx = int(options["task_idx"])
        else:
            self.active_task_idx = int(self.rng.integers(0, len(self.envs)))

        self.active_env = self.envs[self.active_task_idx]
        obs, info = self.active_env.reset()
        info = dict(info)
        info["task_name"] = self.task_names[self.active_task_idx]
        info["task_idx"] = self.active_task_idx
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active_env.step(action)
        info = dict(info)
        info["task_name"] = self.task_names[self.active_task_idx]
        info["task_idx"] = self.active_task_idx
        return self._augment_obs(obs), float(reward), terminated, truncated, info

    def render(self):
        return self.active_env.render()

    def close(self):
        for env in self.envs:
            env.close()


def make_env(rank: int, task_names: Sequence[str], seed: int, max_episode_steps: int, reward_type: str,
             terminate_on_success: bool, append_task_id: bool):
    def _init():
        return CustomMTPairEnv(
            task_names=task_names,
            seed=seed + rank * 1000,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
            terminate_on_success=terminate_on_success,
            append_task_id=append_task_id,
            render_mode=None,
        )
    return _init


def make_vec_env(task_names: Sequence[str], seed: int, n_envs: int, start_method: str,
                 max_episode_steps: int, reward_type: str, terminate_on_success: bool,
                 append_task_id: bool) -> VecNormalize:
    env = SubprocVecEnv(
        [make_env(i, task_names, seed, max_episode_steps, reward_type, terminate_on_success, append_task_id)
         for i in range(n_envs)],
        start_method=start_method,
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on a custom two-task Meta-World environment.")
    parser.add_argument("--pair", choices=sorted(PAIRS.keys()), required=True)
    parser.add_argument("--combo", choices=sorted(PPO_CONFIGS.keys()), default="config_1")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--horizon-label", default=None, help="Run-name label such as 5m or 10m. Defaults from pair.")
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--start-method", choices=["spawn", "forkserver", "fork"], default="spawn")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY_TIMESTEPS)
    parser.add_argument("--max-episode-steps", type=int, default=DEFAULT_MAX_EPISODE_STEPS)
    parser.add_argument("--reward-type", choices=["v1", "v2"], default=DEFAULT_REWARD_TYPE)
    parser.add_argument("--terminate-on-success", action="store_true")
    parser.add_argument("--no-task-id", action="store_true")
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = PAIRS[args.pair]
    cfg = PPO_CONFIGS[args.combo]
    total_timesteps = args.timesteps if args.timesteps is not None else pair.default_total_timesteps
    horizon_label = args.horizon_label if args.horizon_label is not None else pair.horizon_label
    append_task_id = not args.no_task_id

    run_name = make_run_name(args.pair, horizon_label, args.combo)
    run_dir = Path(args.runs_root) / args.pair / run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Pair: {args.pair}")
    print(f"Tasks: {list(pair.task_names)}")
    print(f"Config: {args.combo}")
    print(f"Run name: {run_name}")
    print(f"Run dir: {run_dir}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"n_envs: {args.n_envs}")
    print(f"rollout size: {cfg.n_steps * args.n_envs:,}")
    print(f"checkpoint every: {args.checkpoint_every:,}")
    print(f"device: {args.device}")
    print(f"append task id: {append_task_id}")
    print("=" * 100)

    torch.set_num_threads(1)
    env = make_vec_env(
        task_names=pair.task_names,
        seed=args.seed,
        n_envs=args.n_envs,
        start_method=args.start_method,
        max_episode_steps=args.max_episode_steps,
        reward_type=args.reward_type,
        terminate_on_success=args.terminate_on_success,
        append_task_id=append_task_id,
    )

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=args.verbose,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(tensorboard_dir),
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch))),
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix=run_name,
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=run_name,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        model_path = run_dir / f"{run_name}_final"
        vecnormalize_path = run_dir / f"{run_name}_vecnormalize.pkl"
        model.save(str(model_path))
        env.save(str(vecnormalize_path))
        print(f"Saved model: {model_path}.zip")
        print(f"Saved VecNormalize: {vecnormalize_path}")
    finally:
        env.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
