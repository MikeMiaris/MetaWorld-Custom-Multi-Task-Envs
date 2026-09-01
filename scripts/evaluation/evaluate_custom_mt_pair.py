from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gymnasium as gym
from gymnasium import spaces
import metaworld  #type :ignore
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from custom_mt_config import PAIRS, PPO_CONFIGS, make_run_name

ENV_ID = "Meta-World/MT1"
STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or text.strip() == "":
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def safe_name(text: str) -> str:
    return text.replace("-", "_").replace("/", "_").replace(" ", "_").replace("\\", "_")


def get_task_wrapper(env: gym.Env):
    cur = env
    seen = set()
    while True:
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "tasks") and hasattr(cur, "toggle_sample_tasks_on_reset"):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
            continue
        unwrapped = getattr(cur, "unwrapped", None)
        if unwrapped is not None and unwrapped is not cur:
            cur = unwrapped
            continue
        break
    raise RuntimeError("Could not find Meta-World task wrapper.")


class SingleMT1WithTaskID(gym.Env):
    """One MT1 environment with the same one-hot task id used by training."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_name: str, task_idx: int, task_order: Sequence[str], seed: int,
                 terminate_on_success: bool, max_episode_steps: int, reward_type: str):
        super().__init__()
        self.env_name = env_name
        self.task_idx = int(task_idx)
        self.task_order = list(task_order)
        self.task_id_dim = len(self.task_order)

        self.env = gym.make(
            ENV_ID,
            env_name=env_name,
            task_select="pseudorandom",
            terminate_on_success=terminate_on_success,
            max_episode_steps=max_episode_steps,
            seed=seed,
            reward_function_version=reward_type,
        )
        self.task_wrapper = get_task_wrapper(self.env)
        try:
            self.task_wrapper.toggle_sample_tasks_on_reset(False)
        except Exception as exc:
            warnings.warn(f"Could not disable task sampling for {env_name}: {exc}")

        self.action_space = self.env.action_space
        base_obs_space = self.env.observation_space
        if not isinstance(base_obs_space, spaces.Box):
            raise TypeError("Expected Box observation space from Meta-World.")
        low = np.asarray(base_obs_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(base_obs_space.high, dtype=np.float32).reshape(-1)
        low = np.concatenate([low, np.zeros(self.task_id_dim, dtype=np.float32)])
        high = np.concatenate([high, np.ones(self.task_id_dim, dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        one_hot = np.zeros(self.task_id_dim, dtype=np.float32)
        one_hot[self.task_idx] = 1.0
        return np.concatenate([obs, one_hot]).astype(np.float32)

    def get_tasks(self) -> List[Any]:
        return list(self.task_wrapper.tasks)

    def set_task(self, task: Any) -> None:
        self.task_wrapper.toggle_sample_tasks_on_reset(False)
        self.env.unwrapped.set_task(task)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["env_name"] = self.env_name
        info["task_idx"] = self.task_idx
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["env_name"] = self.env_name
        info["task_idx"] = self.task_idx
        return self._augment_obs(obs), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()


def make_dummy_vecenv_for_vecnormalize(task_order: Sequence[str], max_episode_steps: int, reward_type: str) -> DummyVecEnv:
    return DummyVecEnv([
        lambda: SingleMT1WithTaskID(
            env_name=task_order[0],
            task_idx=0,
            task_order=task_order,
            seed=999,
            terminate_on_success=True,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
        )
    ])


def load_vecnormalize(vecnormalize_path: Path, task_order: Sequence[str], max_episode_steps: int, reward_type: str) -> VecNormalize:
    dummy_env = make_dummy_vecenv_for_vecnormalize(task_order, max_episode_steps, reward_type)
    vecnorm = VecNormalize.load(str(vecnormalize_path), dummy_env)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def normalize_obs(vecnorm: VecNormalize, obs: np.ndarray) -> np.ndarray:
    obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
    return vecnorm.normalize_obs(obs_batch)


def checkpoint_step_from_model_path(path: Path) -> Optional[int]:
    match = STEP_RE.search(path.name)
    if match:
        return int(match.group(1))
    return None


def passes_checkpoint_filter(step: int, min_step: Optional[int], exact_steps: Optional[List[int]], checkpoint_every: Optional[int]) -> bool:
    if min_step is not None and step < min_step:
        return False
    if exact_steps is not None:
        return step in set(int(x) for x in exact_steps)
    if checkpoint_every is not None:
        return step % int(checkpoint_every) == 0
    return True


def find_checkpoints(run_dir: Path, checkpoint_every: Optional[int], exact_steps: Optional[List[int]],
                     min_step: Optional[int], include_final: bool) -> List[Tuple[Path, Path, int, str]]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        print(f"Missing checkpoint folder: {checkpoints_dir}")
        return []

    items: List[Tuple[Path, Path, int, str]] = []
    for model_path in sorted(checkpoints_dir.glob("*.zip")):
        step = checkpoint_step_from_model_path(model_path)
        if step is None or not passes_checkpoint_filter(step, min_step, exact_steps, checkpoint_every):
            continue
        expected_suffix = f"_vecnormalize_{step}_steps.pkl"
        vec_matches = [p for p in checkpoints_dir.glob("*.pkl") if p.name.endswith(expected_suffix)]
        if len(vec_matches) != 1:
            print(f"Skipping {model_path.name}: expected one VecNormalize ending with {expected_suffix}, found {len(vec_matches)}")
            continue
        items.append((model_path, vec_matches[0], step, str(step)))

    items.sort(key=lambda x: x[2])
    if include_final:
        final_model = run_dir / f"{run_dir.name}_final.zip"
        final_vec = run_dir / f"{run_dir.name}_vecnormalize.pkl"
        if final_model.exists() and final_vec.exists():
            final_step = max([x[2] for x in items], default=0)
            items.append((final_model, final_vec, final_step, "final"))
    return items


def evaluate_model_on_env(model: PPO, vecnorm: VecNormalize, run_label: str, config_name: str, env_name: str,
                          task_idx: int, task_order: Sequence[str], eval_seed: int, num_goals: int,
                          episodes_per_goal: int, max_episode_steps: int, reward_type: str,
                          terminate_on_success: bool, deterministic: bool) -> List[Dict[str, Any]]:
    env = SingleMT1WithTaskID(
        env_name=env_name,
        task_idx=task_idx,
        task_order=task_order,
        seed=eval_seed,
        terminate_on_success=terminate_on_success,
        max_episode_steps=max_episode_steps,
        reward_type=reward_type,
    )
    all_tasks = env.get_tasks()
    n_goals = min(num_goals, len(all_tasks))
    rows: List[Dict[str, Any]] = []

    for goal_idx in range(n_goals):
        env.set_task(all_tasks[goal_idx])
        for episode_for_goal in range(episodes_per_goal):
            obs, _ = env.reset()
            episode_return = 0.0
            episode_steps = 0
            success = 0.0
            first_success_step = np.nan
            done = False
            while not done and episode_steps < max_episode_steps:
                norm_obs = normalize_obs(vecnorm, obs)
                action, _ = model.predict(norm_obs, deterministic=deterministic)
                if isinstance(action, np.ndarray) and action.ndim > 1:
                    action = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                if float(info.get("success", 0.0)) > 0.0:
                    success = 1.0
                    if np.isnan(first_success_step):
                        first_success_step = float(episode_steps)
                    if terminate_on_success:
                        done = True
                        break
                done = bool(terminated or truncated)
            rows.append({
                "run_label": run_label,
                "config_name": config_name,
                "env_name": env_name,
                "task_one_hot_idx": task_idx,
                "task_order": ",".join(task_order),
                "eval_seed": eval_seed,
                "goal_idx": goal_idx,
                "episode_for_goal": episode_for_goal,
                "success": success,
                "return": episode_return,
                "steps": episode_steps,
                "first_success_step": first_success_step,
            })
    env.close()
    return rows


def summarize_raw(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    by_seed = (
        raw_df.groupby(["run_label", "config_name", "checkpoint_step", "checkpoint_label", "env_name", "task_one_hot_idx", "eval_seed"])
        .agg(
            success_rate=("success", "mean"),
            avg_return=("return", "mean"),
            std_return=("return", "std"),
            avg_steps=("steps", "mean"),
            std_steps=("steps", "std"),
            avg_first_success_step=("first_success_step", "mean"),
            episodes=("success", "count"),
            goals=("goal_idx", "nunique"),
        )
        .reset_index()
    )
    summary = (
        by_seed.groupby(["run_label", "config_name", "checkpoint_step", "checkpoint_label", "env_name", "task_one_hot_idx"])
        .agg(
            mean_success_rate=("success_rate", "mean"),
            std_success_rate=("success_rate", "std"),
            mean_return=("avg_return", "mean"),
            std_return_across_seeds=("avg_return", "std"),
            mean_steps=("avg_steps", "mean"),
            mean_first_success_step=("avg_first_success_step", "mean"),
            eval_seeds=("eval_seed", "nunique"),
            total_episodes=("episodes", "sum"),
            goals=("goals", "max"),
        )
        .reset_index()
    )
    best_rows = []
    for _, sub in summary.groupby(["run_label", "config_name", "env_name"]):
        sub_ckpt = sub[sub["checkpoint_label"] != "final"]
        source = sub_ckpt if not sub_ckpt.empty else sub
        best = source.sort_values(["mean_success_rate", "mean_return", "checkpoint_step"], ascending=[False, False, True]).iloc[0]
        best_rows.append(best)
    best = pd.DataFrame(best_rows).reset_index(drop=True)
    return by_seed, summary, best


def plot_learning_curves(summary: pd.DataFrame, output_dir: Path) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_df = summary[summary["checkpoint_label"] != "final"].copy()
    if plot_df.empty:
        plot_df = summary.copy()

    for (run_label, config_name), run_sub in plot_df.groupby(["run_label", "config_name"]):
        run_sub = run_sub.sort_values("checkpoint_step")
        prefix = safe_name(f"{run_label}_{config_name}")

        # Combined success-rate curve: one line per environment.
        plt.figure(figsize=(10, 5))
        for env_name, env_sub in run_sub.groupby("env_name"):
            env_sub = env_sub.sort_values("checkpoint_step")
            plt.plot(
                env_sub["checkpoint_step"],
                env_sub["mean_success_rate"],
                marker="o",
                markersize=3,
                label=env_name,
            )
        plt.xlabel("Checkpoint step")
        plt.ylabel("Success rate")
        plt.title(f"{run_label} / {config_name}: success rate per environment")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_success_rate_per_env.png", dpi=200)
        plt.close()

        # Combined return curve: one line per environment.
        plt.figure(figsize=(10, 5))
        for env_name, env_sub in run_sub.groupby("env_name"):
            env_sub = env_sub.sort_values("checkpoint_step")
            plt.plot(
                env_sub["checkpoint_step"],
                env_sub["mean_return"],
                marker="o",
                markersize=3,
                label=env_name,
            )
        plt.xlabel("Checkpoint step")
        plt.ylabel("Average return")
        plt.title(f"{run_label} / {config_name}: average return per environment")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{prefix}_avg_return_per_env.png", dpi=200)
        plt.close()

        for env_name, env_sub in run_sub.groupby("env_name"):
            env_sub = env_sub.sort_values("checkpoint_step")
            env_safe = safe_name(env_name)

            plt.figure(figsize=(10, 5))
            plt.plot(
                env_sub["checkpoint_step"],
                env_sub["mean_success_rate"],
                marker="o",
                markersize=3,
            )
            plt.xlabel("Checkpoint step")
            plt.ylabel("Success rate")
            plt.title(f"{run_label} / {config_name}: {env_name} success rate")
            plt.ylim(-0.05, 1.05)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{prefix}_{env_safe}_success_rate.png", dpi=200)
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(
                env_sub["checkpoint_step"],
                env_sub["mean_return"],
                marker="o",
                markersize=3,
            )
            plt.xlabel("Checkpoint step")
            plt.ylabel("Average return")
            plt.title(f"{run_label} / {config_name}: {env_name} average return")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(figures_dir / f"{prefix}_{env_safe}_avg_return.png", dpi=200)
            plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate custom-MT PPO checkpoints.")
    parser.add_argument("--pair", choices=sorted(PAIRS.keys()), required=True)
    parser.add_argument("--configs", default="config_1,config_2,config_3")
    parser.add_argument("--horizon-label", default=None)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--checkpoint-every", type=int, default=50_000)
    parser.add_argument("--exact-checkpoints", default=None, help="Comma-separated checkpoint steps. Overrides checkpoint-every.")
    parser.add_argument("--min-checkpoint-step", type=int, default=None)
    parser.add_argument("--include-final", action="store_true")
    parser.add_argument("--eval-seeds", default="67")
    parser.add_argument("--num-goals", type=int, default=50)
    parser.add_argument("--episodes-per-goal", type=int, default=1)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--reward-type", choices=["v1", "v2"], default="v2")
    parser.add_argument("--terminate-on-success", action="store_true", default=True)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair = PAIRS[args.pair]
    configs = [x.strip() for x in args.configs.split(",") if x.strip()]
    for cfg in configs:
        if cfg not in PPO_CONFIGS:
            raise ValueError(f"Unknown config: {cfg}")
    horizon_label = args.horizon_label if args.horizon_label is not None else pair.horizon_label
    eval_seeds = parse_int_list(args.eval_seeds) or [67]
    exact_steps = parse_int_list(args.exact_checkpoints)

    output_dir = Path(args.results_root) / args.pair
    output_dir.mkdir(parents=True, exist_ok=True)
    task_order = list(pair.task_names)

    print("=" * 100)
    print(f"Pair: {args.pair}")
    print(f"Task order: {task_order}")
    print(f"Configs: {configs}")
    print(f"Runs root: {args.runs_root}")
    print(f"Output dir: {output_dir}")
    print(f"Eval seeds: {eval_seeds}")
    print("=" * 100)

    all_rows: List[Dict[str, Any]] = []
    for config_name in configs:
        run_name = make_run_name(args.pair, horizon_label, config_name)
        run_dir = Path(args.runs_root) / args.pair / run_name
        checkpoints = find_checkpoints(run_dir, args.checkpoint_every, exact_steps, args.min_checkpoint_step, args.include_final)
        print(f"\n{config_name}: {run_dir} -> selected checkpoints: {len(checkpoints)}")
        for model_path, vec_path, checkpoint_step, checkpoint_label in checkpoints:
            print(f"  checkpoint={checkpoint_label} | model={model_path.name} | vecnorm={vec_path.name}")
            model = PPO.load(str(model_path), device=args.device)
            vecnorm = load_vecnormalize(vec_path, task_order, args.max_episode_steps, args.reward_type)
            for task_idx, env_name in enumerate(task_order):
                for eval_seed in eval_seeds:
                    rows = evaluate_model_on_env(
                        model=model,
                        vecnorm=vecnorm,
                        run_label=args.pair,
                        config_name=config_name,
                        env_name=env_name,
                        task_idx=task_idx,
                        task_order=task_order,
                        eval_seed=eval_seed,
                        num_goals=args.num_goals,
                        episodes_per_goal=args.episodes_per_goal,
                        max_episode_steps=args.max_episode_steps,
                        reward_type=args.reward_type,
                        terminate_on_success=args.terminate_on_success,
                        deterministic=args.deterministic,
                    )
                    for row in rows:
                        row.update({
                            "checkpoint_step": checkpoint_step,
                            "checkpoint_label": checkpoint_label,
                            "model_path": str(model_path),
                            "vecnormalize_path": str(vec_path),
                        })
                    temp = pd.DataFrame(rows)
                    print(f"    seed={eval_seed:<5d} | {env_name:18s} | SR={temp['success'].mean():.3f} | Return={temp['return'].mean():.2f}")
                    all_rows.extend(rows)

    if not all_rows:
        print("No checkpoints evaluated. Check run paths/settings.")
        return

    raw_df = pd.DataFrame(all_rows)
    by_seed_df, summary_df, best_df = summarize_raw(raw_df)
    raw_df.to_csv(output_dir / "raw_episodes.csv", index=False)
    by_seed_df.to_csv(output_dir / "summary_by_seed.csv", index=False)
    summary_df.to_csv(output_dir / "checkpoint_summary.csv", index=False)
    best_df.to_csv(output_dir / "best_checkpoint_per_config_env.csv", index=False)
    pivot = best_df.pivot_table(index=["run_label", "config_name"], columns="env_name", values="mean_success_rate")
    pivot.to_csv(output_dir / "best_success_pivot.csv")
    plot_learning_curves(summary_df, output_dir)
    print("\nBEST CHECKPOINT PER CONFIG / ENV")
    print(best_df[["run_label", "config_name", "env_name", "checkpoint_step", "checkpoint_label", "mean_success_rate", "mean_return", "mean_steps", "mean_first_success_step", "eval_seeds", "total_episodes"]].to_string(index=False))
    print("\nSUCCESS PIVOT")
    print(pivot.to_string())


if __name__ == "__main__":
    main()
