"""
Shared pair and PPO configuration definitions for custom Meta-World multi-task experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class PairConfig:
    pair_id: str
    task_names: Tuple[str, str]
    default_total_timesteps: int
    horizon_label: str


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    net_arch: Tuple[int, int]


PAIRS: Dict[str, PairConfig] = {
    "button_push": PairConfig(
        pair_id="button_push",
        task_names=("button-press-v3", "push-v3"),
        default_total_timesteps=10_000_000,
        horizon_label="10m",
    ),
    "basketball_pickplace": PairConfig(
        pair_id="basketball_pickplace",
        task_names=("basketball-v3", "pick-place-v3"),
        default_total_timesteps=5_000_000,
        horizon_label="5m",
    ),
    "basketball_push": PairConfig(
        pair_id="basketball_push",
        task_names=("basketball-v3", "push-v3"),
        default_total_timesteps=10_000_000,
        horizon_label="10m",
    ),
}


PPO_CONFIGS: Dict[str, PPOConfig] = {
    "config_1": PPOConfig(
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.7,
        max_grad_norm=0.5,
        net_arch=(256, 256),
    ),
    "config_2": PPOConfig(
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=1024,
        n_epochs=15,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.10,
        ent_coef=0.002,
        vf_coef=0.8,
        max_grad_norm=0.3,
        net_arch=(256, 256),
    ),
    "config_3": PPOConfig(
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        net_arch=(256, 256),
    ),
}


def make_run_name(pair_id: str, horizon_label: str, config_name: str) -> str:
    return f"custom_mt_{pair_id}_ppo_{horizon_label}_{config_name}"
