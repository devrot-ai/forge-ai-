"""Parallel environment execution using multiprocessing.

This module provides a wrapper that runs multiple environment instances in
parallel worker processes, similar to Gymnasium's SubprocVecEnv. It collects
observations, actions, and rewards from multiple workers simultaneously,
enabling faster training by parallelizing episode collection.

Usage:
    from envgen.environments.parallel_env import ParallelEnvironment

    env = ParallelEnvironment(
        env_factory=lambda: PuzzleEnvironment(config),
        num_workers=4,
    )
    obs_list = env.reset()
    action_list = [agent.select_action(obs) for obs in obs_list]
    next_obs, rewards, dones, infos = env.step(action_list)
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from envgen.environments.base import Environment, EnvironmentConfig
from envgen.utils.types import Action, Observation


def _worker_pipe(
    parent_conn,
    worker_conn,
    env_factory: Callable[[], Environment],
):
    """Worker process that runs a single environment instance.

    Args:
        parent_conn: Connection to parent process (closed in worker).
        worker_conn: Connection to parent for receiving commands.
        env_factory: Callable that creates a fresh environment instance.
    """
    # Close the parent end in the worker
    parent_conn.close()

    env = env_factory()

    while True:
        try:
            command, data = worker_conn.recv()
        except Exception:
            worker_conn.close()
            return

        if command == "reset":
            seed = data
            if seed is not None:
                env.seed(seed)
            obs = env.reset()
            worker_conn.send(obs)

        elif command == "step":
            action = data
            obs, reward, done, info = env.step(action)
            worker_conn.send((obs, reward, done, info))

        elif command == "get_actions":
            actions = env.get_available_actions()
            worker_conn.send(actions)

        elif command == "render":
            rendered = env.render()
            worker_conn.send(rendered)

        elif command == "close":
            worker_conn.close()
            return

        else:
            worker_conn.send(None)


class ParallelEnvironment:
    """Runs multiple environment instances in parallel worker processes.

    This wrapper spawns multiple worker processes, each running its own
    environment instance. Actions can be sent to all workers simultaneously,
    and results are collected from all workers.

    Attributes:
        num_workers: Number of parallel environment workers.
        env_configs: List of environment configs (one per worker).
    """

    def __init__(
        self,
        env_factory: Callable[[], Environment],
        num_workers: int = 4,
        seed: Optional[int] = None,
    ):
        """Initialize parallel environment workers.

        Args:
            env_factory: Callable that creates a fresh environment instance.
                The same factory is used for all workers.
            num_workers: Number of parallel worker processes.
            seed: Optional base seed for reproducibility.
        """
        self.num_workers = num_workers
        self._env_factory = env_factory
        self._base_seed = seed

        # Track worker state
        self._dones = [False] * num_workers
        self._obs_list: List[Observation] = [None] * num_workers  # type: ignore
        self._closed = False

        # Start worker processes
        self._parent_conns = []
        self._worker_conns = []
        self._processes = []

        self._start_workers()

    def _start_workers(self):
        """Start all worker processes and establish connections."""
        for i in range(self.num_workers):
            parent_conn, worker_conn = mp.Pipe(duplex=True)
            ctx = mp.get_context("spawn")
            proc = ctx.Process(
                target=_worker_pipe,
                args=(parent_conn, worker_conn, self._env_factory),
                daemon=True,
            )
            proc.start()

            self._parent_conns.append(parent_conn)
            self._worker_conns.append(worker_conn)
            self._processes.append(proc)

    def reset(self, seeds: Optional[List[int]] = None) -> List[Observation]:
        """Reset all parallel environments.

        Args:
            seeds: Optional list of seeds, one per worker. If None, uses
                base seed incremented per worker.

        Returns:
            List of initial observations, one per worker.
        """
        if self._closed:
            raise RuntimeError("ParallelEnvironment is closed")

        if seeds is None:
            if self._base_seed is not None:
                seeds = [self._base_seed + i for i in range(self.num_workers)]
            else:
                seeds = [None] * self.num_workers

        # Send reset commands to all workers
        for i, parent_conn in enumerate(self._parent_conns):
            parent_conn.send(("reset", seeds[i] if i < len(seeds) else None))

        # Collect observations
        self._obs_list = []
        for parent_conn in self._parent_conns:
            obs = parent_conn.recv()
            self._obs_list.append(obs)

        self._dones = [False] * self.num_workers
        return self._obs_list

    def step(self, actions: List[Action]) -> Tuple[List[Observation], List[float], List[bool], List[Dict[str, Any]]]:
        """Execute actions in all parallel environments.

        Args:
            actions: List of actions, one per worker environment.

        Returns:
            Tuple of (observations, rewards, dones, infos), each a list
            with one entry per worker.
        """
        if self._closed:
            raise RuntimeError("ParallelEnvironment is closed")

        if len(actions) != self.num_workers:
            raise ValueError(f"Expected {self.num_workers} actions, got {len(actions)}")

        # Send actions to all workers
        for i, (parent_conn, action) in enumerate(zip(self._parent_conns, actions)):
            if not self._dones[i]:
                parent_conn.send(("step", action))
            else:
                # Already done, send noop
                parent_conn.send(("step", Action(action_type="noop")))

        # Collect results
        obs_list = []
        reward_list = []
        done_list = []
        info_list = []

        for i, parent_conn in enumerate(self._parent_conns):
            result = parent_conn.recv()
            obs, reward, done, info = result
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            self._dones[i] = done
            self._obs_list[i] = obs

        return obs_list, reward_list, done_list, info_list

    def get_available_actions(self) -> List[List[Action]]:
        """Get available actions for all parallel environments.

        Returns:
            List of action lists, one per worker.
        """
        if self._closed:
            raise RuntimeError("ParallelEnvironment is closed")

        for parent_conn in self._parent_conns:
            parent_conn.send(("get_actions", None))

        all_actions = []
        for parent_conn in self._parent_conns:
            actions = parent_conn.recv()
            all_actions.append(actions)

        return all_actions

    def render(self) -> List[str]:
        """Render all parallel environments.

        Returns:
            List of rendered strings, one per worker.
        """
        if self._closed:
            raise RuntimeError("ParallelEnvironment is closed")

        for parent_conn in self._parent_conns:
            parent_conn.send(("render", None))

        rendered = []
        for parent_conn in self._parent_conns:
            r = parent_conn.recv()
            rendered.append(r)

        return rendered

    @property
    def dones(self) -> List[bool]:
        """Current done status for each worker."""
        return self._dones.copy()

    @property
    def observations(self) -> List[Observation]:
        """Current observations for each worker."""
        return self._obs_list.copy()

    def close(self):
        """Close all worker processes and connections."""
        if self._closed:
            return

        self._closed = True
        for parent_conn in self._parent_conns:
            try:
                parent_conn.send(("close", None))
            except Exception:
                pass

        for proc in self._processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

        for conn in self._parent_conns + self._worker_conns:
            try:
                conn.close()
            except Exception:
                pass

    def __del__(self):
        """Ensure workers are cleaned up."""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
