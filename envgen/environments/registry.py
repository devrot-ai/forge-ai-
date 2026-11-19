"""Environment registry for saving and loading generated environment configs.

This module provides a way to save, load, and list environment configurations
with metadata (generation cycle, LLM prompt, difficulty, timestamp). This enables
reproducibility by allowing you to reuse previously generated environments.

No external dependencies required — uses only Python standard library.

Example:
    registry = EnvironmentRegistry(registry_dir="./env_registry")
    registry.save(config, cycle=1, prompt="Generate a puzzle...")
    configs = registry.list()
    loaded = registry.load("puzzle_easy_001")
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from envgen.environments.base import EnvironmentConfig


class EnvironmentRegistry:
    """Registry for saving and loading environment configurations.

    Stores environment configs as JSON files with metadata including generation
    cycle, LLM prompt, difficulty, and timestamp. Each config gets a unique ID.

    Configuration:
    - registry_dir: Directory to store registry files (default: './env_registry')

    Example:
        registry = EnvironmentRegistry(registry_dir="./env_registry")
        registry.save(config, cycle=1, prompt="Generate a puzzle...")
        configs = registry.list()
        loaded = registry.load("puzzle_easy_001")
    """

    def __init__(self, registry_dir: str = "./env_registry"):
        """Initialize the environment registry.

        Args:
            registry_dir: Directory to store registry files
        """
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)
        self._index_path = os.path.join(self.registry_dir, "_index.json")
        self._index = self._load_index()

    def save(
        self,
        config: EnvironmentConfig,
        cycle: Optional[int] = None,
        prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save an environment config to the registry.

        Args:
            config: Environment config to save
            cycle: Training cycle number (optional)
            prompt: LLM prompt used to generate this config (optional)
            metadata: Additional metadata (optional)

        Returns:
            Unique ID assigned to this config
        """
        env_id = config.name.replace(" ", "_").replace(" ", "_") + "_" + uuid.uuid4().hex[:8]
        filename = env_id + ".json"
        filepath = os.path.join(self.registry_dir, filename)

        entry = {
            "id": env_id,
            "config": config.to_dict(),
            "metadata": {
                "cycle": cycle,
                "prompt": prompt,
                "difficulty": config.difficulty.value,
                "target_skills": config.target_skills,
                "timestamp": datetime.now().isoformat(),
                **(metadata or {}),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)

        # Update index
        self._index[env_id] = {
            "id": env_id,
            "filename": filename,
            "name": config.name,
            "difficulty": config.difficulty.value,
            "cycle": cycle,
            "timestamp": entry["metadata"]["timestamp"],
        }
        self._save_index()

        return env_id

    def load(self, env_id: str) -> EnvironmentConfig:
        """Load an environment config from the registry.

        Args:
            env_id: Unique ID of the config to load

        Returns:
            EnvironmentConfig object

        Raises:
            FileNotFoundError: If the config is not found in the registry
        """
        if env_id not in self._index:
            raise FileNotFoundError(f"Environment config not found: {env_id}")

        filepath = os.path.join(self.registry_dir, self._index[env_id]["filename"])
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Environment config file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            entry = json.load(f)

        return EnvironmentConfig.from_dict(entry["config"])

    def load_with_metadata(self, env_id: str) -> tuple[EnvironmentConfig, Dict[str, Any]]:
        """Load an environment config with its metadata.

        Args:
            env_id: Unique ID of the config to load

        Returns:
            Tuple of (EnvironmentConfig, metadata dict)

        Raises:
            FileNotFoundError: If the config is not found
        """
        if env_id not in self._index:
            raise FileNotFoundError(f"Environment config not found: {env_id}")

        filepath = os.path.join(self.registry_dir, self._index[env_id]["filename"])
        with open(filepath, "r", encoding="utf-8") as f:
            entry = json.load(f)

        return EnvironmentConfig.from_dict(entry["config"]), entry.get("metadata", {})

    def list(self, difficulty: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all environment configs in the registry.

        Args:
            difficulty: Optional filter by difficulty level name

        Returns:
            List of index entries, optionally filtered by difficulty
        """
        entries = list(self._index.values())
        if difficulty:
            entries = [e for e in entries if e.get("difficulty") == difficulty]
        return entries

    def delete(self, env_id: str) -> bool:
        """Delete an environment config from the registry.

        Args:
            env_id: Unique ID of the config to delete

        Returns:
            True if deleted, False if not found
        """
        if env_id not in self._index:
            return False

        filepath = os.path.join(self.registry_dir, self._index[env_id]["filename"])
        if os.path.exists(filepath):
            os.remove(filepath)

        del self._index[env_id]
        self._save_index()
        return True

    def get_count(self) -> int:
        """Get the total number of configs in the registry.

        Returns:
            Number of configs
        """
        return len(self._index)

    def _load_index(self) -> Dict[str, Any]:
        """Load the registry index from disk."""
        if os.path.exists(self._index_path):
            with open(self._index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        """Save the registry index to disk."""
        with open(self._index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)
