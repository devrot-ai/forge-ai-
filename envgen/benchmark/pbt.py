"""Population-Based Training (PBT) for evolving agent hyperparameters.

This module implements Population-Based Training, which maintains a population
of agents with different hyperparameters. Periodically, the worst performers'
hyperparameters are mutated, and the best configurations are shared across
the population.

Usage:
    python -m envgen train --pbt --population-size 5 --cycles 10
"""

from __future__ import annotations

import copy
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from envgen.utils.types import DifficultyLevel, EvaluationResult

logger = logging.getLogger(__name__)


class AgentMember:
    """A single member of the PBT population with its own hyperparameters."""

    def __init__(self, member_id: int, hyperparams: Dict[str, Any]):
        self.member_id = member_id
        self.hyperparams = hyperparams.copy()
        self.best_score: float = float('-inf')
        self.current_score: float = 0.0
        self.generation: int = 0
        self.history: List[float] = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "member_id": self.member_id,
            "hyperparams": self.hyperparams,
            "best_score": self.best_score,
            "current_score": self.current_score,
            "generation": self.generation,
            "history": self.history[-100:],
        }

    def __repr__(self) -> str:
        return (
            f"AgentMember(id={self.member_id}, best={self.best_score:.2f}, "
            f"gen={self.generation})"
        )


class PopulationBasedTrainer:
    """Population-Based Training for evolving agent hyperparameters.

    Maintains a population of agents with different hyperparameters.
    Periodically evaluates all agents, mutates the worst performers'
    hyperparameters, and copies the best configurations to improve
    the overall population performance.

    The algorithm:
    1. Initialize population with random hyperparameters from search space
    2. Train each agent for a number of cycles
    3. Evaluate all agents and rank by performance
    4. Mutate worst performers' hyperparameters
    5. Copy best configurations to worst performers
    6. Repeat until convergence or max cycles reached
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the PBT trainer.

        Args:
            config: Configuration dict with optional keys:
                - population_size: number of agents in population (default: 5)
                - mutation_rate: probability of mutating each parameter (default: 0.1)
                - mutation_strength: magnitude of mutation (default: 0.1)
                - sharing_interval: cycles between sharing best configs (default: 5)
                - search_space: dict of parameter ranges for each hyperparameter
                - elite_fraction: fraction of population to keep as elite (default: 0.2)
        """
        self.config = config or {}
        self.population_size: int = self.config.get("population_size", 5)
        self.mutation_rate: float = self.config.get("mutation_rate", 0.1)
        self.mutation_strength: float = self.config.get("mutation_strength", 0.1)
        self.sharing_interval: int = self.config.get("sharing_interval", 5)
        self.elite_fraction: float = self.config.get("elite_fraction", 0.2)

        # Search space for hyperparameters
        self.search_space: Dict[str, List[float]] = self.config.get("search_space", {
            "learning_rate": [1e-4, 1e-3, 3e-3, 1e-2],
            "epsilon": [0.1, 0.2, 0.3, 0.5],
            "hidden_size": [32, 64, 128, 256],
        })

        # Population
        self.population: List[AgentMember] = []
        self.generation: int = 0

        # Stats
        self.best_ever_score: float = float('-inf')
        self.best_ever_hyperparams: Dict[str, Any] = {}
        self.generation_scores: List[float] = []

    def initialize_population(self) -> List[AgentMember]:
        """Initialize the population with random hyperparameters from search space.

        Returns:
            List of AgentMember instances with random hyperparameters
        """
        self.population = []

        for i in range(self.population_size):
            hyperparams = {}
            for param_name, param_values in self.search_space.items():
                if isinstance(param_values[0], (int, float)):
                    # Continuous or discrete numeric parameter
                    hyperparams[param_name] = random.choice(param_values)
                else:
                    # Categorical parameter
                    hyperparams[param_name] = random.choice(param_values)

            member = AgentMember(member_id=i, hyperparams=hyperparams)
            self.population.append(member)

        logger.info(
            f"Initialized PBT population with {self.population_size} members"
        )
        return self.population

    def mutate_hyperparams(self, member: AgentMember) -> AgentMember:
        """Mutate an agent's hyperparameters.

        Args:
            member: AgentMember to mutate

        Returns:
            New AgentMember with mutated hyperparameters
        """
        new_hyperparams = member.hyperparams.copy()

        for param_name, param_values in self.search_space.items():
            if random.random() < self.mutation_rate:
                if isinstance(param_values[0], (int, float)):
                    # Gaussian mutation for continuous parameters
                    current_value = new_hyperparams[param_name]
                    min_val = min(param_values)
                    max_val = max(param_values)
                    mutation = random.gauss(0, self.mutation_strength * (max_val - min_val))
                    new_value = current_value + mutation
                    new_value = max(min_val, min(max_val, new_value))
                    new_hyperparams[param_name] = new_value
                else:
                    # Categorical mutation
                    new_hyperparams[param_name] = random.choice(param_values)

        new_member = AgentMember(
            member_id=member.member_id,
            hyperparams=new_hyperparams,
        )
        new_member.generation = member.generation + 1
        return new_member

    def share_best_config(self, best_member: AgentMember, target_member: AgentMember) -> AgentMember:
        """Copy the best configuration to a target member.

        Args:
            best_member: Best performing agent
            target_member: Target agent to update

        Returns:
            Updated target member with best hyperparameters
        """
        new_member = AgentMember(
            member_id=target_member.member_id,
            hyperparams=best_member.hyperparams.copy(),
        )
        new_member.generation = target_member.generation + 1
        logger.info(
            f"Shared best config from member {best_member.member_id} "
            f"to member {target_member.member_id}"
        )
        return new_member

    def evaluate_population(
        self, scores: List[float]
    ) -> Tuple[List[float], List[int]]:
        """Evaluate the population and return sorted scores and indices.

        Args:
            scores: List of scores for each agent in the population

        Returns:
            Tuple of (sorted_scores_desc, sorted_indices)
        """
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        sorted_indices = [idx for idx, _ in indexed_scores]
        sorted_scores = [score for _, score in indexed_scores]
        return sorted_scores, sorted_indices

    def evolve(self, scores: List[float]) -> List[AgentMember]:
        """Evolve the population based on evaluation scores.

        Args:
            scores: List of scores for each agent in the population

        Returns:
            Updated population after evolution
        """
        if len(scores) != len(self.population):
            raise ValueError(
                f"Scores length ({len(scores)}) doesn't match "
                f"population size ({len(self.population)})"
            )

        # Update scores
        for i, score in enumerate(scores):
            self.population[i].current_score = score
            self.population[i].history.append(score)
            if score > self.population[i].best_score:
                self.population[i].best_score = score

        # Track best ever
        max_score = max(scores)
        max_idx = scores.index(max_score)
        if max_score > self.best_ever_score:
            self.best_ever_score = max_score
            self.best_ever_hyperparams = self.population[max_idx].hyperparams.copy()

        self.generation_scores.append(max_score)
        self.generation += 1

        # Sort by score
        sorted_scores, sorted_indices = self.evaluate_population(scores)

        # Determine elite count
        elite_count = max(1, int(len(self.population) * self.elite_fraction))

        # Evolve non-elite members
        new_population = []
        for rank, idx in enumerate(sorted_indices):
            member = self.population[idx]

            if rank < elite_count:
                # Keep elite members as-is
                member.generation += 1
                new_population.append(member)
            else:
                # Mutate non-elite members
                mutated = self.mutate_hyperparams(member)

                # Periodically share best config
                if self.generation % self.sharing_interval == 0 and rank >= len(self.population) // 2:
                    best_member = self.population[sorted_indices[0]]
                    mutated = self.share_best_config(best_member, mutated)

                new_population.append(mutated)

        # Maintain population size
        while len(new_population) < self.population_size:
            best_member = self.population[sorted_indices[0]]
            new_member = AgentMember(
                member_id=len(new_population),
                hyperparams=best_member.hyperparams.copy(),
            )
            new_population.append(new_member)

        self.population = new_population[:self.population_size]
        return self.population

    def get_best_member(self) -> Optional[AgentMember]:
        """Get the best performing member in the current population.

        Returns:
            Best AgentMember or None if population is empty
        """
        if not self.population:
            return None
        return max(self.population, key=lambda m: m.best_score)

    def get_best_hyperparams(self) -> Dict[str, Any]:
        """Get the best hyperparameters found so far.

        Returns:
            Dict of hyperparameter names to values
        """
        if self.best_ever_hyperparams:
            return self.best_ever_hyperparams.copy()
        best = self.get_best_member()
        if best:
            return best.hyperparams.copy()
        return {}

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the PBT process."""
        avg_score = (
            sum(m.current_score for m in self.population) / len(self.population)
            if self.population else 0.0
        )
        return {
            "generation": self.generation,
            "population_size": self.population_size,
            "best_ever_score": self.best_ever_score,
            "best_ever_hyperparams": self.best_ever_hyperparams,
            "avg_current_score": avg_score,
            "generation_scores": self.generation_scores[-100:],
            "mutation_rate": self.mutation_rate,
            "elite_fraction": self.elite_fraction,
        }

    def save_population(self, path: str) -> None:
        """Save the population state to a directory.

        Args:
            path: Directory path to save population state
        """
        os.makedirs(path, exist_ok=True)

        # Save each member
        for member in self.population:
            member_path = os.path.join(path, f"member_{member.member_id}.json")
            with open(member_path, "w") as f:
                import json
                json.dump(member.to_dict(), f, indent=2)

        # Save population stats
        stats_path = os.path.join(path, "pbt_stats.json")
        with open(stats_path, "w") as f:
            import json
            json.dump(self.get_stats(), f, indent=2)

        logger.info(f"Saved PBT population to {path}")

    def load_population(self, path: str) -> List[AgentMember]:
        """Load the population state from a directory.

        Args:
            path: Directory path to load population state from

        Returns:
            List of AgentMember instances
        """
        self.population = []

        for i in range(self.population_size):
            member_path = os.path.join(path, f"member_{i}.json")
            if os.path.exists(member_path):
                with open(member_path, "r") as f:
                    import json
                    data = json.load(f)
                member = AgentMember(
                    member_id=data["member_id"],
                    hyperparams=data["hyperparams"],
                )
                member.best_score = data.get("best_score", float('-inf'))
                member.current_score = data.get("current_score", 0.0)
                member.generation = data.get("generation", 0)
                member.history = data.get("history", [])
                self.population.append(member)

        logger.info(f"Loaded PBT population from {path}")
        return self.population
