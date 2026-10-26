"""Continuous space agents."""

from __future__ import annotations

from itertools import compress
from typing import Protocol

import numpy as np

from mesa.agent import Agent
from mesa.experimental.continuous_space import ContinuousSpace


class HasPositionProtocol(Protocol):
    """Protocol for continuous space position holders."""

    position: np.ndarray


class ContinuousSpaceAgent(Agent):
    """A continuous space agent.

    Attributes:
        space (ContinuousSpace): the continuous space in which the agent is located
        position (np.ndarray): the position of the agent

    """

    __slots__ = ["_mesa_index", "space"]

    @property
    def position(self) -> np.ndarray:
        """Position of the agent."""
        return self.space.agent_positions[self._mesa_index]

    @position.setter
    def position(self, value: np.ndarray) -> None:
        if not self.space.in_bounds(value):
            if self.space.torus:
                value = self.space.torus_correct(value)
            else:
                raise ValueError(f"point {value} is outside the bounds of the space")

        self.space.agent_positions[self._mesa_index] = value

    def __init__(self, space: ContinuousSpace, model):
        """Initialize a continuous space agent.

        Args:
            space: the continuous space in which the agent is located
            model: the model to which the agent belongs

        """
        super().__init__(model)
        self.space: ContinuousSpace = space
        self.space._add_agent(self)
        # self.position[:] = np.nan

    def remove(self) -> None:
        """Remove and delete the agent from the model and continuous space."""
        super().remove()
        self.space._remove_agent(self)
        self._mesa_index = None
        self.space = None

    def get_neighbors_in_radius(
        self, radius: float | int = 1
    ) -> tuple[list[ContinuousSpaceAgent], np.ndarray]:
        """Get neighbors within radius.

        Args:
            radius: radius within which to look for neighbors

        """
        agents, dists = self.space.get_agents_in_radius(self.position, radius=radius)
        logical = np.asarray([agent is not self for agent in agents])
        agents = list(compress(agents, logical))
        return agents, dists[logical]

    def get_nearest_neighbors(
        self, k: int = 1
    ) -> tuple[list[ContinuousSpaceAgent], np.ndarray]:
        """Get neighbors within radius.

        Args:
            k: the number of nearest neighbors to return

        """
        dists, agents = self.space.calculate_distances(self.position)
        # Leave self out before selecting the k nearest. Requesting k + 1 and
        # filtering afterwards fails when more than k other agents share this
        # agent's position, because self may then not be among the k + 1.
        index = self._mesa_index
        dists = np.delete(dists, index)
        agents = agents[:index] + agents[index + 1 :]
        return self.space._select_k_nearest(dists, agents, k)
