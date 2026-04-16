from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np


from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector



ROWS = 6
COLUMNS = 7
EMPTY = 0
PLAYER_0 = 1
PLAYER_1 = -1


def env(render_mode: Optional[str] = None):
    """Standard PettingZoo factory with basic wrappers."""
    base_env = raw_env(render_mode=render_mode)

    # illegal move
    base_env = wrappers.TerminateIllegalWrapper(base_env, illegal_reward=-1)
    # invalid columns
    base_env = wrappers.AssertOutOfBoundsWrapper(base_env)
    # turn order
    base_env = wrappers.OrderEnforcingWrapper(base_env)

    return base_env


def raw_env(render_mode: Optional[str] = None):
    return Connect4Env(render_mode=render_mode)


class Connect4Env(AECEnv):
    metadata = {
        "name": "connect4_v0",
        # human observation
        "render_modes": ["human"],
        # take turns
        "is_parallelizable": False,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        self.possible_agents = ["player_0", "player_1"]
        self.agent_name_mapping = {
            "player_0": PLAYER_0,
            "player_1": PLAYER_1,
        }

        # action ∈ {0,1,2,3,4,5,6}
        self._action_spaces = {
            agent: spaces.Discrete(COLUMNS) for agent in self.possible_agents
        }

        # 
        self._observation_spaces = {
            agent: spaces.Dict(
                {
                    # what game looks like
                    "observation": spaces.Box(
                        low=-1,
                        high=1,
                        shape=(ROWS, COLUMNS),
                        dtype=np.int8,
                    ),

                    # what moves are allowed
                    "action_mask": spaces.Box(
                        low=0,
                        high=1,
                        shape=(COLUMNS,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent in self.possible_agents
        }

        # game board
        self.board = np.zeros((ROWS, COLUMNS), dtype=np.int8)

        # player 0 and player 1
        self.agents: List[str] = []

        # Rewards
        self.rewards: Dict[str, float] = {}
        self._cumulative_rewards: Dict[str, float] = {}

        # {"player_0": True, "player_1": True}
        self.terminations: Dict[str, bool] = {}
        
        # Optional
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}

        # Tracks whose turn it is
        self.agent_selection = ""
        self._agent_selector = None

        # Winner
        self.winner: Optional[str] = None

    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            np.random.seed(seed)

        self.board = np.zeros((ROWS, COLUMNS), dtype=np.int8)
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.winner = None

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

    def observe(self, agent: str):
        player_value = self.agent_name_mapping[agent]
        observation = (self.board * player_value).astype(np.int8)
        return {
            "observation": observation,
            "action_mask": self._action_mask(),
        }

    def step(self, action: Optional[int]):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        player_value = self.agent_name_mapping[agent]

        self._cumulative_rewards[agent] = 0.0

        row = self._drop_row(action)
        self.board[row, action] = player_value

        # win
        if self._has_connect_four(row, action, player_value):
            self.winner = agent
            self.rewards = {
                winner_agent: 1.0 if winner_agent == agent else -1.0
                for winner_agent in self.agents
            }
            self.terminations = {name: True for name in self.agents}
        
        # draw
        elif not self.legal_actions():
            self.rewards = {name: 0.0 for name in self.agents}
            self.terminations = {name: True for name in self.agents}

        # continue
        else:
            self.rewards = {name: 0.0 for name in self.agents}

        self._accumulate_rewards()

        # switch player
        self.agent_selection = self._agent_selector.next()

        if self.render_mode == "human":
            self.render()

    def render(self):
        symbols = {
            EMPTY: ".",
            PLAYER_0: "X",
            PLAYER_1: "O",
        }
        print("\n".join(" ".join(symbols[cell] for cell in row) for row in self.board))
        print("0 1 2 3 4 5 6")

    def close(self):
        pass

    def clone(self) -> "Connect4Env":
        return copy.deepcopy(self)

    def current_player(self) -> Optional[str]:
        if not self.agents:
            return None
        return self.agent_selection

    def legal_actions(self) -> List[int]:
        return [col for col in range(COLUMNS) if self.board[0, col] == EMPTY]

    def is_terminal(self) -> bool:
        return bool(self.agents) and all(self.terminations.values())

    def winner_value(self) -> int:
        if self.winner is None:
            return 0
        return self.agent_name_mapping[self.winner]

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(COLUMNS, dtype=np.int8)
        mask[self.legal_actions()] = 1
        return mask

    def _drop_row(self, action: int) -> int:
        if action is None or not 0 <= action < COLUMNS:
            raise ValueError(f"Invalid action: {action}")

        for row in range(ROWS - 1, -1, -1):
            if self.board[row, action] == EMPTY:
                return row

        raise ValueError(f"Column {action} is full.")

    def _has_connect_four(self, row: int, col: int, player_value: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for delta_row, delta_col in directions:
            count = 1
            count += self._count_direction(row, col, delta_row, delta_col, player_value)
            count += self._count_direction(row, col, -delta_row, -delta_col, player_value)
            if count >= 4:
                return True
        return False

    def _count_direction(
        self,
        row: int,
        col: int,
        delta_row: int,
        delta_col: int,
        player_value: int,
    ) -> int:
        count = 0
        row += delta_row
        col += delta_col
        while 0 <= row < ROWS and 0 <= col < COLUMNS:
            if self.board[row, col] != player_value:
                break
            count += 1
            row += delta_row
            col += delta_col
        return count
