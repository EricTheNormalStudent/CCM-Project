from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from env import Connect4Env


def terminal_value(env: Connect4Env, player: str) -> float:
    """Return terminal outcome from a specific player's perspective."""
    winner = env.winner
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


@dataclass
class Node:
    state: Connect4Env
    parent: Optional["Node"] = None
    action_taken: Optional[int] = None
    visits: int = 0
    value_sum: float = 0.0
    children: Dict[int, "Node"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.player_to_move = self.state.current_player()
        self.untried_actions = self.state.legal_actions()

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def expand(self, rng) -> "Node":
        action = rng.choice(self.untried_actions)
        self.untried_actions.remove(action)

        next_state = self.state.clone()
        next_state.step(action)

        child = Node(
            state=next_state,
            parent=self,
            action_taken=action,
        )
        self.children[action] = child
        return child

    def best_child(self, c_puct: float) -> "Node":
        best_score = -float("inf")
        best_node = None

        for child in self.children.values():
            exploit = -child.q_value()
            explore = c_puct * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit + explore

            if score > best_score:
                best_score = score
                best_node = child

        if best_node is None:
            raise ValueError("best_child called with no children.")
        return best_node


class MCTS:
    def __init__(
        self,
        num_simulations: int = 500,
        exploration_constant: float = 1.4,
        rollout_limit: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.rollout_limit = rollout_limit
        self.rng = random.Random(seed)

    def search(self, root_state: Connect4Env) -> Dict[str, object]:
        root = Node(state=root_state.clone())

        if root.is_terminal():
            return {
                "action": None,
                "root": root,
                "visit_counts": {},
                "action_probs": {},
            }

        for _ in range(self.num_simulations):
            node = root

            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_constant)

            if not node.is_terminal() and node.untried_actions:
                node = node.expand(self.rng)

            value = self.rollout(node.state.clone(), node.player_to_move)
            self.backpropagate(node, value)
        
        
        visit_counts = {action: child.visits for action, child in root.children.items()}
        total_visits = sum(visit_counts.values())
        action_probs = {
            action: visits / total_visits for action, visits in visit_counts.items()
        }

        best_action = max(visit_counts, key=visit_counts.get)
        return {
            "action": best_action,
            "root": root,
            "visit_counts": visit_counts,
            "action_probs": action_probs,
        }

    def rollout(self, state: Connect4Env, rollout_player: Optional[str]) -> float:
        if rollout_player is None:
            return 0.0

        steps = 0
        while not state.is_terminal():
            legal_actions = state.legal_actions()
            action = self.rng.choice(legal_actions)
            state.step(action)

            steps += 1
            if self.rollout_limit is not None and steps >= self.rollout_limit:
                break

        if state.is_terminal():
            return terminal_value(state, rollout_player)
        return 0.0

    def backpropagate(self, node: Node, value: float) -> None:
        while node is not None:
            node.visits += 1
            node.value_sum += value
            value = -value
            node = node.parent


def select_action(
    env: Connect4Env,
    num_simulations: int = 500,
    exploration_constant: float = 1.4,
    seed: Optional[int] = None,
) -> int:
    """Convenience helper: run MCTS once and return the chosen action."""
    searcher = MCTS(
        num_simulations=num_simulations,
        exploration_constant=exploration_constant,
        seed=seed,
    )
    result = searcher.search(env)
    action = result["action"]
    if action is None:
        raise ValueError("Cannot select an action from a terminal state.")
    return action
