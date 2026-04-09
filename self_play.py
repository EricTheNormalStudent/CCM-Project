from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from env import COLUMNS, raw_env
from mcts import MCTS


def state_to_tensor(game) -> np.ndarray:
    """Encode the board from the current player's perspective."""
    current_agent = game.current_player()
    observation = game.observe(current_agent)["observation"]
    return observation.astype(np.int8)


def policy_dict_to_array(action_probs: Dict[int, float]) -> np.ndarray:
    policy = np.zeros(COLUMNS, dtype=np.float32)
    for action, prob in action_probs.items():
        policy[action] = prob
    return policy


def play_self_play_game(
    num_simulations: int = 200,
    exploration_constant: float = 1.4,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Play one MCTS-vs-MCTS game and return training examples."""
    game = raw_env(render_mode="human")
    game.reset()

    searcher = MCTS(
        num_simulations=num_simulations,
        exploration_constant=exploration_constant,
        seed=seed,
    )

    trajectory: List[Dict[str, object]] = []

    while not game.is_terminal():
        current_agent = game.current_player()
        search_result = searcher.search(game)
        action = search_result["action"]
        action_probs = search_result["action_probs"]

        example = {
            "state": state_to_tensor(game),
            "player": current_agent,
            "policy": policy_dict_to_array(action_probs),
            "action": action,
            "value": 0.0,
        }
        trajectory.append(example)

        game.step(action)

    winner = game.winner
    for example in trajectory:
        if winner is None:
            example["value"] = 0.0
        elif example["player"] == winner:
            example["value"] = 1.0
        else:
            example["value"] = -1.0

    return trajectory


def generate_self_play_data(
    num_games: int,
    num_simulations: int = 200,
    exploration_constant: float = 1.4,
    seed: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Generate a dataset by running multiple self-play games."""
    rng = np.random.default_rng(seed)
    dataset: List[Dict[str, object]] = []

    for _ in range(num_games):
        game_seed = int(rng.integers(0, 2**31 - 1)) if seed is not None else None
        dataset.extend(
            play_self_play_game(
                num_simulations=num_simulations,
                exploration_constant=exploration_constant,
                seed=game_seed,
            )
        )

    return dataset


def save_self_play_data(data: List[Dict[str, object]], output_path: str) -> None:
    """Save self-play examples to a compressed NumPy archive."""
    if not data:
        raise ValueError("No self-play data to save.")

    states = np.stack([example["state"] for example in data]).astype(np.int8)
    policies = np.stack([example["policy"] for example in data]).astype(np.float32)
    values = np.array([example["value"] for example in data], dtype=np.float32)
    actions = np.array([example["action"] for example in data], dtype=np.int64)
    players = np.array([example["player"] for example in data], dtype="<U16")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=states,
        policies=policies,
        values=values,
        actions=actions,
        players=players,
    )


if __name__ == "__main__":
    data = generate_self_play_data(num_games=2, num_simulations=100, seed=0)
    save_self_play_data(data, "self_play_data.npz")
    print(f"Generated {len(data)} training examples.")
    print("Saved dataset to self_play_data.npz")
