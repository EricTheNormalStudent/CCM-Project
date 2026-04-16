from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


ROWS = 6
COLUMNS = 7
PLAYER_VALUES = {
    "player_0": 1,
    "player_1": -1,
}


@dataclass
class MoveRecord:
    board_before: np.ndarray
    board_after: np.ndarray
    player: str
    action: int
    value: float
    move_number: int


def to_absolute_board(state: np.ndarray, player: str) -> np.ndarray:
    return state.copy() if player == "player_0" else -state


def drop_piece(board: np.ndarray, action: int, player: str) -> np.ndarray:
    next_board = board.copy()
    piece = PLAYER_VALUES[player]
    for row in range(ROWS - 1, -1, -1):
        if next_board[row, action] == 0:
            next_board[row, action] = piece
            return next_board
    raise ValueError(f"Column {action} is full.")


def split_games(states: np.ndarray, players: np.ndarray, actions: np.ndarray, values: np.ndarray) -> List[List[MoveRecord]]:
    games: List[List[MoveRecord]] = []
    current_game: List[MoveRecord] = []

    for index, (state, player, action, value) in enumerate(zip(states, players, actions, values), start=1):
        absolute_board = to_absolute_board(state, player)
        if np.count_nonzero(absolute_board) == 0 and current_game:
            games.append(current_game)
            current_game = []

        current_game.append(
            MoveRecord(
                board_before=absolute_board,
                board_after=drop_piece(absolute_board, int(action), str(player)),
                player=str(player),
                action=int(action),
                value=float(value),
                move_number=index,
            )
        )

    if current_game:
        games.append(current_game)

    return games


def draw_board(ax: plt.Axes, board: np.ndarray, title: str, action: int | None = None) -> None:
    ax.clear()

    grid = np.zeros((ROWS, COLUMNS, 3), dtype=float)
    grid[:] = np.array([0.08, 0.24, 0.65])
    ax.imshow(grid, extent=(-0.5, COLUMNS - 0.5, ROWS - 0.5, -0.5))

    for row in range(ROWS):
        for col in range(COLUMNS):
            value = board[row, col]
            if value == 1:
                color = "#d62728"
            elif value == -1:
                color = "#f2c744"
            else:
                color = "white"
            circle = plt.Circle((col, row), 0.38, color=color, ec="black", lw=1.5)
            ax.add_patch(circle)

    if action is not None:
        ax.add_patch(plt.Rectangle((action - 0.48, -0.48), 0.96, ROWS - 0.04, fill=False, ec="#00d4ff", lw=3))

    ax.set_xlim(-0.5, COLUMNS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLUMNS))
    ax.set_yticks(range(ROWS))
    ax.set_title(title)
    ax.set_aspect("equal")


def animate_game(game: List[MoveRecord], game_index: int, interval_ms: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.5))

    frames: List[tuple[np.ndarray, str, int | None]] = []
    opening = game[0].board_before
    frames.append((opening, f"Game {game_index + 1}: start", None))

    for ply, move in enumerate(game, start=1):
        title = (
            f"Game {game_index + 1} | Move {ply}\n"
            f"{move.player} played column {move.action} | final value for mover: {move.value:+.1f}"
        )
        frames.append((move.board_after, title, move.action))

    def update(frame_index: int) -> None:
        board, title, action = frames[frame_index]
        draw_board(ax, board, title, action)

    animation = FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, repeat=False)
    fig._animation = animation

    paused = False

    def on_key_press(event) -> None:
        nonlocal paused

        if event.key == " ":
            if paused:
                animation.event_source.start()
                paused = False
                fig.suptitle("")
            else:
                animation.event_source.stop()
                paused = True
                fig.suptitle("Paused - press Space to resume", fontsize=12)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay self-play Connect4 games from self_play_data.npz.")
    parser.add_argument("--file", default="self_play_data.npz", help="Path to the self-play dataset (.npz).")
    parser.add_argument("--game", type=int, default=1, help="1-based game number to replay.")
    parser.add_argument("--interval", type=int, default=900, help="Milliseconds between frames.")
    parser.add_argument("--list", action="store_true", help="List the available games and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        games = split_games(data["states"], data["players"], data["actions"], data["values"])

    if not games:
        raise ValueError("No games found in dataset.")

    if args.list:
        for index, game in enumerate(games, start=1):
            print(f"game {index}: {len(game)} moves")
        return

    game_index = args.game - 1
    if not 0 <= game_index < len(games):
        raise IndexError(f"Game {args.game} out of range. Found {len(games)} games.")

    animate_game(games[game_index], game_index, args.interval)


if __name__ == "__main__":
    main()
