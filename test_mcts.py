from env import raw_env
from mcts import MCTS, select_action


def print_position(label, game):
    print(label)
    game.render()
    print("current player:", game.current_player())
    print("legal actions:", game.legal_actions())
    print()


def test_empty_board():
    game = raw_env()
    game.reset()

    print_position("Empty board", game)

    searcher = MCTS(num_simulations=200, seed=0)
    result = searcher.search(game)

    print("chosen action:", result["action"])
    print("visit counts:", result["visit_counts"])
    print("action probs:", result["action_probs"])
    print("-" * 40)


def test_immediate_win():
    game = raw_env()
    game.reset()

    moves = [0, 4, 1, 5, 2, 6]
    for move in moves:
        game.step(move)

    print_position("Immediate-win position", game)

    best_action = select_action(game, num_simulations=300, seed=1)
    print("best action from MCTS:", best_action)
    print("expected winning action: 3")

    if best_action == 3:
        print("PASS: MCTS found the immediate winning move.")
    else:
        print("FAIL: MCTS did not choose the immediate winning move.")


if __name__ == "__main__":
    test_empty_board()
    test_immediate_win()
