import os
import sys
import random

from rl.random_agent import RandomAgent

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gameengine import Game
from gameengine.controller import GameController


def print_game_state(game: Game):
    """Print the current game state."""
    print("\n" + "="*60)
    print(f"Round {game.round_number} - Phase: {game.phase.value}")
    print(f"Current Player: Player {game.current_player_idx}")
    print(f"Animals remaining in deck: {len(game.animal_deck)}")

    print("\nPlayers:")
    for player in game.players:
        print(f"  {player.name}:")
        print(f"    Money: {player.get_total_money()}")
        animal_counts = player.get_animal_counts()
        if animal_counts:
            print(f"    Animals: {', '.join(f'{at.display_name}({count})' for at, count in animal_counts.items())}")
        else:
            print(f"    Animals: None")
        print(f"    Score: {player.calculate_score()}")
    print("="*60)


def full_game_simulation(seed=999, players=3):
    """Simulate a full game using GameController."""
    print("\n\nFULL GAME SIMULATION (Controller)")
    print(f"Running a complete game with RandomAgents (seed={seed})...\n")

    game = Game(num_players=players, seed=seed)
    game.setup()
    
    # Create agents
    agents = [RandomAgent(f"Bot {i}") for i in range(players)]
    
    # Create controller
    controller = GameController(game, agents)
    
    # Run
    scores = controller.run(max_turns=500)
    
    print("\n\n GAME OVER!")
    print_game_state(game)
    
    print("\nFinal Scores:")
    for player_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  Player {player_id}: {score} points")

    return game


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run You\'re Bluffing demo game')
    parser.add_argument('--seed', type=int, default=random.random(), help='Random seed for game')
    parser.add_argument('--players', type=int, default=3, help='Number of players (1-5')
    args = parser.parse_args()

    full_game_simulation(seed=args.seed, players=args.players)
