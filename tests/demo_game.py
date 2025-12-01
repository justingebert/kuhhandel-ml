import os
import sys
import random
from typing import List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gameengine import Game, AnimalType, ActionType, GamePhase
from gameengine.controller import GameController
from gameengine.agent import Agent
from gameengine.actions import GameAction


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


class RandomAgent(Agent):
    """A simple agent that makes random valid moves."""

    def get_action(self, game: Game, valid_actions: List[ActionType]) -> GameAction:
        from gameengine.actions import Actions
        
        player = game.players[self.player_id]
        
        # 1. Auction Bidding
        if ActionType.BID in valid_actions:
            min_bid = game.auction_high_bid + 10
            # Bid if rich enough and random chance
            if player.get_total_money() >= min_bid and random.random() < 0.5:
                return Actions.bid(amount=min_bid)
            return Actions.pass_action()

        # 2. Auctioneer Decision
        if ActionType.BUY_AS_AUCTIONEER in valid_actions:
            # Buy if cheap
            if game.auction_high_bid < 50 and player.get_total_money() >= game.auction_high_bid:
                return Actions.buy_as_auctioneer()
            return Actions.pass_action()

        # 3. Cow Trade Response
        if ActionType.COUNTER_OFFER in valid_actions:
            # Counter if rich enough
            offer_value = sum(c.value for c in game.trade_offer)
            if player.get_total_money() > offer_value + 10 and random.random() < 0.5:
                # Pick random cards
                num_cards = random.randint(1, min(4, len(player.money)))
                return Actions.counter_offer(money_cards=player.money[:num_cards])
            return Actions.accept_offer()

        # 4. Main Turn Decision
        if ActionType.START_AUCTION in valid_actions and ActionType.START_COW_TRADE in valid_actions:
             # Prefer auction unless deck empty
             if not game.animal_deck:
                 action_type = ActionType.START_COW_TRADE
             else:
                 action_type = random.choice([ActionType.START_AUCTION, ActionType.START_COW_TRADE])
        elif ActionType.START_AUCTION in valid_actions:
            action_type = ActionType.START_AUCTION
        elif ActionType.START_COW_TRADE in valid_actions:
            action_type = ActionType.START_COW_TRADE
        else:
            return Actions.pass_action()

        if action_type == ActionType.START_AUCTION:
            return Actions.start_auction()
        
        if action_type == ActionType.START_COW_TRADE:
            # Find a valid trade
            for animal_type in AnimalType.get_all_types():
                if player.has_animal_type(animal_type) and not player.has_complete_set(animal_type):
                    for target_id, target in enumerate(game.players):
                        if target_id != self.player_id and target.has_animal_type(animal_type):
                            # Can trade even with 0 money (bluffing)
                            if len(player.money) > 0:
                                num_cards = random.randint(1, min(4, len(player.money)))
                                money_cards = player.money[:num_cards]
                            else:
                                money_cards = []
                            
                            return Actions.start_cow_trade(
                                target_id=target_id,
                                animal_type=animal_type,
                                money_cards=money_cards
                            )
            
            return Actions.pass_action()

        return Actions.pass_action()


def full_game_simulation(seed=999, players=5):
    """Simulate a full game using GameController."""
    print("\n\n🎲 FULL GAME SIMULATION (Controller)")
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
