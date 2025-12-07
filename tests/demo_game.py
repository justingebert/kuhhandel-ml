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

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        from gameengine.actions import ActionType
        
        player = game.players[self.player_id]
        
        # Helper to group actions by type
        actions_by_type = {}
        for action in valid_actions:
            if action.type not in actions_by_type:
                actions_by_type[action.type] = []
            actions_by_type[action.type].append(action)

        # 1. Auction Bidding
        # If we can bid, should we?
        if ActionType.BID in actions_by_type:
            # Strategies:
            bids = actions_by_type[ActionType.BID]
            # Find minimum bid
            # Bids are granular now, we valid_actions contains all valid bids with cards.
            # We just need to pick one.
            # Sort by amount (value of money cards)
            # Since BidAction has .money_cards, we need to calculate total.
            from gameengine.Money import calculate_total_value
            
            # Group bids by value to find the cheapest ones
            # (There might be multiple ways to pay the same amount, just pick one)
            bids.sort(key=lambda a: a)#calculate_total_value(a.money_cards))
            min_bid_action = bids[0]
            # min_bid_amount = calculate_total_value(min_bid_action.money_cards)
            
            # Simple logic: 50% chance to bid if we can
            if random.random() < 0.5:
                # Prefer the smallest bid (first in sorted list)
                return min_bid_action
            
            # Else try to pass
            if ActionType.PASS in actions_by_type:
                return actions_by_type[ActionType.PASS][0]

        # 2. Auctioneer Decision
        if ActionType.BUY_AS_AUCTIONEER in actions_by_type:
            buy_action = actions_by_type[ActionType.BUY_AS_AUCTIONEER][0]
            # Buy if cheap (e.g. < 50) and we have money (implicit if action is valid? No, action valid doesn't mean smart)
            # Actually buy_action is only valid if we CAN pay.
            if game.auction_high_bid < 50:
                 return buy_action
            # Else pass (sell)
            if ActionType.PASS in actions_by_type:
                return actions_by_type[ActionType.PASS][0]

        # 3. Cow Trade Response (Counter vs Accept)
        if ActionType.COUNTER_OFFER in actions_by_type:
            # 50% chance to counter implies we prefer counter over accept
            if random.random() < 0.5:
                # Pick a random counter offer (includes bluffs!)
                return random.choice(actions_by_type[ActionType.COUNTER_OFFER])
            # Else accept
            if ActionType.ACCEPT_OFFER in actions_by_type:
                 return actions_by_type[ActionType.ACCEPT_OFFER][0]

        # 4. Main Turn (Start Auction vs Trade)
        possible_types = []
        if ActionType.START_AUCTION in actions_by_type:
            possible_types.append(ActionType.START_AUCTION)
        if ActionType.START_COW_TRADE in actions_by_type:
            possible_types.append(ActionType.START_COW_TRADE)
            
        if not possible_types:
            # Should be pass if nothing else?
             if ActionType.PASS in actions_by_type:
                return actions_by_type[ActionType.PASS][0]
             # Fallback
             return valid_actions[0]

        # Decision
        chosen_type = None
        if ActionType.START_AUCTION in possible_types and ActionType.START_COW_TRADE in possible_types:
            if not game.animal_deck:
                chosen_type = ActionType.START_COW_TRADE
            else:
                chosen_type = random.choice(possible_types)
        elif possible_types:
            chosen_type = possible_types[0]
            
        if chosen_type:
            # Pick one action of that type
            # For auction, usually only one StartAuctionAction
            # For trade, MANY options.
            options = actions_by_type[chosen_type]
            return random.choice(options)

        return valid_actions[0]


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
