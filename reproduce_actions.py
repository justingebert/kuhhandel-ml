import sys
import os

# Add current directory to path so we can import gameengine
sys.path.append(os.getcwd())

from gameengine.game import Game, GamePhase, ActionType, calculate_total_value
from gameengine.actions import *

def print_actions(actions):
    print(f"Found {len(actions)} actions:")
    for i, action in enumerate(actions[:15]): # Limit output
        print(f"  {i}: {action}")
        if isinstance(action, BidAction):
             print(f"     Bid Amount: {calculate_total_value(action.money_cards)}")
    if len(actions) > 15:
        print(f"  ... and {len(actions) - 15} more")

def test_start_actions():
    print("\n--- Testing Start Actions (Player Turn) ---")
    g = Game(num_players=3, seed=42)
    g.setup()
    
    # Init some animals for cow trade potential
    g.players[0].add_animal(g.animal_deck[0]) # Give P0 an animal
    g.players[1].add_animal(g.animal_deck[0]) # Give P1 same animal type (different card)
    
    # Give P0 some zeros for bluffing test
    # P0 starts with 2 zeros usually (from MoneyDeck default)
    # Check valid actions for P0
    print("Player 0 Money:", g.players[0].money)
    actions = g.get_valid_actions(0)
    print_actions(actions)

def test_auction_actions():
    print("\n--- Testing Auction Actions ---")
    g = Game(num_players=3, seed=42)
    g.setup()
    g.start_auction() # Starts auction, P0 turn
    
    # Check valid actions for P1 (bidder)
    print("Player 1 (Bidder) Actions (Money: {}):".format(g.players[1].get_total_money()))
    actions = g.get_valid_actions(1)
    print_actions(actions)
    
    # Place a bid to change state
    # Need to simulate bid. Controller would parse Action and call place_bid.
    # Game.place_bid still takes int.
    g.place_bid(1, 10)
    
    print("Player 2 (Bidder) Actions after P1 bids 10 (Money: {}):".format(g.players[2].get_total_money()))
    actions = g.get_valid_actions(2)
    print_actions(actions)

def test_cow_trade_actions():
    print("\n--- Testing Cow Trade Actions (Response) ---")
    g = Game(num_players=3, seed=42)
    g.setup()
    
    # Force cow trade
    # P0 starts trade with P1
    # Get a specific animal type
    animal = g.animal_deck[0]
    animal_type = animal.animal_type
    
    # Ensure P1 has the SAME type (search in deck)
    animal2 = next(c for c in g.animal_deck if c.animal_type == animal_type and c != animal)
    
    g.players[0].add_animal(animal)
    g.players[1].add_animal(animal2)

    # Valid actions to start trade
    # P0 starts trade
    offer = [g.players[0].money[0]] # 0 value
    result = g.start_cow_trade(1, animal.animal_type, offer)
    print(f"Trade Started: {result}")
    
    # Check actions for P1 (Target)
    print("Player 1 (Target) Money:", g.players[1].money)
    actions = g.get_valid_actions(1)
    print_actions(actions)

if __name__ == "__main__":
    test_start_actions()
    test_auction_actions()
    test_cow_trade_actions()
