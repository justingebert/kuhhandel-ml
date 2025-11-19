import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gameengine import Game, AnimalType


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


def simple_auction_demo():
    """Demonstrate a simple auction."""
    print("\n🎮 SIMPLE AUCTION DEMO")
    print("Starting a 3-player game...\n")

    game = Game(num_players=3, seed=42)
    game.setup()

    print_game_state(game)

    # Player 0 starts an auction
    print("\n📢 Player 0 starts an auction...")
    animal = game.start_auction()
    print(f"Animal up for auction: {animal.animal_type.display_name}")
    print_game_state(game)

    # Player 1 bids 10
    print("\n💰 Player 1 bids 10...")
    success = game.place_bid(1, 10)
    print(f"Bid successful: {success}")

    # Player 2 bids 20
    print("\n💰 Player 2 bids 20...")
    success = game.place_bid(2, 20)
    print(f"Bid successful: {success}")

    # Auctioneer (Player 0) passes
    print("\n✋ Player 0 (auctioneer) passes...")
    game.auctioneer_passes()

    print_game_state(game)

    return game


def cow_trade_demo():
    """Demonstrate a cow trade."""
    print("\n\n🐮 COW TRADE DEMO")
    print("Setting up a scenario where players can trade...\n")

    game = Game(num_players=3, seed=123)
    game.setup()

    # Give both player 0 and player 1 a cow
    from gameengine import AnimalCard
    cow_card_1 = AnimalCard(AnimalType.COW, 100)
    cow_card_2 = AnimalCard(AnimalType.COW, 101)

    game.players[0].add_animal(cow_card_1)
    game.players[1].add_animal(cow_card_2)

    print_game_state(game)

    # Player 0 initiates a cow trade
    print("\n🤝 Player 0 initiates a cow trade with Player 1 for Cows...")
    offer_cards = [game.players[0].money[0], game.players[0].money[1]]  # Offer some money
    print(f"Offering: {[str(c) for c in offer_cards]}")

    success = game.start_cow_trade(1, AnimalType.COW, offer_cards)
    print(f"Trade initiated: {success}")

    print_game_state(game)

    # Player 1 counters with a higher offer
    print("\n💪 Player 1 makes a counter offer...")
    counter_cards = [game.players[1].money[0], game.players[1].money[1], game.players[1].money[2]]
    print(f"Counter offer: {[str(c) for c in counter_cards]}")

    game.counter_trade_offer(counter_cards)

    print_game_state(game)

    return game


def full_game_simulation(seed=999):
    """Simulate a full game with simple Bot."""
    print("\n\n🎲 FULL GAME SIMULATION")
    print(f"Running a complete game with simple AI players (seed={seed})...\n")

    import random
    from gameengine import ActionType, GamePhase

    # Use the game seed for Bot decisions too
    game = Game(num_players=5, seed=seed)
    game.setup()
    random.seed(seed)  # Seed for Bot decisions

    turn_count = 0
    max_turns = 200  # Safety limit

    while not game.is_game_over() and turn_count < max_turns:
        turn_count += 1

        if turn_count % 10 == 1:  # Print every 10 turns
            print(f"\nTurn {turn_count}")
            print_game_state(game)

        # In cow trade phase, check valid actions for target player, not initiator
        if game.phase == GamePhase.COW_TRADE:
            valid_actions = game.get_valid_actions(game.trade_target)
        else:
            valid_actions = game.get_valid_actions()

        if not valid_actions:
            # If player has no valid actions in PLAYER_TURN phase, skip their turn
            if game.phase == GamePhase.PLAYER_TURN:
                game.current_player_idx = (game.current_player_idx + 1) % game.num_players
                game.round_number += 1
                continue
            else:
                print(f"No valid actions in {game.phase.value} phase - game stuck")
                break

        # Simple Bot: Always start auction if possible, otherwise try cow trade

        if game.phase == GamePhase.PLAYER_TURN:
            if ActionType.START_AUCTION in valid_actions:
                animal = game.start_auction()
                if turn_count % 10 == 1:
                    print(f"  Player {game.current_player_idx} auctions {animal.animal_type.display_name}")

            elif ActionType.START_COW_TRADE in valid_actions:
                # Random chance to do a cow trade instead of waiting
                if random.random() < 0.5:  # 50% chance to initiate trade
                    # Find a valid cow trade
                    current_player = game.players[game.current_player_idx]
                    trade_started = False
                    for animal_type in AnimalType.get_all_types():
                        if current_player.has_animal_type(animal_type):
                            for target_id, target in enumerate(game.players):
                                if target_id != game.current_player_idx:
                                    if target.has_animal_type(animal_type) and not target.has_complete_set(animal_type):
                                        # Random offer amount (1-4 cards) - but only if player has money
                                        if len(current_player.money) > 0:
                                            num_cards = random.randint(1, min(4, len(current_player.money)))
                                            offer = current_player.money[:num_cards]
                                            if offer:
                                                success = game.start_cow_trade(target_id, animal_type, offer)
                                                if success:
                                                    trade_started = True
                                                    break
                            if trade_started:
                                break

        elif game.phase == GamePhase.AUCTION:
            # Handle auction phase - players bid or auctioneer decides
            current_player_id = game.current_player_idx

            # Other players bid with some randomness
            has_bidders = False
            for player_id in range(game.num_players):
                if player_id != current_player_id:
                    player = game.players[player_id]
                    # Random chance to bid, and random bid amount
                    if player.get_total_money() >= game.auction_high_bid + 10:
                        if random.random() < 0.7:  # 70% chance to bid
                            bid_amount = game.auction_high_bid + random.choice([10, 20, 30])
                            if player.get_total_money() >= bid_amount:
                                success = game.place_bid(player_id, bid_amount)
                                if success:
                                    has_bidders = True

            # Auctioneer decides with randomness
            auctioneer = game.players[current_player_id]
            # Random threshold between 10 and 30
            threshold = random.randint(10, 30)
            if game.auction_high_bid <= threshold and auctioneer.get_total_money() >= game.auction_high_bid:
                if random.random() < 0.6:  # 60% chance to buy
                    try:
                        game.auctioneer_buys()
                    except:
                        game.auctioneer_passes()
                else:
                    game.auctioneer_passes()
            else:
                game.auctioneer_passes()

        elif game.phase == GamePhase.COW_TRADE:
            # Handle cow trade response
            target = game.players[game.trade_target]

            # Target makes a decision with randomness
            offer_value = sum(card.value for card in game.trade_offer)

            # Random chance to accept vs counter
            if random.random() < 0.3:  # 30% chance to just accept
                game.accept_trade_offer()
            elif target.get_total_money() > offer_value + 10:
                # Counter with random amount (1-5 cards)
                num_cards = random.randint(1, min(5, len(target.money)))
                counter = target.money[:num_cards]
                if counter and sum(card.value for card in counter) > offer_value:
                    game.counter_trade_offer(counter)
                else:
                    game.accept_trade_offer()
            else:
                game.accept_trade_offer()

    print("\n\n🏆 GAME OVER!")
    print_game_state(game)

    scores = game.get_scores()
    print("\nFinal Scores:")
    for player_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        player = game.players[player_id]
        animal_counts = player.get_animal_counts()
        print(f"  {player.name}: {score} points")
        if animal_counts:
            print(f"    Animals: {', '.join(f'{at.display_name}({count})' for at, count in animal_counts.items())}")

    winner = game.get_winner()
    scores = game.get_scores()
    max_score = max(scores.values()) if scores else 0
    winners = [game.players[pid] for pid, score in scores.items() if score == max_score]

    if len(winners) > 1:
        print(f"\n👑 It's a tie between {', '.join(w.name for w in winners)} with {max_score} points!")
    elif winners:
        print(f"\n👑 Winner: {winners[0].name} with {max_score} points!")
    else:
        print(f"\n👑 No winner - all players have 0 points!")

    print(f"\nTotal turns: {turn_count}")
    print(f"Total actions logged: {len(game.action_history)}")

    return game


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run You\'re Bluffing demo game')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for game (default: 999)')
    parser.add_argument('--skip-demos', action='store_true', help='Skip simple demos and only run full game')
    args = parser.parse_args()

    print("="*60)
    print("YOU'RE BLUFFING - GAME ENGINE TEST")
    print("="*60)

    if not args.skip_demos:
        # Run demos
        game1 = simple_auction_demo()
        game2 = cow_trade_demo()

    game3 = full_game_simulation(seed=args.seed)

    print("\n\n✅ All demos completed successfully!")
    print("\nThe game engine is ready for ML training!")

