from typing import List, Dict, Any
from gameengine.agent import Agent
from gameengine.game import Game, GamePhase
from gameengine.actions import GameAction, ActionType

class UserAgent(Agent):
    """An agent that allows a human user to play via the terminal."""

    def __init__(self, name: str):
        super().__init__(name)
        self.last_history_len = 0

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        print(f"\n{'='*40}")
        print(f"YOUR TURN ({self.name})")
        
        # Print what happened since last time
        self._print_event_log(game)
        
        self._print_game_info(game)

        # 1. Auto-select if only one action
        if len(valid_actions) == 1:
            action = valid_actions[0]
            print(f"\nOnly one action available. Automatically choosing: {self._format_action(action)}")
            return action

        # 2. Special Bidding UI
        if game.phase == GamePhase.AUCTION_BIDDING:
            return self._handle_auction_bidding(valid_actions, game)

        # Standard Action Selection
        print("\nAvailable Actions:")
        for i, action in enumerate(valid_actions):
            print(f"[{i}] {self._format_action(action)}")
            
        while True:
            choice = input(f"\nSelect action (0-{len(valid_actions)-1}) or 'animals'/'money': ").strip().lower()
            
            if choice == "animals":
                self._print_animals(game)
                continue
            if choice == "money":
                self._print_money(game)
                continue
                
            try:
                idx = int(choice)
                if 0 <= idx < len(valid_actions):
                    return valid_actions[idx]
                print("Invalid selection. Please choose a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number or command.")

    def _handle_auction_bidding(self, valid_actions: List[GameAction], game: Game) -> GameAction:
        # Separate pass and bid actions
        pass_action = None
        bid_actions = []
        
        for action in valid_actions:
            if action.type == ActionType.AUCTION_PASS:
                pass_action = action
            elif action.type == ActionType.AUCTION_BID:
                bid_actions.append(action)
        
        bid_actions.sort(key=lambda x: x.amount)
        min_bid = bid_actions[0].amount if bid_actions else 0
        min_bid_action = bid_actions[0] if bid_actions else None
        
        print("\nAuction Options:")
        print("[0] Pass")
        if min_bid_action:
            print(f"[1] Bid Minimum ({min_bid})")
            print(f"[2] Custom Bid")
        else:
            print("(!) Cannot bid (not enough money).")
        
        while True:
            choice = input("\nChoose (0/1/2) or 'animals'/'money': ").strip().lower()
            
            if choice == "animals":
                self._print_animals(game)
                continue
            if choice == "money":
                self._print_money(game)
                continue

            if choice == "0":
                if pass_action:
                    return pass_action
                else:
                    print("Error: Pass action not found (this shouldn't happen).")
                    return valid_actions[0] # Fallback
            
            elif choice == "1":
                if min_bid_action:
                    return min_bid_action
                else:
                    print("You cannot bid.")
            
            elif choice == "2":
                if not bid_actions:
                    print("You cannot bid.")
                    continue
                    
                try:
                    amount = int(input(f"Enter bid amount (Multiples of 10, Min {min_bid}): "))
                    
                    # Find matching action
                    for action in bid_actions:
                        if action.amount == amount:
                            return action
                    
                    print(f"Invalid amount. Must be one of: {[a.amount for a in bid_actions]}")
                    
                except ValueError:
                    print("Invalid amount input.")
            else:
                 print("Invalid selection. Please enter 0, 1, or 2.")

    def _print_event_log(self, game: Game):
        current_len = len(game.action_history)
        if current_len > self.last_history_len:
            print(f"\n--- Game Log (Since last turn) ---")
            for i in range(self.last_history_len, current_len):
                entry = game.action_history[i]
                action_type = entry.get("action")
                details = entry.get("details", {})
                
                msg = self._format_log_entry(action_type, details, game)
                if msg:
                    print(f" > {msg}")
            print(f"----------------------------------")
        
        self.last_history_len = current_len

    def _format_log_entry(self, action_type: str, details: Dict[str, Any], game: Game) -> str:
        # Helper to get player name
        def p_name(pid):
            if pid == self.player_id:
                return "You"
            return game.players[pid].name if pid < len(game.players) else f"Player {pid}"

        if action_type == "start_auction":
            return f"Auction started by {p_name(details['player'])} for {details['animal']}."
        
        elif action_type == "bid":
             return f"{p_name(details['player'])} bids {details['amount']}."
             
        elif action_type == "pass":
            return f"{p_name(details['player'])} passes."
            
        elif action_type == "pass_force_money":
            return f"{p_name(details['player'])} forced to pass (not enough money)."

        elif action_type == "high_bidder_wins":
            return f"{p_name(details['bidder'])} wins auction! Pays {details['paid']} to {p_name(details['to'])}."
            
        elif action_type == "auctioneer_buys":
             return f"{p_name(details['auctioneer'])} (Auctioneer) buys animal! Pays {details['paid']} to {p_name(details['to'])}."
             
        elif action_type == "auctioneer_gets_free":
            return f"{p_name(details['auctioneer'])} gets animal for free (no bids)."

        elif action_type == "start_cow_trade":
            return f"Cow Trade: {p_name(details['initiator'])} attacks {p_name(details['target'])} for {details['animal']}."
            
        elif action_type == "cow_trade_offer":
             pass

        elif action_type == "cow_trade_counter_offer":
            pass

        elif action_type == "resolve_trade":
            winner = p_name(game.last_trade_result["winner_player_id"])
            loser = p_name(game.last_trade_result["loser_player_id"])
            return f"Trade Result: {winner} wins! Takes {details['animals_transferred']} animals from {loser}."
            
        elif action_type == "donkey_money":
            return f"Donkey #{details['donkey_number']} revealed! Distributing {details['value']} money."

        return None


    def _print_game_info(self, game: Game):
        player = game.players[self.player_id]
        
        print(f"\n--- State ---")
        print(f"Phase: {game.phase.name}")
        print(f"Cards in Deck: {len(game.animal_deck)}")
        
        if game.phase == GamePhase.AUCTION_BIDDING:
             # Show Auction Initiator
             initiator_idx = game.current_player_idx # Auctioneer
             print(f"Auction Initiated by: {game.players[initiator_idx].name if initiator_idx != self.player_id else 'You'}")

        if game.phase == GamePhase.COW_TRADE_RESPONSE:
            # Show Attack Info
            if game.trade_initiator is not None:
                initiator_name = game.players[game.trade_initiator].name
                animal_name = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
                card_count = game.trade_offer_card_count
                print(f"(!) {initiator_name} wants your {animal_name}!")
                print(f"    They offer {card_count} cards.")

        if game.current_animal:
            print(f"Current Auction Animal: {game.current_animal.animal_type.display_name}")
            high_bidder_id = game.auction_high_bidder
            high_bidder_name = game.players[high_bidder_id].name if high_bidder_id is not None else "None"
            print(f"Current High Bid: {game.auction_high_bid} (by {high_bidder_name})")
        
        print(f"\n--- Your Status ---")
        print(f"Money: {player.get_total_money()} (Cards: {sorted([c.value for c in player.money])})")
        
        # Format animals nicely
        animals_str = []
        for animal_type, count in player.get_animal_counts().items():
            if count > 0:
                animals_str.append(f"{animal_type.display_name}: {count}")
        print(f"Animals: {', '.join(animals_str) if animals_str else 'None'}")
        
    def _print_animals(self, game: Game):
        print(f"\n{'='*20}")
        print("OPPONENT ANIMALS")
        for p in game.players:
            if p.player_id == self.player_id:
                continue
            animals_str = []
            for animal_type, count in p.get_animal_counts().items():
                if count > 0:
                    animals_str.append(f"{animal_type.display_name}: {count}")
            print(f"{p.name} (Player {p.player_id}): {', '.join(animals_str) if animals_str else 'None'}")
        print(f"{'='*20}\n")

    def _print_money(self, game: Game):
        print(f"\n{'='*20}")
        print("OPPONENT MONEY (DEBUG)")
        for p in game.players:
            if p.player_id == self.player_id:
                continue
            print(f"{p.name} (Player {p.player_id}): {p.get_total_money()}")
        print(f"{'='*20}\n")

    def _format_action(self, action: GameAction) -> str:
        # Basic formatting, can be improved based on action type
        return str(action)
