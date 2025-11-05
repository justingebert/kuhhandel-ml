"""
Pytest tests for the game engine.

Run with: pytest tests/test_game.py -v
"""
import pytest
from gameengine import Game, AnimalType, GamePhase, ActionType, AnimalCard


@pytest.fixture
def game():
    """Create a fresh game instance for each test."""
    g = Game(num_players=3, seed=42)
    g.setup()
    return g


@pytest.fixture
def game_with_trades():
    """Create a game where cow trades are possible."""
    g = Game(num_players=3, seed=123)
    g.setup()

    # Give players matching animals
    cow1 = AnimalCard(AnimalType.COW, 100)
    cow2 = AnimalCard(AnimalType.COW, 101)
    g.players[0].add_animal(cow1)
    g.players[1].add_animal(cow2)

    return g


class TestGameInitialization:
    """Test game setup and initialization."""

    def test_game_creation(self):
        """Test that a game can be created."""
        game = Game(num_players=3)
        assert game.num_players == 3
        assert game.phase == GamePhase.SETUP

    def test_invalid_player_count(self):
        """Test that invalid player counts raise errors."""
        with pytest.raises(ValueError):
            Game(num_players=2)

        with pytest.raises(ValueError):
            Game(num_players=6)

    def test_game_setup(self, game):
        """Test that setup initializes game correctly."""
        assert len(game.players) == 3
        assert len(game.animal_deck) == 40  # 10 types x 4 cards
        assert game.phase == GamePhase.PLAYER_TURN

        # Check starting money
        for player in game.players:
            assert player.get_total_money() == 90  # 2x0 + 4x10 + 1x50

    def test_deterministic_with_seed(self):
        """Test that seeded games are deterministic."""
        game1 = Game(num_players=3, seed=42)
        game1.setup()

        game2 = Game(num_players=3, seed=42)
        game2.setup()

        # First animal should be the same
        assert game1.animal_deck[0].animal_type == game2.animal_deck[0].animal_type


class TestAuction:
    """Test auction mechanics."""

    def test_start_auction(self, game):
        """Test starting an auction."""
        initial_deck_size = len(game.animal_deck)
        animal = game.start_auction()

        assert animal is not None
        assert game.phase == GamePhase.AUCTION
        assert len(game.animal_deck) == initial_deck_size - 1
        assert game.current_animal == animal

    def test_valid_auction_actions(self, game):
        """Test valid actions during auction."""
        game.start_auction()

        # Auctioneer can pass
        auctioneer_actions = game.get_valid_actions(game.current_player_idx)
        assert ActionType.PASS in auctioneer_actions

        # Other players can bid
        other_player = (game.current_player_idx + 1) % game.num_players
        other_actions = game.get_valid_actions(other_player)
        assert ActionType.BID in other_actions

    def test_bidding(self, game):
        """Test bidding on an auction."""
        game.start_auction()
        other_player = (game.current_player_idx + 1) % game.num_players

        # Valid bid
        success = game.place_bid(other_player, 10)
        assert success
        assert game.auction_high_bid == 10
        assert game.auction_high_bidder == other_player

        # Bid too low
        success = game.place_bid(other_player, 5)
        assert not success

    def test_auctioneer_passes_no_bids(self, game):
        """Test auctioneer passing with no bids."""
        auctioneer = game.get_current_player()
        game.start_auction()

        game.auctioneer_passes()

        # Auctioneer gets animal for free
        assert game.phase == GamePhase.PLAYER_TURN
        assert len(auctioneer.animals) == 1

    def test_auctioneer_passes_with_bids(self, game):
        """Test auctioneer passing with bids."""
        game.start_auction()
        high_bidder_id = (game.current_player_idx + 1) % game.num_players
        high_bidder = game.players[high_bidder_id]

        game.place_bid(high_bidder_id, 20)
        game.auctioneer_passes()

        # High bidder gets animal
        assert len(high_bidder.animals) == 1
        assert game.phase == GamePhase.PLAYER_TURN


class TestTrade:
    """Test cow trade mechanics."""

    def test_valid_trade_actions(self, game_with_trades):
        """Test that cow trade is available when players have matching animals."""
        game = game_with_trades
        actions = game.get_valid_actions(0)

        assert ActionType.START_COW_TRADE in actions

    def test_start_trade(self, game_with_trades):
        """Test starting a cow trade."""
        game = game_with_trades
        offer_cards = [game.players[0].money[0]]

        success = game.start_cow_trade(1, AnimalType.COW, offer_cards)

        assert success
        assert game.phase == GamePhase.COW_TRADE
        assert game.trade_initiator == 0
        assert game.trade_target == 1

    def test_accept_trade(self, game_with_trades):
        """Test accepting a trade offer."""
        game = game_with_trades
        offer_cards = [game.players[0].money[0], game.players[0].money[1]]

        game.start_cow_trade(1, AnimalType.COW, offer_cards)
        initial_p0_animals = len(game.players[0].animals)

        game.accept_trade_offer()

        # Initiator should get the animal
        assert len(game.players[0].animals) == initial_p0_animals + 1
        assert game.phase == GamePhase.PLAYER_TURN

    def test_counter_offer_win(self, game_with_trades):
        """Test counter offer that wins."""
        game = game_with_trades

        # Offer 0
        offer_cards = [game.players[0].money[0]]
        game.start_cow_trade(1, AnimalType.COW, offer_cards)

        # Counter with 10 (higher)
        counter_cards = [game.players[1].money[4]]  # 10-card
        initial_p1_animals = len(game.players[1].animals)

        game.counter_trade_offer(counter_cards)

        # Target wins (higher counter)
        assert len(game.players[1].animals) == initial_p1_animals + 1
        assert game.phase == GamePhase.PLAYER_TURN

    def test_no_trade_with_complete_set(self, game_with_trades):
        """Test that you can't trade for complete sets."""
        game = game_with_trades

        # Give player 1 all 4 cows
        for i in range(3):
            cow = AnimalCard(AnimalType.COW, 102 + i)
            game.players[1].add_animal(cow)

        # Player 0 shouldn't be able to trade
        actions = game.get_valid_actions(0)
        assert ActionType.START_COW_TRADE not in actions


class TestGameOver:
    """Test game ending conditions."""

    def test_game_not_over_with_animals(self, game):
        """Test game isn't over while animals remain."""
        assert not game.is_game_over()

    def test_game_over_all_complete_sets(self, game):
        """Test game is over when all sets are complete."""
        # Remove all animal cards from deck
        game.animal_deck.clear()

        # Give each player complete sets only
        for player_id, player in enumerate(game.players):
            animal_types = [AnimalType.COW, AnimalType.DOG] if player_id == 0 else (
                [AnimalType.CAT, AnimalType.GOAT] if player_id == 1 else [AnimalType.PIG, AnimalType.HORSE]
            )
            for animal_type in animal_types:
                for i in range(4):
                    animal = AnimalCard(animal_type, player_id * 100 + i)
                    player.add_animal(animal)

        assert game.is_game_over()

    def test_get_winner(self, game):
        """Test getting the winner."""
        # Clear deck
        game.animal_deck.clear()

        # Give player 0 high-value animals
        for animal_type in [AnimalType.HORSE, AnimalType.COW, AnimalType.PIG]:
            for i in range(4):
                animal = AnimalCard(animal_type, i)
                game.players[0].add_animal(animal)

        winner = game.get_winner()
        assert winner is not None
        assert winner.player_id == 0

    ##TODO game end test
    # def test_game_end_detection(self, game):
    #     """Test that game end is properly detected."""
    #     # Initially game should not be over
    #     assert not game.is_game_over()
    #
    #     for animal_type in AnimalType.get_all_types():
    #         for i in range(4):
    #             animal = AnimalCard(animal_type, i)
    #             game.players[0].add_animal(animal)
    #
    #     # Now should detect game over
    #     assert game.is_game_over()


class TestScoring:
    """Test scoring mechanics."""

    def test_score_calculation(self, game):
        """Test that scores are calculated correctly."""
        player = game.players[0]

        # Give player: 2 cows (200), 1 dog (40) = 240 total
        # 2 different types = 240 * 2 = 480
        for i in range(2):
            player.add_animal(AnimalCard(AnimalType.COW, i))
        player.add_animal(AnimalCard(AnimalType.DOG, 10))

        score = player.calculate_score()
        assert score == 240 * 2  # (100*2 + 40) * 2 types

    def test_complete_set_bonus(self, game):
        """Test scoring with complete sets."""
        player = game.players[0]

        # Give player all 4 horses (value 500 for complete set)
        for i in range(4):
            player.add_animal(AnimalCard(AnimalType.HORSE, i))

        score = player.calculate_score()
        assert score == 500  # 500 * 1 type


class TestDonkeyBonus:
    """Test donkey money distribution."""

    def test_donkey_triggers_money(self, game):
        """Test that drawing a donkey gives money to all players."""
        # Find a donkey in the deck
        for i, card in enumerate(game.animal_deck):
            if card.animal_type == AnimalType.DONKEY:
                # Move donkey to top
                game.animal_deck[0], game.animal_deck[i] = game.animal_deck[i], game.animal_deck[0]
                break

        initial_money = [p.get_total_money() for p in game.players]

        game.start_auction()

        # All players should have received 50 (first donkey)
        for i, player in enumerate(game.players):
            assert player.get_total_money() == initial_money[i] + 50


@pytest.mark.parametrize("num_players", [3, 4, 5])
def test_different_player_counts(num_players):
    """Test game works with different player counts."""
    game = Game(num_players=num_players, seed=42)
    game.setup()

    assert len(game.players) == num_players
    assert all(p.get_total_money() == 90 for p in game.players)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

