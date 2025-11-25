"""Type-safe action definitions for the game using dataclasses."""
from dataclasses import dataclass, field
from typing import List, Union
from .Money import MoneyCard
from .Animal import AnimalType
from .game import ActionType


# Base class for all actions (optional but recommended)
@dataclass(frozen=True)
class GameActionBase:
    """Base class for all game actions."""
    pass


@dataclass(frozen=True)
class BidAction(GameActionBase):
    amount: int
    type: ActionType = field(default=ActionType.BID, init=False)


@dataclass(frozen=True)
class PassAction(GameActionBase):
    type: ActionType = field(default=ActionType.PASS, init=False)


@dataclass(frozen=True)
class StartAuctionAction(GameActionBase):
    type: ActionType = field(default=ActionType.START_AUCTION, init=False)


@dataclass(frozen=True)
class BuyAsAuctioneerAction(GameActionBase):
    type: ActionType = field(default=ActionType.BUY_AS_AUCTIONEER, init=False)


@dataclass(frozen=True)
class StartCowTradeAction(GameActionBase):
    target_id: int
    animal_type: AnimalType
    money_cards: List[MoneyCard]
    type: ActionType = field(default=ActionType.START_COW_TRADE, init=False)


@dataclass(frozen=True)
class CounterOfferAction(GameActionBase):
    money_cards: List[MoneyCard]
    type: ActionType = field(default=ActionType.COUNTER_OFFER, init=False)


@dataclass(frozen=True)
class AcceptOfferAction(GameActionBase):
    type: ActionType = field(default=ActionType.ACCEPT_OFFER, init=False)


# Union of all possible actions
GameAction = Union[
    BidAction,
    PassAction,
    StartAuctionAction,
    BuyAsAuctioneerAction,
    StartCowTradeAction,
    CounterOfferAction,
    AcceptOfferAction
]


class Actions:
    """Factory class for creating action objects with a clean API."""
    
    @staticmethod
    def bid(amount: int) -> BidAction:
        """Create a bid action."""
        return BidAction(amount=amount)
    
    @staticmethod
    def pass_action() -> PassAction:
        """Create a pass action."""
        return PassAction()
    
    @staticmethod
    def start_auction() -> StartAuctionAction:
        """Create a start auction action."""
        return StartAuctionAction()
    
    @staticmethod
    def buy_as_auctioneer() -> BuyAsAuctioneerAction:
        """Create a buy as auctioneer action."""
        return BuyAsAuctioneerAction()
    
    @staticmethod
    def start_cow_trade(target_id: int, animal_type: AnimalType, money_cards: List[MoneyCard]) -> StartCowTradeAction:
        """Create a start cow trade action."""
        return StartCowTradeAction(
            target_id=target_id,
            animal_type=animal_type,
            money_cards=money_cards
        )
    
    @staticmethod
    def counter_offer(money_cards: List[MoneyCard]) -> CounterOfferAction:
        """Create a counter offer action."""
        return CounterOfferAction(money_cards=money_cards)
    
    @staticmethod
    def accept_offer() -> AcceptOfferAction:
        """Create an accept offer action."""
        return AcceptOfferAction()

