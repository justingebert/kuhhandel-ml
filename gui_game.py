"""
Graphical User Interface for playing Kuhhandel against AI opponents.
Uses PyQt6 for the GUI framework.
"""
import os
import sys
from typing import List, Dict, Any, Optional, Callable
from queue import Queue
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QScrollArea, QTextEdit, QLineEdit,
    QSlider, QGroupBox, QGridLayout, QSplitter, QMessageBox, QSpinBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont, QTextCursor

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable import distributions as maskable_dist

from gameengine.agent import Agent
from gameengine.game import Game, GamePhase
from gameengine.actions import GameAction, ActionType
from gameengine.Animal import AnimalType
from rl.env import KuhhandelEnv
from rl.agents.model_agent import ModelAgent
from rl.agents.random_agent import RandomAgent
from rl.train.train_selfplay import robust_apply_masking

# Patch for maskable distribution
maskable_dist.MaskableCategorical.apply_masking = robust_apply_masking


# Animal emoji mapping for visual appeal
ANIMAL_EMOJIS = {
    "Chicken": "🐔",
    "Goose": "🦢",
    "Cat": "🐱",
    "Dog": "🐕",
    "Sheep": "🐑",
    "Goat": "🐐",
    "Donkey": "🫏",
    "Pig": "🐷",
    "Cow": "🐄",
    "Horse": "🐴",
}


class GameSignals(QObject):
    """Signals for thread-safe GUI updates."""
    update_state = pyqtSignal(object, list)  # game, valid_actions
    game_over = pyqtSignal(dict)  # scores
    log_message = pyqtSignal(str)
    error = pyqtSignal(str)


class GUIAgent(Agent):
    """An agent that interfaces with the GUI for user input."""

    def __init__(self, name: str, signals: GameSignals):
        super().__init__(name)
        self.signals = signals
        self.last_history_len = 0
        self.action_queue: Queue = Queue()
        self.waiting_for_action = False

    def get_action(self, game: Game, valid_actions: List[GameAction]) -> GameAction:
        """Get action from GUI."""
        self.waiting_for_action = True
        
        # Update event log first
        self._update_event_log(game)
        
        # Signal GUI to update (thread-safe)
        self.signals.update_state.emit(game, valid_actions)
        
        # Wait for user action from GUI
        action = self.action_queue.get()
        self.waiting_for_action = False
        return action

    def submit_action(self, action: GameAction):
        """Submit an action (called from GUI)."""
        self.action_queue.put(action)

    def _update_event_log(self, game: Game):
        """Update event log with new events since last turn."""
        current_len = len(game.action_history)
        if current_len > self.last_history_len:
            for i in range(self.last_history_len, current_len):
                msg = self._format_log_entry(i, game)
                if msg:
                    self.signals.log_message.emit(msg)
        self.last_history_len = current_len

    def _format_log_entry(self, index: int, game: Game) -> Optional[str]:
        """Format a log entry for display."""
        entry = game.action_history[index]
        action_type = entry.get("action")
        details = entry.get("details", {})

        def p_name(pid):
            if pid == self.player_id:
                return "You"
            return game.players[pid].name

        if action_type == "start_auction":
            animal = details.get('animal', 'Unknown')
            emoji = ANIMAL_EMOJIS.get(animal, "")
            return f"🔨 Auction started by {p_name(details['player'])} for {emoji} {animal}"

        elif action_type == "bid":
            return f"💰 {p_name(details['player'])} bids {details['amount']}"

        elif action_type in ("pass", "pass_force_money"):
            return f"⏭️ {p_name(details['player'])} passes"

        elif action_type == "high_bidder_wins":
            return f"✅ {p_name(details['bidder'])} wins auction! Pays {details['paid']} to {p_name(details['to'])}"

        elif action_type == "auctioneer_buys":
            return f"✅ {p_name(details['auctioneer'])} (Auctioneer) buys animal! Pays {details['paid']} to {p_name(details['to'])}"

        elif action_type == "auctioneer_gets_free":
            return f"🎉 {p_name(details['auctioneer'])} gets animal for free (no bids)"

        elif action_type == "start_cow_trade":
            animal = details.get('animal', 'Unknown')
            emoji = ANIMAL_EMOJIS.get(animal, "")
            return f"🤝 Cow Trade: {p_name(details['initiator'])} attacks {p_name(details['target'])} for {emoji} {animal}"

        elif action_type == "resolve_trade":
            initiator_id = None
            target_id = None
            for j in range(index - 1, -1, -1):
                prev_entry = game.action_history[j]
                if prev_entry["action"] == "start_cow_trade":
                    initiator_id = prev_entry["details"]["initiator"]
                    target_id = prev_entry["details"]["target"]
                    break

            if initiator_id is not None:
                winner_role = details.get("winner")
                if winner_role == "initiator":
                    winner = p_name(initiator_id)
                    loser = p_name(target_id)
                else:
                    winner = p_name(target_id)
                    loser = p_name(initiator_id)

                return f"📊 Trade Result: {winner} wins! Takes {details['animals_transferred']} animals from {loser}"

        elif action_type == "donkey_money":
            return f"🫏 Donkey #{details['donkey_number']} revealed! Distributing {details['value']} money"

        return None


class GameThread(QThread):
    """Thread to run the game loop."""
    
    def __init__(self, env: KuhhandelEnv, signals: GameSignals):
        super().__init__()
        self.env = env
        self.signals = signals

    def run(self):
        try:
            scores = self.env.controller.run()
            self.signals.game_over.emit(scores)
        except Exception as e:
            self.signals.error.emit(str(e))


class KuhhandelGUI(QMainWindow):
    """Main GUI class for Kuhhandel game."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🐄 Kuhhandel - Card Game 🐴")
        self.setMinimumSize(1200, 800)
        
        # Game state
        self.env: Optional[KuhhandelEnv] = None
        self.gui_agent: Optional[GUIAgent] = None
        self.game_thread: Optional[GameThread] = None
        self.signals = GameSignals()
        self.current_game: Optional[Game] = None
        self.current_valid_actions: List[GameAction] = []
        
        # Connect signals
        self.signals.update_state.connect(self._on_update_state)
        self.signals.game_over.connect(self._on_game_over)
        self.signals.log_message.connect(self._add_log_message)
        self.signals.error.connect(self._on_error)
        
        # Build UI
        self._build_ui()
        
        # Show start screen
        self._show_start_screen()

    def _build_ui(self):
        """Build the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self._build_left_panel(left_layout)
        splitter.addWidget(left_widget)
        
        # Right panel
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self._build_right_panel(right_layout)
        splitter.addWidget(right_widget)
        
        # Set initial sizes (30% left, 70% right)
        splitter.setSizes([360, 840])

    def _build_left_panel(self, layout: QVBoxLayout):
        """Build the left panel with player info."""
        # Your Status
        self.player_group = QGroupBox("📋 Your Status")
        player_layout = QVBoxLayout(self.player_group)
        
        self.player_money_label = QLabel("💰 Money: -")
        self.player_money_label.setWordWrap(True)
        self.player_cards_label = QLabel("💳 Cards: -")
        self.player_cards_label.setWordWrap(True)
        
        player_layout.addWidget(self.player_money_label)
        player_layout.addWidget(self.player_cards_label)
        layout.addWidget(self.player_group)
        
        # Your Animals
        self.your_animals_group = QGroupBox("🐾 Your Animals")
        animals_layout = QVBoxLayout(self.your_animals_group)
        
        self.your_animals_label = QLabel("None")
        self.your_animals_label.setWordWrap(True)
        animals_layout.addWidget(self.your_animals_label)
        layout.addWidget(self.your_animals_group)
        
        # Opponents
        self.opponents_group = QGroupBox("👥 Opponents")
        self.opponents_layout = QVBoxLayout(self.opponents_group)
        layout.addWidget(self.opponents_group)
        
        # Event Log
        self.log_group = QGroupBox("📜 Event Log")
        log_layout = QVBoxLayout(self.log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Menlo", 10))  # Use Menlo on macOS (monospace font)
        log_layout.addWidget(self.log_text)
        layout.addWidget(self.log_group, stretch=1)
        
        self.opponent_widgets: List[Dict[str, QLabel]] = []

    def _build_right_panel(self, layout: QVBoxLayout):
        """Build the right panel with game area."""
        # Game Info
        self.game_info_group = QGroupBox("🎮 Game Info")
        info_layout = QVBoxLayout(self.game_info_group)
        
        self.phase_label = QLabel("Phase: -")
        self.phase_label.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        self.phase_label.setStyleSheet("color: #2E86AB;")
        
        self.round_label = QLabel("Round: -")
        self.deck_label = QLabel("Cards in Deck: -")
        self.context_info_label = QLabel("")
        self.context_info_label.setWordWrap(True)
        
        info_layout.addWidget(self.phase_label)
        info_layout.addWidget(self.round_label)
        info_layout.addWidget(self.deck_label)
        info_layout.addWidget(self.context_info_label)
        layout.addWidget(self.game_info_group)
        
        # Actions
        self.actions_group = QGroupBox("🎯 Your Actions")
        actions_layout = QVBoxLayout(self.actions_group)
        
        # Turn Status Label
        self.turn_status_label = QLabel("")
        self.turn_status_label.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
        self.turn_status_label.setWordWrap(True)
        self.turn_status_label.setStyleSheet("color: #FF6B6B; padding: 8px; background-color: #F5F5F5; border-radius: 4px;")
        actions_layout.addWidget(self.turn_status_label)
        
        self.action_desc_label = QLabel("Choose your action:")
        self.action_desc_label.setFont(QFont("Helvetica", 12, QFont.Weight.Bold))
        self.action_desc_label.setWordWrap(True)
        actions_layout.addWidget(self.action_desc_label)
        
        # Scroll area for action buttons
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.action_buttons_widget = QWidget()
        self.action_buttons_layout = QVBoxLayout(self.action_buttons_widget)
        self.action_buttons_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        scroll.setWidget(self.action_buttons_widget)
        actions_layout.addWidget(scroll, stretch=1)
        
        layout.addWidget(self.actions_group, stretch=1)

    def _show_start_screen(self):
        """Show the start screen with game options."""
        self._clear_action_buttons()
        
        # Title
        title = QLabel("🐄 Welcome to Kuhhandel! 🐴")
        title.setFont(QFont("Helvetica", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_buttons_layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "Kuhhandel is a classic card game of animal trading and auctions.\n"
            "Try to collect complete sets of 4 animals to score points!\n\n"
            "Configure your game and click Start!"
        )
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        instructions.setWordWrap(True)
        self.action_buttons_layout.addWidget(instructions)
        
        # Spacer
        self.action_buttons_layout.addSpacing(20)
        
        # Model path input
        model_frame = QWidget()
        model_layout = QHBoxLayout(model_frame)
        model_layout.setContentsMargins(0, 0, 0, 0)
        
        model_label = QLabel("AI Model Path:")
        self.model_path_input = QLineEdit("rl\\train\\models\\kuhhandel_ppo_latest.zip")
        self.model_path_input.setMinimumWidth(400)
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_path_input)
        model_layout.addStretch()
        self.action_buttons_layout.addWidget(model_frame)
        
        # Player count
        players_frame = QWidget()
        players_layout = QHBoxLayout(players_frame)
        players_layout.setContentsMargins(0, 0, 0, 0)
        
        players_label = QLabel("Number of Players:")
        self.players_spinbox = QSpinBox()
        self.players_spinbox.setRange(3, 5)
        self.players_spinbox.setValue(3)
        
        players_layout.addWidget(players_label)
        players_layout.addWidget(self.players_spinbox)
        players_layout.addStretch()
        self.action_buttons_layout.addWidget(players_frame)
        
        # Spacer
        self.action_buttons_layout.addSpacing(20)
        
        # Start button
        start_btn = QPushButton("🎮 Start Game")
        start_btn.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        start_btn.setMinimumHeight(50)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        start_btn.clicked.connect(self._start_game)
        self.action_buttons_layout.addWidget(start_btn)
        
        self.action_buttons_layout.addStretch()
        
        self.phase_label.setText("Phase: Setup")
        self.action_desc_label.setText("Configure and start a new game")

    def _start_game(self):
        """Start a new game."""
        num_players = self.players_spinbox.value()
        model_path = self.model_path_input.text()
        
        # Load model if exists
        model_instance = None
        if os.path.exists(model_path):
            self._add_log_message(f"Loading AI model from {model_path}...")
            try:
                model_instance = MaskablePPO.load(model_path, device='cpu')
                self._add_log_message("✅ Model loaded successfully!")
            except Exception as e:
                self._add_log_message(f"⚠️ Failed to load model: {e}")
                self._add_log_message("Using Random AI opponents instead.")
        else:
            self._add_log_message(f"⚠️ Model not found at {model_path}")
            self._add_log_message("Using Random AI opponents instead.")

        # Create opponent generator
        def opponent_generator(env, count):
            agents = []
            for i in range(count):
                name = f"AI {i+1}"
                if model_instance:
                    agents.append(ModelAgent(name, model_path, env, model_instance=model_instance))
                else:
                    agents.append(RandomAgent(name))
            return agents

        # Create environment
        self.env = KuhhandelEnv(num_players=num_players, opponent_generator=opponent_generator)
        
        # Reset to setup the game
        self._add_log_message("Setting up game...")
        self.env.reset()
        
        # Create and inject GUI agent
        self.gui_agent = GUIAgent("You", self.signals)
        self.gui_agent.set_player_id(0)
        self.env.agents[0] = self.gui_agent
        
        # Setup opponent display
        self._setup_opponent_display()
        
        # Start game log
        self._add_log_message("=" * 40)
        self._add_log_message("🎮 GAME STARTED!")
        self._add_log_message("=" * 40)
        
        # Start game thread
        self.game_thread = GameThread(self.env, self.signals)
        self.game_thread.start()

    def _setup_opponent_display(self):
        """Setup opponent info labels."""
        # Clear existing
        for widget in self.opponent_widgets:
            for label in widget.values():
                if isinstance(label, QLabel):
                    label.deleteLater()
        self.opponent_widgets.clear()
        
        # Clear layout
        while self.opponents_layout.count():
            item = self.opponents_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create labels for each opponent
        for i, agent in enumerate(self.env.agents):
            if agent.player_id == 0:  # Skip player
                continue
            
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(5, 5, 5, 5)
            
            name_label = QLabel(f"🤖 {agent.name}")
            name_label.setFont(QFont("Helvetica", 11, QFont.Weight.Bold))
            
            cards_label = QLabel("💳 Money Cards: -")
            animals_label = QLabel("🐾 Animals: None")
            animals_label.setWordWrap(True)
            
            frame_layout.addWidget(name_label)
            frame_layout.addWidget(cards_label)
            frame_layout.addWidget(animals_label)
            
            self.opponents_layout.addWidget(frame)
            
            self.opponent_widgets.append({
                'player_id': agent.player_id,
                'name': name_label,
                'cards': cards_label,
                'animals': animals_label
            })

    def _on_update_state(self, game: Game, valid_actions: List[GameAction]):
        """Handle game state update (called from signal)."""
        self.current_game = game
        self.current_valid_actions = valid_actions
        
        self._update_game_display(game)
        self._present_actions(game, valid_actions)

    def _update_game_display(self, game: Game):
        """Update all game state displays."""
        player = game.players[0]
        
        # Update phase
        phase_name = game.phase.name.replace("_", " ").title()
        self.phase_label.setText(f"Phase: {phase_name}")
        
        # Update round and deck
        self.round_label.setText(f"Round: {game.round_number}")
        self.deck_label.setText(f"Cards in Deck: {len(game.animal_deck)}")
        
        # Update player money
        total_money = player.get_total_money()
        money_cards = sorted([c.value for c in player.money])
        self.player_money_label.setText(f"💰 Total Money: {total_money}")
        self.player_cards_label.setText(f"💳 Cards: {money_cards}")
        
        # Update player animals
        animals_str = self._format_animals(player.get_animal_counts())
        self.your_animals_label.setText(animals_str if animals_str else "None")
        
        # Update opponent info
        for opp_info in self.opponent_widgets:
            opp_player = game.players[opp_info['player_id']]
            opp_info['cards'].setText(f"💳 Money Cards: {len(opp_player.money)}")
            opp_animals = self._format_animals(opp_player.get_animal_counts())
            opp_info['animals'].setText(f"🐾 {opp_animals if opp_animals else 'None'}")
        
        # Update context info
        context_text = self._get_context_info(game)
        self.context_info_label.setText(context_text)
        
        # Update turn status
        self._update_turn_status(game)

    def _format_animals(self, animal_counts: Dict[AnimalType, int]) -> str:
        """Format animal counts as a string with emojis."""
        parts = []
        for animal_type, count in animal_counts.items():
            if count > 0:
                emoji = ANIMAL_EMOJIS.get(animal_type.display_name, "")
                parts.append(f"{emoji}{animal_type.display_name}: {count}")
        return ", ".join(parts)

    def _get_context_info(self, game: Game) -> str:
        """Get context-specific info text."""
        if game.phase == GamePhase.AUCTION_BIDDING and game.current_animal:
            animal = game.current_animal.animal_type.display_name
            emoji = ANIMAL_EMOJIS.get(animal, "")
            initiator = game.players[game.current_player_idx].name
            high_bidder = game.players[game.auction_high_bidder].name if game.auction_high_bidder is not None else "None"
            return f"🔨 Auction: {emoji} {animal} | Initiated by: {initiator}\nHigh Bid: {game.auction_high_bid} by {high_bidder}"
        
        elif game.phase == GamePhase.COW_TRADE_OFFER:
            if game.trade_target is not None:
                target = game.players[game.trade_target].name
                animal = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
                emoji = ANIMAL_EMOJIS.get(animal, "")
                return f"🤝 Cow Trade: You're attacking {target} for their {emoji} {animal}"
        
        elif game.phase == GamePhase.COW_TRADE_RESPONSE:
            if game.trade_initiator is not None:
                initiator = game.players[game.trade_initiator].name
                animal = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
                emoji = ANIMAL_EMOJIS.get(animal, "")
                return f"🛡️ Defense: {initiator} wants your {emoji} {animal}!\nThey offer {game.trade_offer_card_count} cards."
        
        return ""

    def _update_turn_status(self, game: Game):
        """Update turn status label based on current phase."""
        status_text = ""
        
        if game.phase == GamePhase.AUCTION_BIDDING and game.current_animal:
            animal = game.current_animal.animal_type.display_name
            emoji = ANIMAL_EMOJIS.get(animal, "")
            high_bidder = game.players[game.auction_high_bidder].name if game.auction_high_bidder is not None else "None"
            status_text = f"🔨 Current Bid: {game.auction_high_bid} | 🏆 By: {high_bidder} | 🐾 Animal: {emoji} {animal}"
        
        elif game.phase == GamePhase.COW_TRADE_OFFER:
            if game.trade_target is not None:
                animal = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
                emoji = ANIMAL_EMOJIS.get(animal, "")
                status_text = f"🤝 Trading for {emoji} {animal}"
        
        elif game.phase == GamePhase.COW_TRADE_RESPONSE:
            if game.trade_initiator is not None:
                animal = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
                emoji = ANIMAL_EMOJIS.get(animal, "")
                card_count = game.trade_offer_card_count
                status_text = f"🛡️ Defending | 💳 Cards Offered: {card_count} | 🐾 Animal: {emoji} {animal}"
        
        self.turn_status_label.setText(status_text)

    def _clear_action_buttons(self):
        """Clear all action buttons."""
        while self.action_buttons_layout.count():
            item = self.action_buttons_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _present_actions(self, game: Game, valid_actions: List[GameAction]):
        """Present action choices to the user."""
        self._clear_action_buttons()
        
        # Auto-select if only one action
        if len(valid_actions) == 1:
            action = valid_actions[0]
            self.action_desc_label.setText(f"Only one action available: {action}")
            # Delay to let user see the message
            QTimer.singleShot(300, lambda: self.gui_agent.submit_action(action))
            return
        
        # Handle different phases
        if game.phase == GamePhase.AUCTION_BIDDING:
            self._present_auction_bidding(valid_actions, game)
        elif game.phase == GamePhase.COW_TRADE_OFFER:
            self._present_cow_trade_offer(valid_actions, game)
        elif game.phase == GamePhase.COW_TRADE_RESPONSE:
            self._present_cow_trade_response(valid_actions, game)
        else:
            self._present_standard_actions(valid_actions)

    def _create_action_button(self, text: str, action: GameAction, color: str = "#2196F3") -> QPushButton:
        """Create a styled action button."""
        btn = QPushButton(text)
        btn.setMinimumHeight(40)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
        """)
        btn.clicked.connect(lambda: self.gui_agent.submit_action(action))
        return btn

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color by 15%."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r = max(0, int(r * 0.85))
        g = max(0, int(g * 0.85))
        b = max(0, int(b * 0.85))
        return f"#{r:02x}{g:02x}{b:02x}"

    def _present_standard_actions(self, valid_actions: List[GameAction]):
        """Present standard action buttons."""
        self.action_desc_label.setText("Choose your action:")
        
        for action in valid_actions:
            btn = self._create_action_button(
                self._format_action_button(action),
                action
            )
            self.action_buttons_layout.addWidget(btn)
        
        self.action_buttons_layout.addStretch()

    def _format_action_button(self, action: GameAction) -> str:
        """Format action for button display."""
        if action.type == ActionType.START_AUCTION:
            return "🔨 Start Auction"
        elif action.type == ActionType.START_COW_TRADE:
            return "🤝 Start Cow Trade"
        elif action.type == ActionType.COW_TRADE_CHOOSE_OPPONENT:
            return f"🎯 Trade with Player {action.target_id}"
        elif action.type == ActionType.COW_TRADE_CHOOSE_ANIMAL:
            emoji = ANIMAL_EMOJIS.get(action.animal_type.display_name, "")
            return f"{emoji} Trade for {action.animal_type.display_name}"
        elif action.type == ActionType.BUY_AS_AUCTIONEER:
            return "💵 Buy (Auctioneer)"
        elif action.type == ActionType.PASS_AS_AUCTIONEER:
            return "⏭️ Pass (Auctioneer)"
        else:
            return str(action)

    def _present_auction_bidding(self, valid_actions: List[GameAction], game: Game):
        """Present auction bidding UI."""
        self.action_desc_label.setText("🔨 Auction - Place your bid:")
        
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
        max_bid = bid_actions[-1].amount if bid_actions else 0
        
        # Pass button
        if pass_action:
            pass_btn = self._create_action_button("⏭️ Pass", pass_action, "#9E9E9E")
            self.action_buttons_layout.addWidget(pass_btn)
        
        if bid_actions:
            # Minimum bid button
            min_btn = self._create_action_button(
                f"💰 Bid Minimum ({min_bid})",
                bid_actions[0],
                "#4CAF50"
            )
            self.action_buttons_layout.addWidget(min_btn)
            
            # Slider section
            slider_frame = QFrame()
            slider_layout = QVBoxLayout(slider_frame)
            
            slider_label = QLabel(f"Custom Bid ({min_bid} - {max_bid}):")
            slider_layout.addWidget(slider_label)
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_bid)
            slider.setMaximum(max_bid)
            slider.setValue(min_bid)
            slider.setTickInterval(10)
            slider.setSingleStep(10)
            
            value_label = QLabel(f"Selected: {min_bid}")
            
            def on_slider_change(value):
                rounded = round(value / 10) * 10
                value_label.setText(f"Selected: {rounded}")
            
            slider.valueChanged.connect(on_slider_change)
            
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label)
            
            # Store references to avoid garbage collection
            self._current_slider = slider
            self._current_bid_actions = bid_actions
            
            # Submit button
            def submit_bid():
                amount = round(self._current_slider.value() / 10) * 10
                for action in self._current_bid_actions:
                    if action.amount == amount:
                        self.gui_agent.submit_action(action)
                        return
                closest = min(self._current_bid_actions, key=lambda a: abs(a.amount - amount))
                self.gui_agent.submit_action(closest)
            
            submit_btn = QPushButton("💵 Place Bid")
            submit_btn.setMinimumHeight(40)
            submit_btn.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    border: none;
                    border-radius: 6px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background-color: #F57C00;
                }
            """)
            submit_btn.clicked.connect(submit_bid)
            slider_layout.addWidget(submit_btn)
            
            self.action_buttons_layout.addWidget(slider_frame)
        else:
            no_bid_label = QLabel("⚠️ Not enough money to bid")
            self.action_buttons_layout.addWidget(no_bid_label)
        
        self.action_buttons_layout.addStretch()

    def _present_cow_trade_offer(self, valid_actions: List[GameAction], game: Game):
        """Present cow trade offer UI."""
        target_name = game.players[game.trade_target].name if game.trade_target is not None else "Unknown"
        animal_name = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
        emoji = ANIMAL_EMOJIS.get(animal_name, "")
        
        self.action_desc_label.setText(f"🤝 Offer for {target_name}'s {emoji} {animal_name}:")
        
        offer_actions = [a for a in valid_actions if a.type == ActionType.COW_TRADE_OFFER]
        offer_actions.sort(key=lambda x: x.amount)
        
        max_offer = offer_actions[-1].amount if offer_actions else 0
        
        # Quick offer buttons
        quick_frame = QWidget()
        quick_layout = QHBoxLayout(quick_frame)
        
        for amount in [0, 10, 50, 100]:
            if amount <= max_offer:
                action = next((a for a in offer_actions if a.amount == amount), None)
                if action:
                    btn = self._create_action_button(f"💰 {amount}", action, "#4CAF50")
                    btn.setMinimumWidth(60)
                    quick_layout.addWidget(btn)
        
        self.action_buttons_layout.addWidget(quick_frame)
        
        # Slider
        slider_frame = QFrame()
        slider_layout = QVBoxLayout(slider_frame)
        
        slider_label = QLabel(f"Custom Offer (0 - {max_offer}):")
        slider_layout.addWidget(slider_label)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max_offer)
        slider.setValue(0)
        slider.setTickInterval(10)
        
        value_label = QLabel("Selected: 0")
        
        def on_slider_change(value):
            rounded = round(value / 10) * 10
            value_label.setText(f"Selected: {rounded}")
        
        slider.valueChanged.connect(on_slider_change)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        
        # Store references
        self._current_slider = slider
        self._current_offer_actions = offer_actions
        
        def submit_offer():
            amount = round(self._current_slider.value() / 10) * 10
            for action in self._current_offer_actions:
                if action.amount == amount:
                    self.gui_agent.submit_action(action)
                    return
            closest = min(self._current_offer_actions, key=lambda a: abs(a.amount - amount))
            self.gui_agent.submit_action(closest)
        
        submit_btn = QPushButton("📤 Make Offer")
        submit_btn.setMinimumHeight(40)
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        submit_btn.clicked.connect(submit_offer)
        slider_layout.addWidget(submit_btn)
        
        self.action_buttons_layout.addWidget(slider_frame)
        self.action_buttons_layout.addStretch()

    def _present_cow_trade_response(self, valid_actions: List[GameAction], game: Game):
        """Present cow trade response UI."""
        initiator_name = game.players[game.trade_initiator].name if game.trade_initiator is not None else "Unknown"
        animal_name = game.trade_animal_type.display_name if game.trade_animal_type else "Unknown"
        emoji = ANIMAL_EMOJIS.get(animal_name, "")
        card_count = game.trade_offer_card_count
        
        self.action_desc_label.setText(
            f"🛡️ {initiator_name} offers {card_count} cards for your {emoji} {animal_name}. Counter-offer:"
        )
        
        counter_actions = [a for a in valid_actions if a.type == ActionType.COUNTER_OFFER]
        counter_actions.sort(key=lambda x: x.amount)
        
        max_counter = counter_actions[-1].amount if counter_actions else 0
        
        # Quick counter buttons
        quick_frame = QWidget()
        quick_layout = QHBoxLayout(quick_frame)
        
        for amount in [0, 10, 50, 100]:
            if amount <= max_counter:
                action = next((a for a in counter_actions if a.amount == amount), None)
                if action:
                    btn = self._create_action_button(f"💰 {amount}", action, "#4CAF50")
                    btn.setMinimumWidth(60)
                    quick_layout.addWidget(btn)
        
        self.action_buttons_layout.addWidget(quick_frame)
        
        # Slider
        slider_frame = QFrame()
        slider_layout = QVBoxLayout(slider_frame)
        
        slider_label = QLabel(f"Custom Counter (0 - {max_counter}):")
        slider_layout.addWidget(slider_label)
        
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max_counter)
        slider.setValue(0)
        slider.setTickInterval(10)
        
        value_label = QLabel("Selected: 0")
        
        def on_slider_change(value):
            rounded = round(value / 10) * 10
            value_label.setText(f"Selected: {rounded}")
        
        slider.valueChanged.connect(on_slider_change)
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        
        # Store references
        self._current_slider = slider
        self._current_counter_actions = counter_actions
        
        def submit_counter():
            amount = round(self._current_slider.value() / 10) * 10
            for action in self._current_counter_actions:
                if action.amount == amount:
                    self.gui_agent.submit_action(action)
                    return
            closest = min(self._current_counter_actions, key=lambda a: abs(a.amount - amount))
            self.gui_agent.submit_action(closest)
        
        submit_btn = QPushButton("📤 Counter Offer")
        submit_btn.setMinimumHeight(40)
        submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        submit_btn.clicked.connect(submit_counter)
        slider_layout.addWidget(submit_btn)
        
        self.action_buttons_layout.addWidget(slider_frame)
        self.action_buttons_layout.addStretch()

    def _on_game_over(self, scores: Dict[int, int]):
        """Handle game over."""
        self._clear_action_buttons()
        
        self._add_log_message("=" * 40)
        self._add_log_message("🏁 GAME OVER!")
        self._add_log_message("=" * 40)
        
        # Title
        title = QLabel("🏆 Game Over! 🏆")
        title.setFont(QFont("Helvetica", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_buttons_layout.addWidget(title)
        
        self.action_buttons_layout.addSpacing(20)
        
        # Sort scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        winner_id = sorted_scores[0][0]
        
        # Results
        for rank, (player_id, score) in enumerate(sorted_scores, 1):
            player_name = self.env.agents[player_id].name
            emoji = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
            
            label = QLabel(f"{emoji} #{rank}: {player_name} - {score} points")
            label.setFont(QFont("Helvetica", 14 if rank == 1 else 12, 
                               QFont.Weight.Bold if rank == 1 else QFont.Weight.Normal))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.action_buttons_layout.addWidget(label)
            
            self._add_log_message(f"{emoji} #{rank}: {player_name} - {score} points")
        
        self.action_buttons_layout.addSpacing(20)
        
        # Win/lose message
        if winner_id == 0:
            result_text = "🎉 Congratulations! You won! 🎉"
        else:
            result_text = f"Better luck next time! {self.env.agents[winner_id].name} won."
        
        result_label = QLabel(result_text)
        result_label.setFont(QFont("Helvetica", 14, QFont.Weight.Bold))
        result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.action_buttons_layout.addWidget(result_label)
        
        self.action_buttons_layout.addSpacing(20)
        
        # Play again button
        play_again_btn = QPushButton("🔄 Play Again")
        play_again_btn.setMinimumHeight(50)
        play_again_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        play_again_btn.clicked.connect(self._reset_for_new_game)
        self.action_buttons_layout.addWidget(play_again_btn)
        
        self.action_buttons_layout.addStretch()
        
        self.phase_label.setText("Phase: Game Over")

    def _reset_for_new_game(self):
        """Reset the GUI for a new game."""
        # Clear log
        self.log_text.clear()
        
        # Clear opponents
        for widget in self.opponent_widgets:
            for label in widget.values():
                if isinstance(label, QLabel):
                    label.deleteLater()
        self.opponent_widgets.clear()
        
        while self.opponents_layout.count():
            item = self.opponents_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Reset labels
        self.player_money_label.setText("💰 Money: -")
        self.player_cards_label.setText("💳 Cards: -")
        self.your_animals_label.setText("None")
        self.context_info_label.setText("")
        
        # Show start screen
        self._show_start_screen()

    def _add_log_message(self, message: str):
        """Add a message to the event log."""
        self.log_text.append(message)
        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    def _on_error(self, error_msg: str):
        """Handle error from game thread."""
        self._add_log_message(f"❌ Error: {error_msg}")
        QMessageBox.critical(self, "Error", f"An error occurred: {error_msg}")


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    gui = KuhhandelGUI()
    gui.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
