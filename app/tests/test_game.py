import pytest
from main import BlackjackGame
from src.dealer import Dealer
from agents.player import Player
from src.Deck import Deck, Card

class DummyPlayer(Player):
    def __init__(self, name="TestPlayer", chips=1000):
        super().__init__(name, chips)
        self.actions_to_play = []
        self.bet_sequence = []
        self.bet_index = 0

    def display_cards(self, screen):
        pass  # Přeskočíme vykreslování
    
    def play_action(self, dealer_card, deck, index, actions):
        return self.actions_to_play.pop(0) if self.actions_to_play else '2'

    def place_bet(self, deck):
        if self.bet_index < len(self.bet_sequence):
            bet = self.bet_sequence[self.bet_index]
            self.bet_index += 1
            return bet
        return -1

class DummyDealer(Dealer):
    def display_cards(self, screen=None, visible=False):
        pass 

class DummyGame(BlackjackGame):
    def __init__(self):
        super().__init__()
        self.screen = None
        self.dealer = DummyDealer()
        self.deck = Deck()
        self.players = [DummyPlayer(name="Test player", chips=1000)]
    
    def display_message(self, message, color=(255,255,255)):
        pass

    def display_table(self):
        pass
    
    def get_bet(self):
        pass

@pytest.fixture
def game_with_players():
    game = BlackjackGame()
    game.players = [DummyPlayer(name="Alice", chips=1000), DummyPlayer(name="Bob", chips=1000)]
    game.dealer = DummyDealer()
    return game

def test_initial_cards_dealt(game_with_players):
    game = game_with_players
    game.deal_initial_cards()
    for player in game.players:
        assert len(player.hands[0]) == 2, f"{player.name} should have 2 cards"
    assert len(game.dealer.hand) == 2, "Dealer should have 2 cards"

@pytest.fixture
def game():
    return DummyGame()

def test_player_turn_stand(game):
    player = game.players[0]
    player.hands = [[Card('10', 'Hearts'), Card('6', 'Spades')]]
    player.bets = [10]
    game.dealer.receive_card(Card('7', 'Clubs'))  # dealer upcard
    player.actions_to_play = ['2']  # stand
    game.player_turn(player)
    assert len(player.hands[0]) == 2  # stále 2 karty

def test_player_turn_hit(game):
    player = game.players[0]
    player.hands = [[Card('5', 'Hearts'), Card('6', 'Spades')]]
    player.bets = [10]
    game.dealer.receive_card(Card('7', 'Clubs'))
    player.actions_to_play = ['1', '2']  # hit, pak stand
    game.player_turn(player)
    assert len(player.hands[0]) == 3  # měla by mít 3 karty po hitu

def test_player_turn_double_down(game):
    player = game.players[0]
    player.hands = [[Card('5', 'Hearts'), Card('6', 'Spades')]]
    player.bets = [10]
    player.chips = 1000
    game.dealer.receive_card(Card('7', 'Clubs'))
    player.actions_to_play = ['3']  # double down
    game.player_turn(player)
    assert player.bets[0] == 20  # sázka zdvojnásobená
    assert len(player.hands[0]) == 3  # jedna další karta přidána

def test_player_turn_split(game):
    player = game.players[0]
    player.hands = [[Card('8', 'Hearts'), Card('8', 'Spades')]]
    player.bets = [10]
    player.chips = 1000
    game.dealer.receive_card(Card('7', 'Clubs'))
    player.actions_to_play = ['4', '2', '2']  
    # 4 = split, poté stand na první ruku, stand na druhou
    game.player_turn(player)
    assert len(player.hands) == 2  # dvě ruce po splitu
    assert all(len(hand) >= 2 for hand in player.hands)  # každá ruka má karty

def test_place_valid_bet(game):
    player = game.players[0]
    player.bet_sequence = [50]
    player.bets = []
    game.place_bets()
    assert player.bets == [50]
    assert player in game.players

def test_place_invalid_bet_then_valid(game):
    player = game.players[0]
    player.bet_sequence = [5, 50]  # 5 je invalid (menší než 10), 50 valid
    player.bets = []
    game.place_bets()
    assert player.bets == [50]
    assert player in game.players