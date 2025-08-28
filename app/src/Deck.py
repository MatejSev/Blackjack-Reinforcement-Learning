'''Module representing cards and deck for Blackjack'''

import random
from sortedcontainers import SortedList

class Card:
    '''Represents a single card with rank and suit'''

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

class Deck:
    '''Represents a deck of multiple standard 52-card decks combined'''

    def __init__(self, num_decks=6):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.cards = [Card(rank, suit) for rank in ranks for suit in suits for _ in range(num_decks)]
        self.used_cards = SortedList()
        random.shuffle(self.cards)

    def calculate_value(self, card):
        '''Return the Blackjack value of a card'''
        rank = card.rank
        if rank.isdigit():
            return int(rank)
        if rank in ['K', 'Q', 'J']:
            return 10
        if rank == 'A':
            return 11
        return 0

    def add_to_used_cards(self, card):
        '''Add the card's value to the used_cards list'''
        card_value = self.calculate_value(card)
        self.used_cards.add(card_value)

    def draw_card(self, show_card=True):
        '''Draw a card from the deck'''
        card = self.cards.pop()
        if show_card:
            self.add_to_used_cards(card)
        return card

    def remaining_cards(self):
        '''Return number of cards left in the deck'''
        return len(self.cards)

    def reshuffle(self):
        '''Reshuffle the deck'''
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
        self.cards = [Card(rank, suit) for rank in ranks for suit in suits for _ in range(6)]
        self.used_cards.clear()
        random.shuffle(self.cards)

    def get_used_cards(self):
        '''Return the list of used cards'''
        return self.used_cards
