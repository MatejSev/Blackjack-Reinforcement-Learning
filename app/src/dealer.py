'''Module representing the dealer in the card game'''

import pygame

class Dealer:
    '''Represents the dealer'''

    def __init__(self):
        self.hand = []

    def receive_card(self, card):
        '''Add a card to dealer's hand'''
        self.hand.append(card)

    def clear_hand(self):
        '''Clear dealer's hand for a new round'''
        self.hand = []

    def display_partial_hand(self):
        '''Show only first dealer card'''
        if self.hand:
            return [f"{self.hand[0].rank} of {self.hand[0].suit}", 'Unknown']
        return []

    def display_hand(self):
        '''Show all cards in dealer's hand'''
        return [f"{card.rank} of {card.suit}" for card in self.hand]

    def get_dealer_card(self, index=0):
        '''Return dealer's card or None if empty'''
        if self.hand:
            return self.hand[index]
        return None

    def display_cards(self, screen, visible=False):
        '''Display dealer cards on the pygame screen'''
        for i, card in enumerate(self.hand):
            if not visible and i == 1:
                path = r'images\Backs\Card-Back-04.png'
            else:
                path = rf'images\Cards\Classic\{card.suit}{card.rank}.png'
            card_image = pygame.image.load(path)
            card_image = pygame.transform.scale_by(card_image, 0.1)
            x = i * 20
            screen.blit(card_image, (736 + x, 190))
        pygame.display.flip()
