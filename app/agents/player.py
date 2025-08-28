'''Module representing a player in Blackjack'''

import pygame

class Player:
    '''Represents a player in Blackjack'''

    def __init__(self, name, chips):
        '''Initialize player with a name, chips, empty hands, and bets'''
        self.name = name
        self.hands = [[]]
        self.chips = chips
        self.bets = []
        self.insurance_bet = 0
        self.params = None

    def set_params(self, params):
        '''Add params for GUI to player'''
        self.params = params

    def receive_card(self, card, index=0):
        '''Add a card to the specified hand'''
        self.hands[index].append(card)

    def clear_hand(self):
        '''Clear all hands, bets, and insurance bet'''
        self.hands = [[]]
        self.bets = []
        self.insurance_bet = 0

    def display_hand(self):
        '''Return a list of cards in all hands'''
        return [[f"{card.rank} of {card.suit}" for card in hand] for hand in self.hands]

    def place_bet(self, deck):
        '''Function for placing a bet; to be implemented by agents'''
        return -1

    def place_insurance(self, deck, dealer_card):
        '''Function for placing insurance bet; to be implemented by agents'''
        return -1

    def play_action(self, dealer_card, deck, index, actions):
        '''Function for playing an action; to be implemented by agents'''
        return -1

    def get_reward(self, reward, dealer_card, deck, index, actions):
        '''Process the reward or penalty (for agents)'''
        pass

    def display_cards(self, screen):
        '''Display cards on the pygame screen'''
        x_off, y_off = self.params[0]
        angle = self.params[1]
        a, b = self.params[2]
        split_offsets = self.params[3] if len(self.hands) == 2 else [(0, 0)]
        for (j, k), hand in zip(split_offsets, self.hands):
            for i, card in enumerate(hand):
                path = f'images\\Cards\\Classic\\{card.suit}{card.rank}.png'
                card_image = pygame.image.load(path)
                card_image = pygame.transform.scale_by(card_image, 0.1)
                card_image = pygame.transform.rotate(card_image, angle)
                x = i * a
                y = i * b
                screen.blit(card_image, (736 + x + j + x_off, 590 + y + k + y_off))
        pygame.display.flip()

    def display_balance(self):
        '''Display player's balance on the screen'''
        x_off, y_off = self.params[4]
        angle = self.params[1]
        font = pygame.font.SysFont('Arial', 24)
        balance_text = font.render(f'{self.name}: ${self.chips}', True, (255, 255, 255))
        balance_text = pygame.transform.rotate(balance_text, angle)
        screen_surface = pygame.display.get_surface()
        screen_surface.blit(balance_text, (688 + x_off, 750 + y_off))

    def display_bet(self):
        '''Display player's bet(s) on the screen'''
        x_off, y_off = self.params[5]
        angle = self.params[1]
        if self.bets:
            offsets = self.params[3] if len(self.hands) == 2 else [(0, 0)]
            font = pygame.font.SysFont('Arial', 24, bold=True)
            screen_surface = pygame.display.get_surface()
            for (j, k), bet in zip(offsets, self.bets):
                bet_text = font.render(f'Bet: ${bet}', True, (255, 255, 255))
                bet_text = pygame.transform.rotate(bet_text, angle)
                screen_surface.blit(bet_text, (740 + x_off + j, 550 + y_off + k))
