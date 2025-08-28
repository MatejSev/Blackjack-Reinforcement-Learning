'''Module to implement Card counting Agent'''

import random
import numpy as np
from agents.player import Player

class HiLoAgent(Player):
    '''Represents HiLoAgent, a card-counting player based on the Hi-Lo strategy'''
    def __init__(self, name, chips, base_unit):
        super().__init__(name, chips)
        self.base_unit = base_unit

    def place_bet(self, deck):
        '''Places a bet based on the current count'''
        if self.chips >= 2000:
            return 1

        min_bet = 10
        if self.chips >= min_bet:
            used_cards = deck.get_used_cards()
            unseen_cards = (6 * 52) - len(used_cards)
            count = 0
            for value in used_cards:
                if value in [2, 3, 4, 5, 6]:
                    count += 1
                elif value in [10, 11]:
                    count -= 1

            HiLo = (count / unseen_cards) * 100
            if HiLo < 4:
                bet = self.base_unit
            elif HiLo < 6:
                bet = 2*self.base_unit
            elif HiLo < 8:
                bet = 3*self.base_unit
            elif HiLo < 10:
                bet = 4*self.base_unit
            else:
                bet = 5*self.base_unit
            bet = min(bet, (self.chips // 10) * 10)
            self.bets.append(bet)
            return bet
        else:
            return 0

    def place_insurance(self, deck, dealer_card):
        '''Places an insurance based on the current count'''
        used_cards = deck.get_used_cards()
        unseen_cards = (6 * 52) - len(used_cards)
        count = 0
        for value in used_cards:
            if value in [2, 3, 4, 5, 6]:
                count += 1
            elif value in [10, 11]:
                count -= 1

        HiLo = (count / unseen_cards) * 100
        if HiLo > 8:
            self.insurance_bet = self.bets[0] / 2
            return '1'
        return '2'

    def play_action(self, dealer_card, deck, index, actions):
        '''Performs player actions based on the current count'''
        used_cards = deck.get_used_cards()
        unseen_cards = (6 * 52) - len(used_cards)
        count = 0
        for value in used_cards:
            if value in [2, 3, 4, 5, 6]:
                count += 1
            elif value in [10, 11]:
                count -= 1

        HiLo = (count / unseen_cards) * 100
        
        player_sum = 0
        num_aces = 0
        for card in self.hands[index]:
            if card.rank in ['J', 'Q', 'K']:
                player_sum += 10
            elif card.rank == 'A':
                num_aces += 1
                player_sum += 11
            else:
                player_sum += int(card.rank)
        while player_sum > 21 and num_aces > 0:
            player_sum -= 10
            num_aces -= 1
        
        # Split
        if '4' in actions: 
            if self.hands[index][0].rank == 'A':
                if dealer_card.rank == '7':
                    if HiLo > -33:
                        return '4'
                elif dealer_card.rank == '8':
                    if HiLo > -24:
                        return '4'
                elif dealer_card.rank == '9':
                    if HiLo > -22:
                        return '4'
                elif dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo > -20:
                        return '4'
                elif dealer_card.rank == 'A':
                    if HiLo > -17:
                        return '4'
            elif self.hands[index][0].rank in ['10', 'J', 'Q', 'K']:
                if dealer_card.rank == '2':
                    if HiLo > 25:
                        return '4'
                elif dealer_card.rank == '3':
                    if HiLo > 17:
                        return '4'
                elif dealer_card.rank == '4':
                    if HiLo > 10:
                        return '4'
                elif dealer_card.rank == '5':
                    if HiLo > 6:
                        return '4'
                elif dealer_card.rank == '6':
                    if HiLo > 7:
                        return '4'
                elif dealer_card.rank == '7':
                    if HiLo > 19:
                        return '4'
            elif self.hands[index][0].rank == '9':
                if dealer_card.rank == '2':
                    if HiLo > -3:
                        return '4'
                elif dealer_card.rank == '3':
                    if HiLo > -8:
                        return '4'
                elif dealer_card.rank == '4':
                    if HiLo > -10:
                        return '4'
                elif dealer_card.rank == '5':
                    if HiLo > -15:
                        return '4'
                elif dealer_card.rank == '6':
                    if HiLo > -14:
                        return '4'
                elif dealer_card.rank == '7':
                    if HiLo > 8:
                        return '4'
                elif dealer_card.rank == '8':
                    if HiLo > -16:
                        return '4'
                elif dealer_card.rank == '9':
                    if HiLo > -22:
                        return '4'
                elif dealer_card.rank == 'A':
                    if HiLo > 10:
                        return '4'
            elif self.hands[index][0].rank == '8':
                if dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo < 24:
                        return '4'
                elif dealer_card.rank == 'A':
                    if HiLo > -18:
                        return '4'
            elif self.hands[index][0].rank == '7':
                if dealer_card.rank == '2':
                    if HiLo > -22:
                        return '4'
                elif dealer_card.rank == '3':
                    if HiLo > -29:
                        return '4'
                elif dealer_card.rank == '4':
                    if HiLo > -35:
                        return '4'
            elif self.hands[index][0].rank == '6':
                if dealer_card.rank == '2':
                    if HiLo > 0:
                        return '4'
                elif dealer_card.rank == '3':
                    if HiLo > -3:
                        return '4'
                elif dealer_card.rank == '4':
                    if HiLo > -8:
                        return '4'
                elif dealer_card.rank == '5':
                    if HiLo > -13:
                        return '4'
                elif dealer_card.rank == '6':
                    if HiLo > -16:
                        return '4'
                elif dealer_card.rank == '7':
                    if HiLo > -8:
                        return '4'
            elif self.hands[index][0].rank == '4':
                if dealer_card.rank == '3':
                    if HiLo > 18:
                        return '4'
                elif dealer_card.rank == '4':
                    if HiLo > 8:
                        return '4'
                elif dealer_card.rank == '5':
                    if HiLo > 0:
                        return '4'
            elif self.hands[index][0].rank == '3':
                if dealer_card.rank == '2':
                    if HiLo > -21:
                        return '4'
                elif dealer_card.rank == '3':
                    if HiLo > -34:
                        return '4'
                elif dealer_card.rank == '8':
                    if HiLo > 6 or HiLo < -2:
                        return '4'
            elif self.hands[index][0].rank == '2':
                if dealer_card.rank == '2':
                    if HiLo > -9:
                        return '4'
                elif dealer_card.rank == '3':
                    if HiLo > -15:
                        return '4'
                elif dealer_card.rank == '4':
                    if HiLo > -22:
                        return '4'
                elif dealer_card.rank == '5':
                    if HiLo > -30:
                        return '4'
        # Hard Hand
        if not 'A' in self.hands[index]:
            if player_sum >= 18:
                return '2'
            elif player_sum == 17:
                if dealer_card.rank == 'A':
                    if HiLo > -15:
                        return '2'
                    else:
                        return '1'
                else:
                    return '2'
            elif player_sum == 16:
                if dealer_card.rank == '2':
                    if HiLo > -21:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > -25:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -30:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -34:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -35:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > 10:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '8':
                    if HiLo > 11:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '9':
                    if HiLo > 6:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo > 2:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == 'A':
                    if HiLo > 14:
                        return '2'
                    else:
                        return '1'
            elif player_sum == 15:
                if dealer_card.rank == '2':
                    if HiLo > -12:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > -17:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -21:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -26:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -28:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > 13:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '8':
                    if HiLo > 15:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '9':
                    if HiLo > 12:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo > 8:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == 'A':
                    if HiLo > 16:
                        return '2'
                    else:
                        return '1'
            elif player_sum == 14:
                if dealer_card.rank == '2':
                    if HiLo > -5:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > -8:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -13:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -17:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -17:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > 20:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '8':
                    if HiLo > 38:
                        return '2'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 13:
                if dealer_card.rank == '2':
                    if HiLo > 1:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > -2:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -5:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -9:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -8:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > 50:
                        return '2'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 12:
                if dealer_card.rank == '2':
                    if HiLo > 14:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > 6:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > 2:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -1:
                        return '2'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > 0:
                        return '2'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 11:
                if dealer_card.rank == '2':
                    if HiLo > -23 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > -26 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -29 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -33 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -35 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > -26 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '8':
                    if HiLo > -16 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '9':
                    if HiLo > -10 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo > -9 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == 'A':
                    if HiLo > -3 and '3' in actions:
                        return '3'
                    else:
                        return '1'
            elif player_sum == 10:
                if dealer_card.rank == '2':
                    if HiLo > -15 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > -17 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -21 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -24 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -26 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > -17 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '8':
                    if HiLo > -9 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '9':
                    if HiLo > -3 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo > 7 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == 'A':
                    if HiLo > 6 and '3' in actions:
                        return '3'
                    else:
                        return '1'
            elif player_sum == 9:
                if dealer_card.rank == '2':
                    if HiLo > 3 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '3':
                    if HiLo > 0 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > -5 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > -10 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > -12 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > 4 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '8':
                    if HiLo > 14 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 8:
                if dealer_card.rank == '3':
                    if HiLo > 22 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > 11 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > 5 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > 5 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '7':
                    if HiLo > 22 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 7:
                if dealer_card.rank == '3':
                    if HiLo > 45 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '4':
                    if HiLo > 21 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > 14 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > 17 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 6:
                if dealer_card.rank == '4':
                    if HiLo > 27 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '5':
                    if HiLo > 18 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > 24 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                else:
                    return '1'
            elif player_sum == 5:
                if dealer_card.rank == '5':
                    if HiLo > 20 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                elif dealer_card.rank == '6':
                    if HiLo > 26 and '3' in actions:
                        return '3'
                    else:
                        return '1'
                else:
                    return '1'
            else:
                return '1'
        # Soft Hand
        if 'A' in self.hands[index]:
            if len(self.hands[index]) == 2:
                if self.hands[index][0].rank == '9' or self.hands[index][1].rank == '9':
                    if dealer_card.rank == '3':
                        if HiLo > 20 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > 12 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > 8 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > 8 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '8' or self.hands[index][1].rank == '8':
                    if dealer_card.rank == '3':
                        if HiLo > 9 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > 5 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > 1 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > 0 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '7' or self.hands[index][1].rank == '7':
                    if dealer_card.rank == '3':
                        if HiLo > -2 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > -15 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > -18 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > -23 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '6' or self.hands[index][1].rank == '6':
                    if dealer_card.rank == '2':
                        if HiLo > 1 and HiLo < 10 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '3':
                        if HiLo > -8 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > -14 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > -28 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > -30 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '5' or self.hands[index][1].rank == '5':
                    if dealer_card.rank == '3':
                        if HiLo > 21 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > -6 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > -16 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > -32 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '4' or self.hands[index][1].rank == '4':
                    if dealer_card.rank == '3':
                        if HiLo > 19 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > -7 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > -16 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > -23 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '3' or self.hands[index][1].rank == '3':
                    if dealer_card.rank == '3':
                        if HiLo > 11 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > -8 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > -13 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > -19 and '3' in actions:
                            return '3'
                elif self.hands[index][0].rank == '2' or self.hands[index][1].rank == '2':
                    if dealer_card.rank == '3':
                        if HiLo > 10 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '4':
                        if HiLo > 2 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '5':
                        if HiLo > -19 and '3' in actions:
                            return '3'
                    elif dealer_card.rank == '6':
                        if HiLo > -13 and '3' in actions:
                            return '3'
            if player_sum >= 19:
                return '2'
            elif player_sum == 18:
                if dealer_card.rank == '9':
                    return '1'
                elif dealer_card.rank in ['10', 'J', 'Q', 'K']:
                    if HiLo > 12:
                        return '1'
                    else:
                        return '2'
                elif dealer_card.rank == 'A':
                    if HiLo > -6:
                        return '1'
                    else:
                        return '2'
                else:
                    return '2'
            elif player_sum == 17:
                if dealer_card.rank == '7':
                    if HiLo > 29:
                        return '1'
                    else:
                        return '2' 
                else:
                    return '1' 
            else:
                return '1'
