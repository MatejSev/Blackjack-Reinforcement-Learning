'''Module to implement Q-learning Agent'''

import random
import numpy as np
from collections import defaultdict
from sortedcontainers import SortedList
from agents.player import Player
import lmdb
import os
import hashlib
import msgpack
import math

class QLearningAgent(Player):
    '''Represents an agent that uses Q-learning'''
    def __init__(self, name, chips, db_path="q_learning_HiLo.lmdb",
                alpha_betting=0.1, gamma_betting=0.9, epsilon_betting=0, epsilon_decay_betting=0.995, epsilon_min_betting=0.01,
                alpha_insurance=0.1, gamma_insurance=0.9, epsilon_insurance=0, epsilon_decay_insurance=0.995, epsilon_min_insurance=0.01,
                alpha_playing=0.1, gamma_playing=0.9, epsilon_playing=0, epsilon_decay_playing=0.995, epsilon_min_playing=0.01):
        super().__init__(name, chips)
        self.alpha_betting = alpha_betting
        self.gamma_betting = gamma_betting
        self.epsilon_betting = epsilon_betting
        self.alpha_insurance = alpha_insurance
        self.gamma_insurance = gamma_insurance
        self.epsilon_insurance = epsilon_insurance
        self.epsilon_decay_betting = epsilon_decay_betting
        self.epsilon_min_betting = epsilon_min_betting
        self.epsilon_decay_insurance = epsilon_decay_insurance
        self.epsilon_min_insurance = epsilon_min_insurance
        self.epsilon_decay_playing = epsilon_decay_playing
        self.epsilon_min_playing = epsilon_min_playing
        self.alpha_playing = alpha_playing
        self.gamma_playing = gamma_playing
        self.epsilon_playing = epsilon_playing
        self.choice = None
        self.choices = []
        self.insurance_choice = None
        self.bet_state = None
        self.play_state = None
        self.insurance_state = None
        self.env = lmdb.open(db_path, map_size=20*1024*1024*1024, sync=False, writemap=True)
        self.num_rounds = 0

    def clear_hand(self):
        '''Resets the agent's hand'''
        super().clear_hand()
        self.choices = []
        self.num_rounds += 1

    def _get_q_value(self, phase, state, action):
        '''Retrieves the Q-value for a given phase, state, and action from the LMDB database'''
        with self.env.begin() as txn:
            raw_key = f"{phase}_{state}_{action}"
            key = hashlib.md5(raw_key.encode()).digest()
            value = txn.get(key)
            return msgpack.unpackb(value, raw=False) if value else 0.0

    def _set_q_value(self, phase, state, action, value):
        '''Stores the Q-value for a given phase, state, and action in the LMDB database'''
        with self.env.begin(write=True) as txn:
            raw_key = f"{phase}_{state}_{action}"
            key = hashlib.md5(raw_key.encode()).digest()
            txn.put(key, msgpack.packb(value, use_bin_type=True))

    # Betting phase
    def get_betting_state(self, used_cards):
        '''Creates a betting state'''
        chips = (self.chips // 100) * 100
        return (chips, used_cards)

    def choose_bet_action(self, state):
        '''Selects an action for betting based on the current state'''
        if self.chips >= 2000:
            return 1

        min_bet = 100
        chips = int(self.chips)
        max_bet = 500 if chips >= 500 else chips - (chips % 10)

        if self.chips < min_bet:
            return 0

        possible_bets = list(range(min_bet, max_bet + 1, 10))

        if random.uniform(0, 1) < self.epsilon_betting:
            return random.choice(possible_bets)
        else:
            return max(possible_bets, key=lambda bet: self._get_q_value("betting", state, bet))

    def update_q_table_betting(self, state, action, reward, next_state):
        '''Update Q-value for betting phase using Q-learning function'''
        old_value = self._get_q_value("betting", state, action)
        next_chips = int(next_state[0])
        max_bet = 500 if next_chips >= 500 else next_chips - (next_chips % 10)
        possible_bets = list(range(10, max_bet + 1, 10))
        future_value = max((self._get_q_value("betting", next_state, a) for a in possible_bets), default=0)
        k = -math.log(0.1) / 50
        penalization = math.exp(-k * self.num_rounds)
        penalized_reward = penalization * reward
        new_value = old_value + self.alpha_betting * (penalized_reward + self.gamma_betting * future_value - old_value)
        self._set_q_value("betting", state, action, new_value)
        self.epsilon_betting = max(self.epsilon_min_betting, self.epsilon_betting * self.epsilon_decay_betting)

    def place_bet(self, deck):
        '''Agent places bet'''
        used_cards = deck.get_used_cards()
        used_cards_count = 0
        for value in used_cards:
            if value in [2, 3, 4, 5, 6]:
                used_cards_count += 1
            elif value in [10, 11]:
                used_cards_count -= 1
        self.bet_state = self.get_betting_state(used_cards_count)
        bet = self.choose_bet_action(self.bet_state)
        self.bets.append(bet)
        self.choices.append(bet)
        return bet

    # Insurance phase
    def get_insurance_state(self, player_sum, num_aces, dealer_sum, used_cards):
        '''Creates a insurance state'''
        chips = (self.chips // 100) * 100
        return (player_sum, num_aces, dealer_sum, used_cards, self.bets[0], chips)

    def choose_insurance_action(self, state):
        '''Selects an action for insurance based on the current state'''
        if random.uniform(0, 1) < self.epsilon_insurance:
            return random.choice(['1','2'])
        else:
            return max(['1','2'], key=lambda action: self._get_q_value("insurance", state, action))
    
    def update_q_table_insurance(self, state, action, reward, next_state):
        '''Update Q-value for insurance phase using Q-learning function'''
        old_value = self._get_q_value("insurance", state, action)
        future_value = max(self._get_q_value("insurance", next_state, a) for a in ['1','2'])
        k = -math.log(0.1) / 50
        penalization = math.exp(-k * self.num_rounds)
        penalized_reward = penalization * reward
        new_value = old_value + self.alpha_betting * (penalized_reward + self.gamma_betting * future_value - old_value)
        self._set_q_value("insurance", state, action, new_value)
        self.epsilon_insurance = max(self.epsilon_min_insurance, self.epsilon_insurance * self.epsilon_decay_insurance)

    def place_insurance(self, deck, dealer_card):
        '''Agent places insurance'''
        player_sum = 0
        num_aces = 0
        for card in self.hands[0]:
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
        
        if dealer_card.rank in ['J', 'Q', 'K']:
            dealer_sum = 10
        elif dealer_card.rank == 'A':
            dealer_sum = 11
        else:
            dealer_sum = int(dealer_card.rank)
        used_cards = deck.get_used_cards()
        used_cards_count = 0
        for value in used_cards:
            if value in [2, 3, 4, 5, 6]:
                used_cards_count += 1
            elif value in [10, 11]:
                used_cards_count -= 1
        self.insurance_state = self.get_insurance_state(player_sum, num_aces, dealer_sum, used_cards_count)
        self.insurance_choice = self.choose_insurance_action(self.insurance_state)
        if self.insurance_choice == 1:
            self.insurance_bet = self.bets[0] / 2
        return self.insurance_choice

    # Playing phase
    def get_playing_state(self, player_sum, num_aces, dealer_sum, used_cards, index):
        '''Creates a playing state'''
        chips = (self.chips // 100) * 100
        return (player_sum, num_aces, dealer_sum, used_cards, self.bets[index], chips)

    def choose_play_action(self, state, actions):
        '''Selects an action based on the current state'''
        if random.uniform(0, 1) < self.epsilon_playing:
            return random.choice(actions)
        else:
            return max(actions, key=lambda action: self._get_q_value("playing", state, action))

    def update_q_table_playing(self, state, action, reward, next_state, actions):
        '''Update Q-value for playing phase using Q-learning function'''
        old_value = self._get_q_value("playing", state, action)
        future_value = max(self._get_q_value("playing", next_state, a) for a in actions)
        k = -math.log(0.1) / 50
        penalization = math.exp(-k * self.num_rounds)
        penalized_reward = penalization * reward
        new_value = old_value + self.alpha_betting * (penalized_reward + self.gamma_betting * future_value - old_value)
        self._set_q_value("playing", state, action, new_value)
        self.epsilon_playing = max(self.epsilon_min_playing, self.epsilon_playing * self.epsilon_decay_playing)

    def play_action(self, dealer_card, deck, index, actions):
        '''Agent plays action'''
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
        
        if dealer_card.rank in ['J', 'Q', 'K']:
            dealer_sum = 10
        elif dealer_card.rank == 'A':
            dealer_sum = 11
        else:
            dealer_sum = int(dealer_card.rank)
        used_cards = deck.get_used_cards()
        used_cards_count = 0
        for value in used_cards:
            if value in [2, 3, 4, 5, 6]:
                used_cards_count += 1
            elif value in [10, 11]:
                used_cards_count -= 1
        self.play_state = self.get_playing_state(player_sum, num_aces, dealer_sum, used_cards_count, index)
        self.choice = self.choose_play_action(self.play_state, actions)
        if self.choice == '4' and '4' in actions:
            self.choices.append(self.choices[index])
        return self.choice

    def get_reward(self, reward, dealer_card, deck, index, actions):
        '''Receives the reward and updates Q-values accordingly'''
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
        
        if dealer_card.rank in ['J', 'Q', 'K']:
            dealer_sum = 10
        elif dealer_card.rank == 'A':
            dealer_sum = 11
        else:
            dealer_sum = int(dealer_card.rank)
        used_cards = deck.get_used_cards()
        used_cards_count = 0
        for value in used_cards:
            if value in [2, 3, 4, 5, 6]:
                used_cards_count += 1
            elif value in [10, 11]:
                used_cards_count -= 1
        next_playing_state = self.get_playing_state(player_sum, num_aces, dealer_sum, used_cards_count, index)

        if reward == 0:
            self.update_q_table_playing(self.play_state, self.choice, reward, next_playing_state, actions)
        else:
            if self.insurance_choice != None:
                reward = reward - self.insurance_bet
                insurance_reward = self.insurance_bet
                next_insurance_state = self.get_insurance_state(player_sum, num_aces, dealer_sum, used_cards_count)
                self.update_q_table_insurance(self.insurance_state, self.insurance_choice, insurance_reward, next_insurance_state)
                self.insurance_choice = None
            if self.choice != None:
                self.update_q_table_playing(self.play_state, self.choice, reward, next_playing_state, actions)
            next_betting_state = self.get_betting_state(used_cards_count)
            self.update_q_table_betting(self.bet_state, self.choices[index], reward, next_betting_state)
