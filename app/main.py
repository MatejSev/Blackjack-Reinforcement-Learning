'''Module implements main game of Blackjack'''

import pygame
import time
from src.dealer import Dealer
from src.Deck import Deck
from gui.InputBox import InputBox
from gui.Button import Button
from gui.Circlebutton import CircleButton
from agents.player import Player
from agents.qlearning import QLearningAgent
from agents.dqn import DQNagent
from agents.card_counting import HiLoAgent
import sys
import json
import os

class BlackjackGame:
    '''Represents the Blackjack game'''
    def __init__(self):
        self.num_players = 0
        self.players = []
        self.dealer = Dealer()
        self.deck = Deck()
        self.actions = ['1', '2']
        self.filepath = "src/saved_players.json"
        pygame.init()
        self.screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
        pygame.display.set_caption('Blackjack')

    def set_player_params(self):
        if self.num_players == 1:
            split_offsets = [(-50, 0), (50, 0)]
            params = ((0, 0), 0, (20, 0), split_offsets, (0, 0), (0, 0))
            self.players[0].set_params(params)
        elif 2 <= self.num_players <= 3:
            split_offsets = [[(-50, -35), (35, 15)], [(-50, 0), (50, 0)], [(-35, 15), (50, -35)]][:self.num_players]
            angles = [-30, 0, 30][:self.num_players]
            offsets = [(-220, -60), (0, 0), (180, -50)][:self.num_players]
            shifts = [(15, 10), (20, 0), (15, -10)][:self.num_players]
            balance_offsets = [(-250, -105), (0, 0), (280, -110)][:self.num_players]
            bet_offsets = [(-220, -60), (0, 0), (180, -50)][:self.num_players]
            for i in range(self.num_players):
                params = (offsets[i], angles[i], shifts[i], split_offsets[i], balance_offsets[i], bet_offsets[i])
                self.players[i].set_params(params)        
        else:
            split_offsets = [[(-30, -55), (25, 35)], [(-50, -35), (35, 15)], [(-50, 0), (50, 0)], [(-35, 15), (50, -35)], [(-15, 30), (35, -60)]][:self.num_players]
            angles = [-60, -30, 0, 30, 60][:self.num_players]
            offsets = [(-380, -195), (-220, -60), (0, 0), (180, -50), (330, -185)][:self.num_players]
            shifts = [(10, 15), (15, 10), (20, 0), (15, -10), (10, -15)][:self.num_players]
            balance_offsets = [(-420, -300), (-250, -105), (0, 0), (280, -110), (500, -310)][:self.num_players]
            bet_offsets = [(-280, -170), (-170, -55), (0, 0), (150, -55), (290, -175)][:self.num_players]
            for i in range(self.num_players):
                params = (offsets[i], angles[i], shifts[i], split_offsets[i], balance_offsets[i], bet_offsets[i])
                self.players[i].set_params(params)
    
    def display_menu(self):
        screen_width, screen_height = self.screen.get_size()
        background_image = pygame.image.load('images\\blackjack_table.jpg')
        bg_width, bg_height = background_image.get_size()
        screen_ratio = screen_width / screen_height
        bg_ratio = bg_width / bg_height
        if screen_ratio > bg_ratio:
            new_width = screen_width
            new_height = int(screen_width / bg_ratio)
        else:
            new_height = screen_height
            new_width = int(screen_height * bg_ratio)        
        background_image = pygame.transform.scale(background_image, (new_width, new_height))
        blur_factor = 5
        small_width = new_width // blur_factor
        small_height = new_height // blur_factor
        small_surface = pygame.transform.scale(background_image, (small_width, small_height))
        blurred_background = pygame.transform.scale(small_surface, (new_width, new_height))
        bg_x_pos = (screen_width - new_width) // 2
        bg_y_pos = (screen_height - new_height) // 2
        self.screen.blit(blurred_background, (bg_x_pos, bg_y_pos))
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(100)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        title_font = pygame.font.Font(None, 96)
        title_text = title_font.render("MENU", True, (255, 215, 0))
        title_rect = title_text.get_rect(center=(screen_width // 2 + 35, screen_height // 4 - 30))
        shadow_text = title_font.render("MENU", True, (0, 0, 0))
        shadow_rect = shadow_text.get_rect(center=(screen_width // 2 + 3 + 35, screen_height // 4 + 3 - 30))
        self.screen.blit(shadow_text, shadow_rect)
        self.screen.blit(title_text, title_rect)
        start_button = Button(x=700, y=300, w=200, h=60, text='Start Game', color=(0, 0, 0), text_color=(255, 255, 255))
        quit_button = Button(x=700, y=400, w=200, h=60, text='Quit Game', color=(0, 0, 0), text_color=(255, 255, 255))
        font = pygame.font.Font(None, 40)
        add_label = font.render("Add player", True, (0, 0, 200))
        remove_label = font.render("Remove player", True, (200, 0, 0))
        add_player = InputBox(x=585, y=515, w=200, h=60, active_color=(0,0,200))
        remove_player = InputBox(x=815, y=515, w=200, h=60, active_color=(200,0,0))
        added = removed = True
        running = True
        while running:
            for event in pygame.event.get():
                if not start_button.handle_event(event):
                    if self.num_players != 0:
                        self.set_player_params()
                        running = False
                if not quit_button.handle_event(event):
                    pygame.quit()
                    sys.exit()
                added = add_player.handle_event(event)
                if not added:
                    if self.num_players < 5:
                        name = str(add_player.get_text())
                        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
                            with open(self.filepath, 'r') as f:
                                list_players = json.load(f)
                        else:
                            list_players = []
                        player_data = next((p for p in list_players if p['name'] == name), None)
                        self.num_players += 1 
                        is_player = Button(x=430, y=600, w=170, h=60, text='Player', color=(170, 0, 0), text_color=(255, 255, 255))
                        is_dqn = Button(x=620, y=600, w=170, h=60, text='DQN', color=(0, 0, 170), text_color=(255, 255, 255))
                        is_qlearning = Button(x=810, y=600, w=170, h=60, text='Qlearning', color=(0, 170, 0), text_color=(255, 255, 255))
                        is_counting = Button(x=1000, y=600, w=170, h=60, text='Card Counting', color=(170, 170, 0), text_color=(255, 255, 255))    
                        if player_data:
                            name=player_data['name']
                            chips=player_data['chips']
                        else:
                            chips=1000
                        run = True
                        while run:
                            for event in pygame.event.get():
                                if not is_player.handle_event(event):
                                    self.players.append(Player(name, chips))
                                    run = False
                                if not is_dqn.handle_event(event):
                                    agent = DQNagent(name, chips)
                                    agent.rounds = 0
                                    self.players.append(agent)
                                    run = False
                                if not is_qlearning.handle_event(event):
                                    agent = QLearningAgent(name, chips)
                                    agent.rounds = 0
                                    self.players.append(agent)
                                    run = False
                                if not is_counting.handle_event(event):
                                    agent = HiLoAgent(name, chips, base_unit=100)
                                    agent.rounds = 0
                                    self.players.append(agent)
                                    run = False
                            is_player.draw(self.screen)
                            is_dqn.draw(self.screen)
                            is_qlearning.draw(self.screen)
                            is_counting.draw(self.screen)
                            pygame.display.flip()
                    added = True
                removed = remove_player.handle_event(event)
                if not removed:
                    self.num_players -= 1
                    name_to_remove = str(remove_player.get_text())
                    self.players = [p for p in self.players if p.name != name_to_remove]
                    removed = True
            self.screen.blit(blurred_background, (bg_x_pos, bg_y_pos))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(shadow_text, shadow_rect)
            self.screen.blit(title_text, title_rect)
            title_font = pygame.font.SysFont(None, 50)
            player_font = pygame.font.SysFont(None, 35)
            title_surface = title_font.render("Players", True, (255, 255, 255))
            self.screen.blit(title_surface, (300, 200 - 60))
            for i, p in enumerate(self.players):
                text_surface = player_font.render(p.name, True, (255, 255, 255))
                self.screen.blit(text_surface, (300, 200 + i * 40))
            start_button.draw(self.screen)
            quit_button.draw(self.screen)
            self.screen.blit(add_label, (add_player.rect.x, add_player.rect.y - 30))
            self.screen.blit(remove_label, (remove_player.rect.x, remove_player.rect.y - 30))
            add_player.draw(self.screen)
            remove_player.draw(self.screen)
            pygame.display.flip()
        pygame.display.flip()

    def display_table(self):
        screen_width, screen_height = self.screen.get_size()
        background_image = pygame.image.load('images\\blackjack_table.jpg')
        bg_width, bg_height = background_image.get_size()
        screen_ratio = screen_width / screen_height
        bg_ratio = bg_width / bg_height
        if screen_ratio > bg_ratio:
            new_width = screen_width
            new_height = int(screen_width / bg_ratio)
        else:
            new_height = screen_height
            new_width = int(screen_height * bg_ratio)    
        background_image = pygame.transform.scale(background_image, (new_width, new_height))
        bg_x_pos = (screen_width - new_width) // 2
        bg_y_pos = (screen_height - new_height) // 2        
        self.screen.blit(background_image, (bg_x_pos, bg_y_pos))
        for i, player in enumerate(self.players):
            player.display_balance()
            player.display_bet()
        pygame.display.flip()

    def display_message(self, message, color=(255, 255, 255), duration=1000, visible=False):
        font = pygame.font.Font(None, 40)
        text_surface = font.render(message, True, color)
        text_rect = text_surface.get_rect(center=(self.screen.get_width() // 2, 396))
        self.screen.blit(text_surface, text_rect)
        pygame.display.flip()
        pygame.time.delay(duration)
        self.display_table()
        for i, player in enumerate(self.players):
            player.display_cards(self.screen)
        self.dealer.display_cards(self.screen, visible)

    def deal_initial_cards(self):
        '''Deal two initial cards to each player and the dealer'''
        if self.deck.remaining_cards() < 104:  # 104 is 2 of 6 decks
            self.deck.reshuffle()
        for i in range(2):
            for player in self.players:
                card = self.deck.draw_card()
                player.receive_card(card)
                player.display_cards(self.screen)
                time.sleep(0.5)
            if i == 0:
                card = self.deck.draw_card()
                self.dealer.receive_card(card)
                self.dealer.display_cards(self.screen)
                time.sleep(0.5)
            else:
                show_card = False
                card = self.deck.draw_card(show_card)
                self.dealer.receive_card(card)
                self.dealer.display_cards(self.screen)
                time.sleep(0.5)

    def get_action(self, actions):
        hit_button = Button(558, 336, 200, 50, text='Hit', color=(255,0,0))
        stand_button = Button(778, 336, 200, 50, text='Stand', color=(0,0,255))
        double_button = Button(558, 396, 200, 50, text='Double Down', color=(0,255,0))
        split_button = Button(778, 396, 200, 50, text='Split', color=(255,255,0))
        hit = stand = double = split = False
        running = True
        self.display_table()
        for i, player in enumerate(self.players):
            player.display_cards(self.screen)
        self.dealer.display_cards(self.screen)
        while running:
            for event in pygame.event.get():
                if '1' in actions and hit_button.handle_event(event) == False:
                    hit = True
                    running = False
                if '2' in actions and stand_button.handle_event(event) == False:
                    stand = True
                    running = False
                if '3' in actions and double_button.handle_event(event) == False:
                    double = True
                    running = False
                if '4' in actions and split_button.handle_event(event) == False:
                    split = True
                    running = False
            if '1' in actions:
                hit_button.draw(self.screen)
            if '2' in actions:
                stand_button.draw(self.screen)
            if '3' in actions:
                double_button.draw(self.screen)
            if '4' in actions:
                split_button.draw(self.screen)
            pygame.display.flip()
        self.display_table()
        for i, player in enumerate(self.players):
            player.display_cards(self.screen)
        self.dealer.display_cards(self.screen)
        if hit:
            return '1' 
        elif stand:
            return '2'
        elif double:
            return '3'
        elif split:
            return '4'

    def player_turn(self, player):
        '''Handle the turn of a player'''
        for index in range(len(player.hands)):
            while True:
                dealer_card = self.dealer.get_dealer_card()
                
                # Player has Blackjack
                player_value = self.calculate_hand_value(player.hands[index])
                if len(player.hands[index]) == 2 and player_value == 21:
                    self.display_message(f"{player.name} has Blackjack!")
                    break
                
                self.actions = ['1', '2']
                if player.chips >= 2 * player.bets[index] + player.insurance_bet:
                    self.actions.append('3')

                if len(player.hands) == 1 and len(player.hands[index]) == 2 and (player.hands[index][0].rank == player.hands[index][1].rank or (player.hands[index][0].rank in ['10','J','Q','K'] and player.hands[index][1].rank in ['10','J','Q','K'])):
                    if player.chips >= 2 * player.bets[index] + player.insurance_bet:
                        self.actions.append('4')
                
                choice = player.play_action(dealer_card, self.deck, index, self.actions)
                if choice == -1:
                    choice = self.get_action(self.actions)
                else:
                    self.display_table()
                    for player in self.players:
                        player.display_cards(self.screen)
                    self.dealer.display_cards(self.screen)
                time.sleep(0.5)
                # Hit
                if choice == '1':
                    card = self.deck.draw_card()
                    player.receive_card(card, index)
                    player.display_cards(self.screen)
                    player_value = self.calculate_hand_value(player.hands[index])
                    if player_value > 21:
                        break
                    elif player_value == 21:
                        self.display_message("You have 21. Cannot hit or stand.")
                        break
                    player.get_reward(0, dealer_card, self.deck, index, self.actions)
                # Stand
                elif choice == '2':
                    break
                # Double Down
                elif choice == '3':
                    if '3' in self.actions:
                        player.bets[index] *= 2
                        card = self.deck.draw_card()
                        player.receive_card(card, index)
                        player.display_cards(self.screen)
                        break
                    else:
                        self.display_message("Invalid choice. Please choose again!", color=(255,0,0))
                elif choice == '4':
                    if '4' in self.actions:
                        player.bets.append(player.bets[index])
                        card_to_split = player.hands[index].pop()
                        player.hands.append([card_to_split])
                        player.receive_card(self.deck.draw_card(), index)
                        player.receive_card(self.deck.draw_card(), len(player.hands) - 1)
                        player.display_cards(self.screen)
                        self.player_turn(player)
                        return
                    else:
                        self.display_message("Invalid choice. Please choose again!", color=(255,0,0))
                else:
                    self.display_message("Invalid choice. Please choose again!", color=(255,0,0))

    def dealer_turn(self):
        '''Dealer draws cards until hand value is at least 17'''
        dealer_card = self.dealer.get_dealer_card(1)
        self.deck.add_to_used_cards(dealer_card)
        self.display_table()
        while self.calculate_hand_value(self.dealer.hand) < 17:
            card = self.deck.draw_card()
            self.dealer.receive_card(card)
            self.dealer.display_cards(self.screen, visible=True)
            for player in self.players:
                player.display_cards(self.screen)
            time.sleep(0.5)
        self.dealer.display_cards(self.screen, visible=True)
        for i, player in enumerate(self.players):
            player.display_cards(self.screen)
        time.sleep(1)

    def calculate_hand_value(self, hand):
        '''Calculate the total value of a hand'''
        value = 0
        num_aces = 0

        for card in hand:
            rank = card.rank
            if rank.isdigit():
                value += int(rank)
            elif rank in ['K', 'Q', 'J']:
                value += 10
            elif rank == 'A':
                value += 11
                num_aces += 1

        while value > 21 and num_aces:
            value -= 10
            num_aces -= 1

        return value

    def get_bet(self):
        font = pygame.font.Font(None, 40)
        text_surface = font.render("Enter your bet amount", True, (255, 255, 255))
        input_box = InputBox(668, 416, 200, 32)
        running = True
        while running:
            for event in pygame.event.get():
                running = input_box.handle_event(event)
            self.display_table()
            self.screen.blit(text_surface, (618, 386))
            input_box.draw(self.screen)            
            pygame.display.flip()
        bet = int(input_box.get_text())
        self.display_table()
        return bet

    def place_bets(self):
        '''Ask each player to place their bet or leave the game'''
        for player in self.players[:]:
            while True:
                try:
                    bet = player.place_bet(self.deck)
                    if bet == -1:
                        bet = self.get_bet()
                    else:
                        self.display_table()
                    if bet >= 10 and bet <= 500 and bet % 10 == 0:
                        player.bets.append(bet)
                        break
                    elif bet == 0:
                        self.display_message(f"{player.name} has left the game.")
                        player.clear_hand()
                        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
                            with open(self.filepath, 'r') as f:
                                list_players = json.load(f)
                        else:
                            list_players = []
                        if player.chips >= 10:
                            for i, p in enumerate(list_players):
                                if p['name'] == player.name:
                                    list_players[i] = {'name': player.name, 'chips': player.chips}
                                    break
                            else:
                                list_players.append({'name': player.name, 'chips': player.chips})
                        else:
                            list_players = [p for p in list_players if p['name'] != player.name]
                        with open(self.filepath, 'w') as f:
                            json.dump(list_players, f, indent=2)
                        self.players.remove(player)
                        break
                    else:
                        self.display_message(f"Invalid bet, place the bet again!", color=(255,0,0))
                except ValueError:
                    self.display_message(f"Invalid input. Please enter a number!", color=(255,0,0))

    def get_insurance(self):
        yes_button = CircleButton(700, 470, 50, "YES", (0, 0, 255), (255, 255, 255))
        no_button = CircleButton(850, 470, 50, "NO", (255, 0, 0), (255, 255, 255))
        font = pygame.font.Font(None, 40)
        text_surface = font.render("Would you like insurance?", True, (255, 255, 255))
        running = True
        selection = None
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()  
                if not yes_button.handle_event(event):
                    selection = '1'
                    running = False
                if not no_button.handle_event(event):
                    selection = '2'
                    running = False
            self.screen.blit(text_surface, (618, 386))
            yes_button.draw(self.screen)
            no_button.draw(self.screen)
            pygame.display.flip()
        self.display_table()
        for i, player in enumerate(self.players):
            player.display_cards(self.screen)
        self.dealer.display_cards(self.screen)
        return selection

    def offer_insurance(self, dealer_card, player):
        '''Offer insurance bet to player if dealer has an Ace'''
        while True:
            try:
                choice = player.place_insurance(self.deck, dealer_card)
                if choice == -1:
                    choice = self.get_insurance()
                else:
                    self.display_table()
                    for i, player in enumerate(self.players):
                        player.display_cards(self.screen)
                    self.dealer.display_cards(self.screen)
                if choice == '1':
                    break
                elif choice == '2':
                    break
                else:
                    self.display_message(f"Invalid response, try again!",color=(255,0,0))
            except ValueError:
                self.display_message(f"Invalid input. Please enter a number!",color=(255,0,0))

    def play_round(self):
        '''Play a full round: deal, insurance, player turns, dealer turn, evaluate results.'''
        self.deal_initial_cards()
        
        dealer_card = self.dealer.get_dealer_card()
        
        if dealer_card.rank == 'A':
            for player in self.players:
                if player.chips >= 1.5 * player.bets[0]:
                    self.offer_insurance(dealer_card, player)
        
        for player in self.players:
            self.display_message(f"{player.name}'s Turn:")
            self.player_turn(player)

        self.display_message(f"Dealer's Turn:")
        self.dealer_turn()

        dealer_value = self.calculate_hand_value(self.dealer.hand)
        for player in self.players:
            for index in range(len(player.hands)):
                reward = 0
                player_value = self.calculate_hand_value(player.hands[index])
                if player.insurance_bet != 0 and dealer_value == 21 and len(self.dealer.hand) == 2:
                    reward = player.insurance_bet
                if player_value > 21 or (dealer_value <= 21 and dealer_value > player_value):
                    self.display_message(f"{player.name} loses!", visible=True)
                    # LOSE
                    dealer_card = self.dealer.get_dealer_card()
                    reward -= player.bets[index]
                    player.get_reward(reward, dealer_card, self.deck, index, self.actions)
                    player.chips += reward
                elif player_value == 21 and len(player.hands[index]) == 2:
                    # BLACKJACK
                    if dealer_value != 21 or len(self.dealer.hand) != 2:
                        dealer_card = self.dealer.get_dealer_card()
                        reward += 1.5 * player.bets[index]
                        player.get_reward(reward, dealer_card, self.deck, index, self.actions)
                        player.chips += reward
                        self.display_message(f"{player.name} wins with Blackjack!", visible=True)
                    # BLACKJACK but BOTH
                    else:
                        dealer_card = self.dealer.get_dealer_card()
                        player.get_reward(reward, dealer_card, self.deck, index, self.actions)
                        player.chips += reward 
                elif player_value == dealer_value:
                    # BLACKJACK DEALER
                    if dealer_value == 21 and len(self.dealer.hand) == 2:
                        # Insurance
                        dealer_card = self.dealer.get_dealer_card()
                        reward -= player.bets[index]
                        player.get_reward(reward, dealer_card, self.deck, index, self.actions)
                        player.chips += reward
                        self.display_message(f"{player.name} loses!", visible=True)
                    # DRAW
                    else:
                        self.display_message(f"It's a draw!", visible=True)
                        dealer_card = self.dealer.get_dealer_card()
                        player.get_reward(reward, dealer_card, self.deck, index, self.actions)
                        player.chips += reward
                else:
                    self.display_message(f"{player.name} wins!", visible=True)
                    # WIN
                    dealer_card = self.dealer.get_dealer_card()
                    reward += player.bets[index]
                    player.get_reward(reward, dealer_card, self.deck, index, self.actions)
                    player.chips += reward

    def reset_hands(self):
        self.display_table()
        '''Clear hands for all players and dealer.'''
        for player in self.players + [self.dealer]:
            player.clear_hand()

    def start_game(self):
        '''Main game loop: place bets, play rounds, end if no players.'''
        while True:
            self.display_menu()
            self.display_table()
            running = True
            while running:
                self.place_bets()
                if self.players:
                    self.play_round()
                    self.reset_hands()
                else:
                    self.display_message(f'Thanks for playing.')
                    self.num_players = 0
                    self.players = []
                    running = False

if __name__ == "__main__":
    game = BlackjackGame()
    game.start_game()
