'''Module to implement Deep Q-learning Agent'''

from agents.player import Player
import math
import gc
import time
import torch
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NoisyLinear(nn.Module):
    '''Noisy Linear Layer for exploration'''
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()  
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))  
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        '''Initialize weights and biases'''
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        '''Scale noise for factorized Gaussian noise'''
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        '''Reset factorized Gaussian noise'''
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        '''Forward pass with noise'''
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelingNoisyDQNNetwork(nn.Module):
    '''DQN with dueling architecture, noisy layers, and layer normalization'''
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 256, 384, 256, 128]):
        super(DuelingNoisyDQNNetwork, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.layer_norm1 = nn.LayerNorm(hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(len(hidden_dims)-1):
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layer_norms.append(nn.LayerNorm(hidden_dims[i+1]))
        
        self.value_noisy = NoisyLinear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.value_out = NoisyLinear(hidden_dims[-1] // 2, 1)
        self.advantage_noisy = NoisyLinear(hidden_dims[-1], hidden_dims[-1] // 2)
        self.advantage_out = NoisyLinear(hidden_dims[-1] // 2, output_dim)
        self._initialize_weights()
    
    def _initialize_weights(self):
        '''Initialize weights'''
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        '''Forward pass through the network with dueling architecture'''
        x = self.input_layer(x)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        prev_x = x
        for i, (hidden_layer, layer_norm) in enumerate(zip(self.hidden_layers, self.layer_norms)):
            x = hidden_layer(x)
            x = layer_norm(x)
            
            if i % 2 == 0 and i > 0 and prev_x.shape == x.shape:
                x = F.gelu(x + prev_x)
            else:
                x = F.gelu(x)
                
            prev_x = x

        value = F.gelu(self.value_noisy(x))
        value = self.value_out(value)
        advantage = F.gelu(self.advantage_noisy(x))
        advantage = self.advantage_out(advantage)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)        
        return q_values
    
    def reset_noise(self):
        '''Reset noise for all noisy layers'''
        self.value_noisy.reset_noise()
        self.value_out.reset_noise()
        self.advantage_noisy.reset_noise()
        self.advantage_out.reset_noise()

class ReplayBuffer:
    '''Replay buffer with prioritized experience replay'''
    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        '''Add a transition to the buffer with max priority'''
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        '''Sample a batch using prioritized sampling'''
        if self.size < batch_size:
            return [], [], []
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights
    
    def update_priorities(self, indices, td_errors):
        '''Update priorities based on new TD errors'''
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class DQNagent(Player):
    '''Represents agent that uses DQN'''
    def __init__(self, name, chips,    
                alpha_betting=0.001, gamma_betting=0.99,
                alpha_insurance=0.001, gamma_insurance=0.99,
                alpha_playing=0.001, gamma_playing=0.99):
        super().__init__(name, chips)
        self.alpha_betting = alpha_betting
        self.gamma_betting = gamma_betting
        self.alpha_insurance = alpha_insurance
        self.gamma_insurance = gamma_insurance
        self.alpha_playing = alpha_playing
        self.gamma_playing = gamma_playing
        self.choice = None
        self.choices = []
        self.insurance_choice = None
        self.bet_state = None
        self.play_state = None
        self.insurance_state = None
        self.batch_size = 256
        self.memory_betting = ReplayBuffer(capacity=20000)
        self.memory_playing = ReplayBuffer(capacity=20000)
        self.memory_insurance = ReplayBuffer(capacity=10000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_betting = self.build_model(11, 50)
        self.target_betting = self.build_model(11, 50)
        self.target_betting.load_state_dict(self.q_betting.state_dict())
        self.optimizer_betting = optim.Adam(self.q_betting.parameters(), lr=alpha_betting, weight_decay=1e-5)
        self.scheduler_betting = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_betting, 'min', patience=5, factor=0.5)

        self.q_playing = self.build_model(15, 4)
        self.target_playing = self.build_model(15, 4)
        self.target_playing.load_state_dict(self.q_playing.state_dict())
        self.optimizer_playing = optim.Adam(self.q_playing.parameters(), lr=alpha_playing, weight_decay=1e-5)
        self.scheduler_playing = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_playing, 'min', patience=5, factor=0.5)
        
        self.q_insurance = self.build_model(15, 2)
        self.target_insurance = self.build_model(15, 2)
        self.target_insurance.load_state_dict(self.q_insurance.state_dict())
        self.optimizer_insurance = optim.Adam(self.q_insurance.parameters(), lr=alpha_insurance, weight_decay=1e-5)
        self.scheduler_insurance = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_insurance, 'min', patience=5, factor=0.5)
        
        self.criterion = nn.SmoothL1Loss()
        self.episode = 0
        self.target_update_frequency = 10
        
        self.losses = {
            "betting": [],
            "playing": [],
            "insurance": []
        }

    def save_models(self, path="dqn_models_better"):
        '''Save models for later usage'''
        torch.save({
            'q_betting_state_dict': self.q_betting.state_dict(),
            'target_betting_state_dict': self.target_betting.state_dict(),
            'optimizer_betting_state_dict': self.optimizer_betting.state_dict(),
            'scheduler_betting_state_dict': self.scheduler_betting.state_dict(),
        }, f"{path}_betting.pth")
        
        torch.save({
            'q_playing_state_dict': self.q_playing.state_dict(),
            'target_playing_state_dict': self.target_playing.state_dict(),
            'optimizer_playing_state_dict': self.optimizer_playing.state_dict(),
            'scheduler_playing_state_dict': self.scheduler_playing.state_dict(),
        }, f"{path}_playing.pth")
        
        torch.save({
            'q_insurance_state_dict': self.q_insurance.state_dict(),
            'target_insurance_state_dict': self.target_insurance.state_dict(),
            'optimizer_insurance_state_dict': self.optimizer_insurance.state_dict(),
            'scheduler_insurance_state_dict': self.scheduler_insurance.state_dict(),
        }, f"{path}_insurance.pth")
        
        print("Models saved to disk.")

    def load_models(self, path="dqn_models_better"):
        '''Load models'''
        try:
            betting_checkpoint = torch.load(f"{path}_betting.pth", map_location=self.device)
            self.q_betting.load_state_dict(betting_checkpoint['q_betting_state_dict'])
            self.target_betting.load_state_dict(betting_checkpoint['target_betting_state_dict'])
            self.optimizer_betting.load_state_dict(betting_checkpoint['optimizer_betting_state_dict'])
            self.scheduler_betting.load_state_dict(betting_checkpoint['scheduler_betting_state_dict'])
            
            playing_checkpoint = torch.load(f"{path}_playing.pth", map_location=self.device)
            self.q_playing.load_state_dict(playing_checkpoint['q_playing_state_dict'])
            self.target_playing.load_state_dict(playing_checkpoint['target_playing_state_dict'])
            self.optimizer_playing.load_state_dict(playing_checkpoint['optimizer_playing_state_dict'])
            self.scheduler_playing.load_state_dict(playing_checkpoint['scheduler_playing_state_dict'])
            
            insurance_checkpoint = torch.load(f"{path}_insurance.pth", map_location=self.device)
            self.q_insurance.load_state_dict(insurance_checkpoint['q_insurance_state_dict'])
            self.target_insurance.load_state_dict(insurance_checkpoint['target_insurance_state_dict'])
            self.optimizer_insurance.load_state_dict(insurance_checkpoint['optimizer_insurance_state_dict'])
            self.scheduler_insurance.load_state_dict(insurance_checkpoint['scheduler_insurance_state_dict'])
            
            print("Models loaded successfully from disk.")
        except Exception as e:
            print(f"Error loading models: {e}")

    def clear_hand(self):
        '''Reset the agent's hand'''
        super().clear_hand()
        self.choices = []
        
        if self.episode % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.episode % self.target_update_frequency == 0:
            self.target_betting.load_state_dict(self.q_betting.state_dict())
            self.target_playing.load_state_dict(self.q_playing.state_dict())
            self.target_insurance.load_state_dict(self.q_insurance.state_dict())
            print(f"Target networks updated at episode {self.episode}")
        self.episode += 1

    def build_model(self, dims, output_dims):
        '''Creates and returns a Dueling Noisy DQN'''
        model = DuelingNoisyDQNNetwork(dims, output_dims, hidden_dims=[128, 256, 384, 256, 128]).to(self.device)
        return model
    
    def remember(self, state, action, reward, next_state, done, memory):
        '''Stores a transition in the replay buffer'''
        memory.add(state, action, reward, next_state, done)

    def replay(self, memory, model, target_model, optimizer, gamma, phase, scheduler=None):
        '''Trains the model using a batch of experiences from the replay buffer'''
        batch, indices, weights = memory.sample(self.batch_size)
        if not batch:
            return 0
        
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        if phase == "betting":
            actions = torch.LongTensor([int(a / 10) - 1 for a in actions]).to(self.device)
        else:
            actions = torch.LongTensor([int(a) - 1 for a in actions]).to(self.device)
        
        current_q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            online_next_actions = model(next_states).argmax(dim=1, keepdim=True)
            next_q_values = target_model(next_states).gather(1, online_next_actions).squeeze(1)
            target_q_values = rewards.clone()
            non_terminal_mask = ~dones
            target_q_values[non_terminal_mask] += gamma * next_q_values[non_terminal_mask]
        
        td_errors = target_q_values.detach() - current_q_values.detach()
        
        memory.update_priorities(indices, td_errors.cpu().numpy())
        
        optimizer.zero_grad()
        loss = (weights * self.criterion(current_q_values, target_q_values)).mean()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step(loss)
        
        loss_value = loss.item()
        self.losses[phase].append(loss_value)
            
        return loss_value

    def get_card_counts(self, used_cards):
        '''Count occurrences of each card value'''
        card_counts = [0] * 10
        for value in used_cards:
           card_counts[value - 2] += 1
        return card_counts

    # Betting phase
    def get_betting_state(self, used_cards):
        '''Creates a betting state'''
        card_counts = self.get_card_counts(used_cards)
        state = np.array([self.chips] + card_counts, dtype=np.float32)
        state[0] /= 2000.0
        for i in range(1, len(state)):
            state[i] /= 24.0
        return state

    def choose_bet_action(self, state):
        '''Selects an action for betting based on the current state'''
        if self.chips >= 2000:  # Success, mission accomplished
            return 1
        min_bet = 10
        chips = int(self.chips)
        max_bet = 500 if chips >= 500 else chips - (chips % 10)
        if self.chips < min_bet:
            return 0
        possible_bets = list(range(min_bet, max_bet + 1, 10))
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_betting(state_tensor)
            for i in range(max_bet // 10, 50):
                q_values[0][i] = float('-inf')
            return (torch.argmax(q_values[0]).item() + 1) * 10

    def update_q_table_betting(self, state, action, reward, next_state, done):
        '''Update Q-value for betting phase using Q-learning function'''
        self.remember(state, action, reward, next_state, done, self.memory_betting)
        loss = self.replay(
            self.memory_betting, 
            self.q_betting, 
            self.target_betting, 
            self.optimizer_betting,
            self.gamma_betting, 
            "betting",
            self.scheduler_betting
        )

    def place_bet(self, deck):
        '''Agent places bet'''
        used_cards = deck.get_used_cards()
        self.bet_state = self.get_betting_state(used_cards)
        bet = self.choose_bet_action(self.bet_state)
        self.bets.append(bet)
        self.choices.append(bet)
        return bet

    # Insurance phase
    def get_insurance_state(self, player_sum, num_aces, dealer_sum, used_cards):
        '''Creates a insurance state'''
        card_counts = self.get_card_counts(used_cards)
        state = np.array([player_sum, num_aces, dealer_sum, self.bets[0], self.chips] + card_counts, dtype=np.float32)
        # Normalize state values
        state[0] /= 21.0
        state[1] /= 21.0
        state[2] /= 11.0
        state[3] /= 500.0
        state[4] /= 2000.0
        for i in range(5, len(state)):
            state[i] /= 24.0
        return state

    def choose_insurance_action(self, state):
        '''Selects an action for insurance based on the current state'''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_insurance(state_tensor)
            return str(torch.argmax(q_values[0]).item() + 1)  # Mapping insurance actions

    def update_q_table_insurance(self, state, action, reward, next_state):
        '''Update Q-value for insurance phase using Q-learning function'''
        self.remember(state, action, reward, next_state, True, self.memory_insurance)
        loss = self.replay(
            self.memory_insurance, 
            self.q_insurance, 
            self.target_insurance, 
            self.optimizer_insurance,
            self.gamma_insurance, 
            "insurance",
            self.scheduler_insurance
        )

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
        self.insurance_state = self.get_insurance_state(player_sum, num_aces, dealer_sum, used_cards)
        self.insurance_choice = self.choose_insurance_action(self.insurance_state)
        if self.insurance_choice == '1':
            self.insurance_bet = self.bets[0] / 2
        return self.insurance_choice

    # Playing phase
    def get_playing_state(self, player_sum, num_aces, dealer_sum, used_cards, index):
        '''Creates a playing state'''
        card_counts = self.get_card_counts(used_cards)
        state = np.array([player_sum, num_aces, dealer_sum, self.bets[index], self.chips] + card_counts, dtype=np.float32)
        # Normalize state values
        state[0] /= 21.0
        state[1] /= 21.0
        state[2] /= 11.0 
        state[3] /= 500.0
        state[4] /= 2000.0
        for i in range(5, len(state)):
            state[i] /= 24.0
        return state

    def choose_play_action(self, state, actions):
        '''Selects an action based on the current state'''
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_playing(state_tensor)
            action_mask = torch.ones(4, device=self.device) * float('-inf')
            for a in actions:
                action_idx = int(a) - 1
                action_mask[action_idx] = 0
            
            masked_q_values = q_values + action_mask
            return str(torch.argmax(masked_q_values[0]).item() + 1)
    
    def update_q_table_playing(self, state, action, reward, next_state, actions, done):
        '''Update Q-value for playing phase using Q-learning function'''
        self.remember(state, action, reward, next_state, done, self.memory_playing)
        loss = self.replay(
            self.memory_playing, 
            self.q_playing, 
            self.target_playing, 
            self.optimizer_playing,
            self.gamma_playing, 
            "playing",
            self.scheduler_playing
        )

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
        self.play_state = self.get_playing_state(player_sum, num_aces, dealer_sum, used_cards, index)
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
        next_playing_state = self.get_playing_state(player_sum, num_aces, dealer_sum, used_cards, index)

        if reward == 0:
            self.update_q_table_playing(self.play_state, self.choice, reward, next_playing_state, actions, False)
        else:
            if self.insurance_choice != None:
                insurance_reward = 0
                if dealer_card.rank == 'A' and self.insurance_choice == '1':
                    insurance_reward = self.insurance_bet * 2
                elif self.insurance_choice == '1':
                    insurance_reward = -self.insurance_bet
                
                next_insurance_state = self.get_insurance_state(player_sum, num_aces, dealer_sum, used_cards)
                self.update_q_table_insurance(self.insurance_state, self.insurance_choice, insurance_reward, next_insurance_state)
                self.insurance_choice = None
            
            if self.choice != None:
                self.update_q_table_playing(self.play_state, self.choice, reward, next_playing_state, actions, True)
            
            next_betting_state = self.get_betting_state(used_cards)
            done = self.chips >= 2000 or self.chips <= 0
            self.update_q_table_betting(self.bet_state, self.choices[index], reward, next_betting_state, done)
