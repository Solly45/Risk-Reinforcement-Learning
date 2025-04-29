import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import json
import csv
from ai import AI
from collections import deque
import logging

class SkynetAI(AI):
    def __init__(self, player, game, world, **kwargs):
        super().__init__(player, game, world, **kwargs)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = self._get_state_size()
        self.action_size = len(world.territories)
        
        self.policy_net = self._build_network().to(self.device)
        self.target_net = self._build_network().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.05  # Minimum exploration rate
        self.epsilon_decay = 0.99999  # Decay factor for epsilon
        self.training_steps = 0  # Track the number of training iterations

        self.memory = deque(maxlen=1000000)  # Experience replay memory with priority
        self.priority_memory = deque(maxlen=1000000)  # Stores TD-errors for prioritization
        self.batch_size = 256  # Training batch size
        
        self.tau = 0.005  # Soft update rate for target network
        
        self.model_path = "models/risk_rl.pth"
        self.memory_path = "models/memory.pkl"
        os.makedirs("models", exist_ok=True)
        
        self.load_model()
        self.load_memory()
        self.episode_rewards = {
                "reinforcement": 0,
                "initial_placement": 0,
                "attack": 0,
                "freemove": 0,
                "total": 0
            }
        self.reward_log_path = "logs/reward_log.json"
        os.makedirs("logs", exist_ok=True)  # Ensure the logs directory exists
        self.total_game_loss = 0  # Track total loss per game
        self.loss_log_path = "logs/game_loss_log.csv"
        os.makedirs("logs", exist_ok=True)  # Ensure the logs directory exists
        
        

    # Set up logging to file
    logging.basicConfig(
        filename="training_log.txt",
        level=logging.INFO,
        format="%(asctime)s - Step: %(message)s",
        filemode="w"  # Overwrite file each run
    )

    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 512),
            nn.LayerNorm(512),  # BatchNorm1d to LayerNorm
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),  
            nn.ReLU(),

            nn.Linear(128, self.action_size)
        )



    
    def _get_state(self):
        state = []
        max_forces = max(t.forces for t in self.world.territories.values()) or 1  # Avoid division by zero
        for t in self.world.territories.values():
            state.extend([
                1 if t.owner == self.player else 0,
                t.forces / max_forces,  # Normalize forces
                len([adj for adj in t.connect if adj.owner != self.player]),
                sum(adj.forces for adj in t.connect if adj.owner != self.player) / max_forces  # Normalize
            ])
        return np.array(state, dtype=np.float32)

    
    def _get_state_size(self):
        return len(self.world.territories) * 4
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            actions = self.policy_net(state_tensor)
        return torch.argmax(actions).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Stores experiences in replay memory and updates priority values dynamically."""
        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)

        q_values = self.policy_net(state_tensor.unsqueeze(0)).detach()
        next_q_values = self.target_net(next_state_tensor.unsqueeze(0)).detach()
        target = reward + (1 - done) * self.gamma * next_q_values.max()
        td_error = abs(target - q_values[0, action]).item()

        self.memory.append((state, action, reward, next_state, done))

        # Keep priorities fresh to prevent overfitting
        if len(self.priority_memory) < self.memory.maxlen:
           self.priority_memory.append(td_error * 0.9 + 0.1 * np.mean(list(self.priority_memory)[-100:]))

        else:
            self.priority_memory[random.randint(0, len(self.priority_memory) - 1)] = td_error

        if len(self.memory) > self.memory.maxlen * 0.8:
            self.memory.popleft()

    
    def sample_experiences(self):
        """Samples experiences using Prioritized Experience Replay (PER) with importance sampling."""
        if len(self.memory) < self.batch_size:
            print("Skipping training due to insufficient experiences.")
            return None

        num_prioritized = int(self.batch_size * 0.6)  # 60% from priority memory
        num_random = self.batch_size - num_prioritized  # 40% from random memory

        # Convert priority memory to a numpy array and normalize for stable probability distribution
        priorities = np.array(self.priority_memory, dtype=np.float32)

        if len(priorities) == 0 or np.isnan(priorities).any():

            probabilities = np.ones(len(self.memory)) / len(self.memory)
        else:
            min_priority = np.min(priorities)
            max_priority = np.max(priorities)

            if min_priority == max_priority:
                print(f"INFO: All priorities are the same ({min_priority}), using uniform sampling.")
                probabilities = np.ones(len(self.memory)) / len(self.memory)
            else:
                probabilities = (priorities - min_priority) / (max_priority - min_priority + 1e-5)
                probabilities = probabilities ** 0.6  # Reduce bias from high-priority experiences
                probabilities /= np.sum(probabilities)

            # Ensure probabilities sum to 1 and contain no NaN values
            if np.isnan(probabilities).any() or np.sum(probabilities) == 0:
                print("ERROR: Normalization resulted in NaN or zero-sum probabilities. Using uniform sampling.")
                probabilities = np.ones(len(self.memory)) / len(self.memory)

        # Sample based on adjusted priority probabilities
        try:
            priority_indices = np.random.choice(len(self.memory), num_prioritized, p=probabilities)
        except ValueError as e:
            print("CRITICAL ERROR: Issue with probabilities in np.random.choice!")
            print("Probabilities:", probabilities)
            print("Priority memory:", self.priority_memory)
            raise e

        # Sample additional random experiences for diversity
        random_indices = np.random.choice(len(self.memory), num_random, replace=False)

        selected_indices = np.concatenate((priority_indices, random_indices))
        batch = [self.memory[i] for i in selected_indices]

        # Apply importance sampling weights
        weights = (1 / (len(self.memory) * probabilities[selected_indices])) ** 0.4  # Importance sampling correction
        weights /= weights.max()  # Normalize to prevent exploding gradients

        return zip(*batch), torch.tensor(weights, dtype=torch.float32).to(self.device)





    
    def train(self):
        """Training method with PER and adaptive updates."""
        self.training_steps += 1

        if len(self.memory) < self.batch_size:
            return  # Not enough experiences yet

        if self.training_steps % 10 != 0:  # Train every 10 steps 
            return

        experiences, importance_weights = self.sample_experiences()
        if experiences is None:
            return

        states, actions, rewards, next_states, dones = experiences

        states_tensor = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Double DQN: Use policy network for action selection, but target network for Q-value calculation
        next_actions = self.policy_net(next_states_tensor).argmax(dim=1, keepdim=True)
        next_q_values = self.target_net(next_states_tensor).gather(1, next_actions).squeeze(1).detach()
        targets = rewards_tensor + (1 - dones_tensor) * self.gamma * next_q_values

        # Compute current Q-values
        q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Apply importance weights to loss function to prevent biased updates
        loss = (importance_weights * (q_values - targets.detach()) ** 2).mean()
        
        # Gradient Clipping for stability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        #loss log
        self.total_game_loss += loss.item()

        # Adaptive Learning Rate Adjustment: Prevents stuck strategies
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(0.0001, param_group['lr'] * 0.999)  # Slow decay for better adaptation

        # Soft update for the target network
        self.soft_update_target_network()

        # Adaptive Epsilon Decay for Exploration
        if self.training_steps % 500 == 0:  
            win_rate = self.game.stats.win_counts.get(self.player.name, 0) / max(self.game.stats.total_games, 1)
            self.epsilon = max(self.epsilon_min, self.epsilon * (0.98 if win_rate < 0.4 else 0.995))

        logging.info(f"Step {self.training_steps}, Loss: {loss.item():.4f}, Epsilon: {self.epsilon:.4f}")




    def soft_update_target_network(self):
        """Performs a soft update on the target network to stabilize training."""
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * policy_param.data)



    
    def save_model(self):
        torch.save(self.policy_net.state_dict(), self.model_path)

        # Save training steps and epsilon
        training_data = {
            "training_steps": self.training_steps,
            "epsilon": self.epsilon
        }
        with open("models/training_data.json", "w") as f:
            json.dump(training_data, f)

        # Save optimizer state
        torch.save(self.optimizer.state_dict(), "models/optimizer.pth")

    def load_model(self):
        if os.path.exists(self.model_path):
            self.policy_net.load_state_dict(torch.load(self.model_path))
            self.target_net.load_state_dict(self.policy_net.state_dict())  # Copy weights
        else:
            print("No saved model found, starting fresh.")

        # Load optimizer state
        optimizer_path = "models/optimizer.pth"
        if os.path.exists(optimizer_path):
            try:
                self.optimizer.load_state_dict(torch.load(optimizer_path))
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state. Error: {e}")

        # Load training steps and epsilon
        training_data_path = "models/training_data.json"
        if os.path.exists(training_data_path):
            with open(training_data_path, "r") as f:
                training_data = json.load(f)
                self.training_steps = training_data.get("training_steps", 0)
                self.epsilon = training_data.get("epsilon", 1.0)
        else:
            self.training_steps = 0
            self.epsilon = 1.0



    
    def load_memory(self):
        if os.path.exists(self.memory_path) and os.path.getsize(self.memory_path) > 0:
            try:
                with open(self.memory_path, "rb") as f:
                    self.memory = pickle.load(f)
                    self.priority_memory = deque([1.0] * len(self.memory), maxlen=1000000)
            except (EOFError, pickle.UnpicklingError):
                print("Warning: Memory file is empty or corrupted. Initializing new memory.")
                self.memory = deque(maxlen=1000000)
                self.priority_memory = deque(maxlen=1000000)
        else:
            print("No existing memory file found. Creating new memory.")
            self.memory = deque(maxlen=1000000)
            self.priority_memory = deque(maxlen=1000000)
            
    def load_optimizer(self):
        optimizer_path = "models/optimizer.pth"
        if os.path.exists(optimizer_path):
            self.optimizer.load_state_dict(torch.load(optimizer_path))
        else:
            print("No saved optimizer state found, initializing new optimizer.")
    
    def end(self):
        self.save_model()
        
        with open(self.memory_path, "wb") as f:
            pickle.dump(self.memory, f)

       
        # Save episode rewards
        if os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, "r") as f:
                reward_data = json.load(f)
        else:
            reward_data = []

        reward_data.append({
            "reinforcement": self.episode_rewards["reinforcement"],
            "initial_placement": self.episode_rewards["initial_placement"],
            "attack": self.episode_rewards["attack"],
            "freemove": self.episode_rewards["freemove"],
            "total": self.episode_rewards["total"],
            "turns": self.game.turn
        })

        with open(self.reward_log_path, "w") as f:
            json.dump(reward_data, f, indent=4)

        self.episode_rewards = {
            "reinforcement": 0,
            "initial_placement": 0,
            "attack": 0,
            "freemove": 0,
            "total": 0
        }

        self.log_total_loss()
        self.total_game_loss = 0


    def log_total_loss(self):
        """Logs total loss per game into a CSV file."""
        file_exists = os.path.isfile(self.loss_log_path)

        with open(self.loss_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Game Number", "Total Loss"])  # Write headers if new file
            writer.writerow([self.game.stats.total_games, self.total_game_loss])


            
    def reinforce(self, available):
        """Reinforcement strategy that prioritizes border defense, continent control, and strategic depth."""
        state = self._get_state()
        if available <= 0:
            return {}

        reinforcement_priorities = {}
        total_enemy_forces = sum(adj.forces for t in self.player.territories for adj in t.connect if adj.owner != self.player)

        for t in self.player.territories:
            enemy_threat = sum(adj.forces for adj in t.connect if adj.owner != self.player)
            reinforcement_score = enemy_threat * 2 + (1 if t.border else 0.5)

            if t.area.owner == self.player:
                reinforcement_score += 3  

            if total_enemy_forces > 0:
                reinforcement_score += (enemy_threat / total_enemy_forces) * 5  

            reinforcement_priorities[t] = reinforcement_score

        top_territories = sorted(reinforcement_priorities, key=reinforcement_priorities.get, reverse=True)[:3]
        reinforcements = {t: available // len(top_territories) for t in top_territories}

        remaining = available - sum(reinforcements.values())
        if remaining > 0:
            reinforcements[top_territories[0]] += remaining

        action = self.choose_action(state)
        reward = sum(10 * (reinforcements[t] / max(enemy_threat, 1)) if t.border and enemy_threat > 0 else 2 for t in reinforcements)

        if all(t.forces > 0 for t in reinforcements):
            reward += 5
        else:
            reward -= 4

        if len([p for p in self.game.players.values() if p.alive]) == 1:
            winner = [p for p in self.game.players.values() if p.alive][0]
            if winner == self.player:
                reward += 20  

        self.episode_rewards["reinforcement"] += reward
        self.episode_rewards["total"] += reward

        next_state = self._get_state()
        done = False
        self.store_experience(state, action, reward, next_state, done)
        self.train()

        return reinforcements





        
    def attack(self):
        """Decision-driven attack phase based on probability of success and strategic value."""
        state = self._get_state()
        possible_attacks = [(t, adj) for t in self.player.territories for adj in t.connect if adj.owner != self.player and t.forces > 1]

        if not possible_attacks:
            return  

        while possible_attacks:
            action = self.choose_action(state)
            attack_index = action % len(possible_attacks)  
            attacker, defender = possible_attacks[attack_index]

            victory_chance, surviving_attackers, _ = self.simulate(attacker.forces, defender.forces)

            aggression_threshold = 0.6
            strategic_value = defender.area.value * 0.2  

            if victory_chance > aggression_threshold or strategic_value > 1.0:
                reward = 5.0 + (0.5 * strategic_value)
                yield (attacker, defender, None, lambda n: min(n - 1, int(surviving_attackers)))

                possible_attacks = [(t, adj) for t in self.player.territories for adj in t.connect if adj.owner != self.player and t.forces > 1]

                self.episode_rewards["attack"] += reward
                self.episode_rewards["total"] += reward

                if len([p for p in self.game.players.values() if p.alive]) == 1:
                    winner = [p for p in self.game.players.values() if p.alive][0]
                    if winner == self.player:
                        reward += 20  
                        self.episode_rewards["attack"] += 20
                        self.episode_rewards["total"] += 20

                next_state = self._get_state()
                done = False
                self.store_experience(state, action, reward, next_state, done)
                self.train()
            else:
                reward = -1.0  
                self.episode_rewards["attack"] += reward
                self.episode_rewards["total"] += reward

                next_state = self._get_state()
                done = False
                self.store_experience(state, action, reward, next_state, done)
                self.train()

                possible_attacks.pop(attack_index)





   
    def freemove(self):
        """Moves troops from safe territories to reinforce key defensive positions."""
        owned_territories = list(self.player.territories)
        if len(owned_territories) < 2:
            return None

        state = self._get_state()
        safe_territories = [t for t in owned_territories if not t.border and t.forces > 1]
        if not safe_territories:
            return None
        source = max(safe_territories, key=lambda t: t.forces)

        border_territories = [t for t in owned_territories if t.border]
        if not border_territories:
            return None
        target = min(border_territories, key=lambda t: t.forces)

        move_count = max(1, source.forces // 2)

        action = self.choose_action(state)

        reward = 5.0 if move_count > 1 and target.border else -2.0  

        if len([p for p in self.game.players.values() if p.alive]) == 1:
            winner = [p for p in self.game.players.values() if p.alive][0]
            if winner == self.player:
                reward += 10  

        self.episode_rewards["freemove"] += reward
        self.episode_rewards["total"] += reward

        next_state = self._get_state()
        done = False
        self.store_experience(state, action, reward, next_state, done)
        self.train()

        return (source, target, move_count)




    
    def initial_placement(self, empty, remaining):
        """Strategic initial placement focusing on high-value territories and area control."""
        state = self._get_state()
        action = self.choose_action(state)

        continent_priority = sorted(
            self.world.areas.values(),
            key=lambda area: (area.value / len(area.territories), -len([t for t in area.territories if t.area_border])),
            reverse=True
        )
        
        if empty:
            available_territories = sorted(
                empty,
                key=lambda t: continent_priority.index(t.area) if t.area in continent_priority else len(continent_priority)
            )
            chosen_territory = available_territories[action % len(available_territories)]
            
            base_reward = 6.0
            if chosen_territory.area_border:
                base_reward += 2.0  
            if len([t for t in chosen_territory.connect if t.owner is None]) > 1:
                base_reward -= 1.0  
        else:
            owned_territories = sorted(
                self.player.territories,
                key=lambda t: (continent_priority.index(t.area) if t.area in continent_priority else len(continent_priority), -t.forces)
            )
            chosen_territory = owned_territories[action % len(owned_territories)]

            base_reward = 3.0  
            if chosen_territory.border:
                base_reward += 2.0  
            if chosen_territory.area.owner == self.player:
                base_reward += 1.5  

        if len([p for p in self.game.players.values() if p.alive]) == 1:
            winner = [p for p in self.game.players.values() if p.alive][0]
            if winner == self.player:
                base_reward += 20  

        self.episode_rewards["initial_placement"] += base_reward
        self.episode_rewards["total"] += base_reward

        next_state = self._get_state()
        done = False
        self.store_experience(state, action, base_reward, next_state, done)
        self.train()
        
        return chosen_territory


