import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from copy import deepcopy

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        # Debug print
        if isinstance(x, torch.Tensor):
            x = x.view(x.size(0), -1)  # Flatten any extra dimensions
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        # Ensure state is flattened
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_size,
        n_actions,
        hidden_size=128,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        
        print(f"Initializing DQN with state_size={state_size}")
        
        # Networks
        self.policy_net = DQN(state_size, hidden_size, n_actions).to(self.device)
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Other parameters
        self.n_actions = n_actions
        self.target_update = target_update
        self.steps = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.n_actions)
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # # Debug prints
        # print(f"States shape: {states.shape}")
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # # Debug prints
        # print(f"States tensor shape: {states.shape}")
        # print(f"Actions tensor shape: {actions.shape}")
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def train_dqn(env, agent, n_episodes=1000, max_steps=250):
    rewards_history = []
    loss_history = []
    
    # Debug: Check initial state shape
    initial_state = env.reset()
    print(f"Initial state shape: {initial_state.shape}")
    print(f"Initial state: {initial_state}")
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition and train
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            
            if done:
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record metrics
        rewards_history.append(total_reward)
        if episode_losses:
            loss_history.append(np.mean(episode_losses))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_loss = np.mean(loss_history[-10:]) if loss_history else 0
            print(f"Episode {episode + 1}")
            print(f"Average Reward (last 10): {avg_reward:.2f}")
            print(f"Average Loss (last 10): {avg_loss:.4f}")
            print(f"Epsilon: {agent.epsilon:.3f}\n")
    torch.save(agent, 'trained_agent.pt')
    return rewards_history, loss_history

class HopfieldNetwork:
    def __init__(self, size, max_memories=100):
        self.size = size
        self.max_memories = max_memories
        self.memories = []
        self.weights = np.zeros((size, size))
    
    def store_pattern(self, pattern):
        """Store a new pattern in the Hopfield network"""
        # Convert to bipolar representation (-1, 1)
        pattern = np.array(pattern)
        pattern = np.where(pattern > 0, 1, -1)
        
        # Add to memories
        self.memories.append(pattern)
        if len(self.memories) > self.max_memories:
            self.memories.pop(0)  # Remove oldest memory
        
        # Recalculate weights using Hebbian learning
        self.weights = np.zeros((self.size, self.size))
        for mem in self.memories:
            self.weights += np.outer(mem, mem)
        
        # Zero out diagonal
        np.fill_diagonal(self.weights, 0)
        
        # Normalize weights
        self.weights /= len(self.memories)

    def retrieve_pattern(self, probe, n_iterations=10):
        """Retrieve stored pattern most similar to probe pattern"""
        # Convert to bipolar
        state = np.where(probe > 0, 1, -1)
        
        # Run network dynamics
        for _ in range(n_iterations):
            for i in range(self.size):
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation > 0 else -1
        
        # Convert back to original representation
        return np.where(state > 0, 1, 0)

    def get_closest_memory(self, probe):
        """Find the stored memory most similar to probe pattern"""
        if not self.memories:
            return probe
        
        probe = np.where(probe > 0, 1, -1)
        similarities = [np.dot(probe, mem) for mem in self.memories]
        closest_idx = np.argmax(similarities)
        return np.where(self.memories[closest_idx] > 0, 1, 0)

class HopfieldDQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, episodic_size=64):
        super(HopfieldDQN, self).__init__()
        
        self.episodic_size = episodic_size
        self.hopfield = HopfieldNetwork(episodic_size)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, episodic_size)
        )
        
        # Main DQN network
        self.network = nn.Sequential(
            nn.Linear(input_size + episodic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def store_episode(self, state):
        """Store state pattern in episodic memory"""
        with torch.no_grad():
            # Ensure state is 2D
            if state.ndim == 1:
                state = state.unsqueeze(0)
            encoded = self.encoder(torch.FloatTensor(state))
            encoded = encoded.numpy()
            self.hopfield.store_pattern(encoded[0])  # Store the first pattern
    
    def forward(self, x):
        # Ensure input is 2D
        if x.ndim == 1:
            x = x.unsqueeze(0)
            
        # Encode current state
        encoded = self.encoder(x)
        
        # Process each sample in the batch
        batch_size = x.size(0)
        retrieved = []
        
        for i in range(batch_size):
            # Get memory for each sample
            memory = self.hopfield.get_closest_memory(encoded[i].detach().cpu().numpy())
            retrieved.append(memory)
            
        # Stack retrieved memories and convert to tensor
        retrieved = torch.FloatTensor(np.stack(retrieved)).to(x.device)
        
        # Concatenate original input with retrieved memory
        combined = torch.cat([x, retrieved], dim=1)
        
        # Process through main network
        return self.network(combined)

class HopfieldDQNAgent(DQNAgent):
    def __init__(
        self,
        state_size,
        n_actions,
        hidden_size=128,
        episodic_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update=10,
        memory_update_freq=5
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        
        # Networks
        self.policy_net = HopfieldDQN(
            state_size, 
            hidden_size, 
            n_actions, 
            episodic_size
        ).to(self.device)
        self.target_net = deepcopy(self.policy_net)
        self.target_net.eval()
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Other parameters
        self.n_actions = n_actions
        self.target_update = target_update
        self.memory_update_freq = memory_update_freq
        self.steps = 0
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state.flatten()).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.n_actions)
    
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Store in episodic memory periodically
        if self.steps % self.memory_update_freq == 0:
            self.policy_net.store_episode(states[0].cpu())
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()

def train_hopfield_dqn(env, agent, n_episodes=1000, max_steps=250):
    rewards_history = []
    loss_history = []
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Store transition and train
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)
            
            state = next_state
            
            if done:
                break
        
        # Update exploration rate
        agent.update_epsilon()
        
        # Record metrics
        rewards_history.append(total_reward)
        if episode_losses:
            loss_history.append(np.mean(episode_losses))
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            avg_loss = np.mean(loss_history[-10:]) if loss_history else 0
            print(f"Episode {episode + 1}")
            print(f"Average Reward (last 10): {avg_reward:.2f}")
            print(f"Average Loss (last 10): {avg_loss:.4f}")
            print(f"Epsilon: {agent.epsilon:.3f}\n")
    
    return rewards_history, loss_history

def plot_training_results(rewards, losses):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'))
    ax1.set_title('Average Reward over Training')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward (100-episode window)')
    
    # Plot losses
    if losses:
        ax2.plot(np.convolve(losses, np.ones(100)/100, mode='valid'))
        ax2.set_title('Average Loss over Training')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Loss (100-episode window)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from harlow_task import Harlow_1D
    
    # # Initialize environment
    # env = Harlow_1D(verbose=False)
    # initial_state = env.reset()
    # print(f"Initial state shape from environment: {initial_state.shape}")
    
    # # Initialize agent with correct state size
    # agent = DQNAgent(
    #     state_size=initial_state.size,  # Use the actual size of the flattened state
    #     n_actions=2,   # Left or right
    #     hidden_size=128,
    #     learning_rate=1e-3,
    #     gamma=0.99,
    #     epsilon_start=1.0,
    #     epsilon_end=0.01,
    #     epsilon_decay=0.995,
    #     buffer_size=10000,
    #     batch_size=64,
    #     target_update=10
    # )
    
    # # Train the agent
    # rewards, losses = train_dqn(env, agent, n_episodes=5000)
    
    
    # Initialize environment
    env = Harlow_1D(verbose=False)
    initial_state = env.reset()

    # Initialize Hopfield-DQN agent
    agent = HopfieldDQNAgent(
        state_size=initial_state.size,
        n_actions=2,
        hidden_size=128,
        episodic_size=64,  # Size of patterns stored in Hopfield network
        memory_update_freq=5  # How often to store new patterns
    )

    # Train the agent
    rewards, losses = train_hopfield_dqn(env, agent, n_episodes=1000)
    # Plot results
    plot_training_results(rewards, losses)