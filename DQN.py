import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from copy import deepcopy
import os

from harlow_task import Harlow_1D

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

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=2):  # output_size=2 for binary object choice
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # outputs Q-values for selecting object 1 or object 2
        )
    
    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.view(x.size(0), -1)
        return self.network(x)

class DQNAgent:
    def __init__(
        self,
        state_size,
        n_actions = 2,
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
        
        # Networks - output size is 2 for binary object choice
        self.policy_net = DQN(state_size, hidden_size, 2).to(self.device)
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
        
        # Track object positions
        self.left_obj = None
        self.right_obj = None
    
    def update_object_positions(self, state):
        """Update knowledge of which object is on which side"""
        center = len(state) // 2
        offset = 3  # from the original environment
        
        self.left_obj = state[center - offset]
        self.right_obj = state[center + offset]
    
    def select_action(self, state):
        # Update object positions
        self.update_object_positions(state)
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                object_choice = q_values.argmax().item()  # 0 for first object, 1 for second object
                
                # Convert object choice to movement direction
                if object_choice == 0:  # Choose first object
                    return 0 if self.left_obj > 0 else 1  # go left if object is on left, right if on right
                else:  # Choose second object
                    return 0 if self.right_obj > 0 else 1  # go left if object is on left, right if on right
        else:
            return random.randrange(2)  # Still return 0 or 1 for left/right movement
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
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
        
        # Convert movement actions to object choices for Q-learning
        object_choices = []
        for i, state in enumerate(states):
            center = len(state) // 2
            offset = 3
            left_obj = state[center - offset].item()
            right_obj = state[center + offset].item()
            
            # Determine which object was chosen based on the movement action
            if actions[i] == 0:  # went left
                chosen_obj = 0 if left_obj > 0 else 1
            else:  # went right
                chosen_obj = 0 if right_obj > 0 else 1
            object_choices.append(chosen_obj)
        
        object_choices = torch.LongTensor(object_choices).unsqueeze(1).to(self.device)
        
        # Compute current Q values for object choices
        current_q_values = self.policy_net(states).gather(1, object_choices)
        
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
    
    initial_state = env.reset()
    
    
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
    def __init__(self, input_size, hidden_size, output_size=2, episodic_size=64):
        super(HopfieldDQN, self).__init__()
        
        self.episodic_size = episodic_size
        self.hopfield = HopfieldNetwork(episodic_size)
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, episodic_size)
        )
        
        # Main DQN network - now outputs Q-values for object selection
        self.network = nn.Sequential(
            nn.Linear(input_size + episodic_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)  # outputs Q-values for selecting object 1 or 2
        )
    
    def store_episode(self, state):
        """Store state pattern in episodic memory"""
        with torch.no_grad():
            if state.ndim == 1:
                state = state.unsqueeze(0)
            encoded = self.encoder(state.to(self.encoder[0].weight.device))
            self.hopfield.store_pattern(encoded[0].cpu().numpy())
    
    def forward(self, x):
        device = self.encoder[0].weight.device
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.to(device)
            
        # Encode current state
        encoded = self.encoder(x)
        
        # Process each sample in batch
        batch_size = x.size(0)
        retrieved = []
        
        for i in range(batch_size):
            memory = self.hopfield.retrieve_pattern(encoded[i].detach().cpu().numpy())
            retrieved.append(memory)
            
        retrieved = torch.FloatTensor(np.stack(retrieved)).to(device)
        
        # Concatenate original input with retrieved memory
        combined = torch.cat([x, retrieved], dim=1)
        
        # Process through main network to get object selection Q-values
        return self.network(combined)

class HopfieldDQNAgent(DQNAgent):
    def __init__(
        self,
        state_size,
        n_actions=2,
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
            input_size=state_size,
            hidden_size=hidden_size,
            output_size=2,  # binary object choice
            episodic_size=episodic_size
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
        
        # Track object positions
        self.left_obj = None
        self.right_obj = None
    
    def update_object_positions(self, state):
        """Update knowledge of which object is on which side"""
        center = len(state) // 2
        offset = 3  # from the original environment
        
        self.left_obj = state[center - offset]
        self.right_obj = state[center + offset]
    
    def select_action(self, state):
        # Update object positions
        self.update_object_positions(state)
        
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state.flatten()).to(self.device)
                q_values = self.policy_net(state)
                object_choice = q_values.argmax().item()  # 0 for first object, 1 for second object
                
                # Convert object choice to movement direction
                if object_choice == 0:  # Choose first object
                    return 0 if self.left_obj > 0 else 1  # go left if object is on left, right if on right
                else:  # Choose second object
                    return 0 if self.right_obj > 0 else 1  # go left if object is on left, right if on right
        else:
            return random.randrange(2)
    
    def store_episode(self, state):
        """Store state in episodic memory"""
        self.policy_net.store_episode(torch.FloatTensor(state))
    
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
            self.store_episode(states[0])
        
        # Convert movement actions to object choices for Q-learning
        object_choices = []
        for i, state in enumerate(states):
            center = len(state) // 2
            offset = 3
            left_obj = state[center - offset].item()
            right_obj = state[center + offset].item()
            
            # Determine which object was chosen based on the movement action
            if actions[i] == 0:  # went left
                chosen_obj = 0 if left_obj > 0 else 1
            else:  # went right
                chosen_obj = 0 if right_obj > 0 else 1
            object_choices.append(chosen_obj)
        
        object_choices = torch.LongTensor(object_choices).unsqueeze(1).to(self.device)
        
        # Compute current Q values for object choices
        current_q_values = self.policy_net(states).gather(1, object_choices)
        
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
    
def save_rewards(rewards, base_path, run_title, seed, worker):
    """Save rewards for a specific run and worker"""
    run_dir = os.path.join(base_path, f"{run_title}_{seed}")
    os.makedirs(run_dir, exist_ok=True)
    np.save(os.path.join(run_dir, f"rewards_{worker}.npy"), rewards)

def train_dqn_multi_seed(env_class, agent_class, n_seeds=8, n_workers=8, base_path="ckpt", run_title="Harlow_DQN"):
    for seed in range(1, n_seeds+1):
        print(f"Training seed {seed}")
        for worker in range(n_workers):
            # Set seeds
            torch.manual_seed(seed * 100 + worker)
            np.random.seed(seed * 100 + worker)
            
            # Initialize environment and agent
            env = env_class(verbose=False)
            agent = agent_class(
                state_size=env.reset().size,
                n_actions=2
            )
            
            # Train and collect rewards
            rewards_history, _ = train_dqn(env, agent, n_episodes=2500)
            
            # Save rewards
            save_rewards(rewards_history, base_path, run_title, seed, worker)

def plot_training_results(base_path, run_title, n_seeds=8, n_workers=8):
    all_rewards = []
    for seed in range(1, n_seeds+1):
        run = run_title + f"_{seed}"
        run_rewards = []
        for worker in range(n_workers):
            path = os.path.join(base_path, run, f"rewards_{worker}.npy")
            if os.path.exists(path):
                rewards = np.load(path)
                run_rewards += [rewards[:2500]]
        all_rewards += [np.array(run_rewards).mean(axis=0)]
    
    all_rewards = np.stack(all_rewards)
    quantiles = [0, 500, 1000, 1500, 2000, 2500]
    n_quantiles = len(quantiles)-1
    n_trials = all_rewards.shape[2]
    
    plt.figure(figsize=(10, 6))
    for i in range(n_quantiles):
        line = []
        stds = []
        for j in range(n_trials):
            q = all_rewards[:,quantiles[i]:quantiles[i+1],j]
            performance = q.mean(axis=1)
            line += [performance.mean()*100]
            stds += [(performance.std()*100)]
        plt.errorbar(np.arange(1,7), line, fmt='o-', yerr=stds)
    
    plt.plot([1,6], [50,50], '--')
    plt.xlabel("Trial")
    plt.ylabel("Performance (%)")
    plt.legend(["Random", "1st", "2nd", "3rd", "4th", "Final"], title="Training Quantile")
    plt.title("Harlow Task - DQN Performance")
    plt.show()

# if __name__ == "__main__":
#     from harlow_task import Harlow_1D
    
#     # Train multiple seeds
#     train_dqn_multi_seed(Harlow_1D, DQNAgent)
    
#     # Plot results
#     plot_training_results("ckpt", "Harlow_DQN")

if __name__ == "__main__":
#     from harlow_task import Harlow_1D
    
    # # Initialize environment
    env = Harlow_1D(verbose=False)
    initial_state = env.reset()
    print(f"Initial state shape from environment: {initial_state.shape}")
    
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
    # rewards, losses = train_dqn(env, agent, n_episodes=2500)
    
    # # Initialize Hopfield-DQN agent
    agent = HopfieldDQNAgent(
        state_size=initial_state.size,
        n_actions=2,
        hidden_size=128,
        episodic_size=64,  # Size of patterns stored in Hopfield network
        memory_update_freq=5  # How often to store new patterns
    )
    rewards, losses = train_hopfield_dqn(env, agent, n_episodes=1000)
   