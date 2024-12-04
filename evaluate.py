import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from harlow_task import Harlow_1D
from DQN import DQNAgent, HopfieldDQNAgent
from AstroDQN import NeuronAstrocyteDQNAgent, train_dqn

def evaluate_models(env_class, model_classes, n_seeds=5, n_episodes=1000):
    results = {}
    
    for model_name, model_class in model_classes.items():
        print(f"Training {model_name}...")
        step_rewards = np.zeros((n_seeds, n_episodes, 6))  # Track rewards for each of the 6 steps
        
        for seed in range(n_seeds):
            print(f"Seed {seed + 1}/{n_seeds}")
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = env_class(verbose=False)
            agent = model_class(
                state_size=env.reset().size,
                n_actions=2
            )
            
            # Training loop with step tracking
            for episode in range(n_episodes):
                state = env.reset()
                step_in_episode = 0
                
                done = False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    
                    # Store step reward if it's from object selection (not fixation)
                    if abs(reward) == 1:  # Full reward/punishment indicates object selection
                        step_rewards[seed, episode, step_in_episode] = 1 if reward > 0 else 0
                        step_in_episode += 1
                    
                    # Use normal training process
                    agent.memory.push(state, action, reward, next_state, done)
                    if len(agent.memory) >= agent.batch_size:
                        loss = agent.train_step()
                    
                    state = next_state
                
                agent.update_epsilon()
                
                if (episode + 1) % 10 == 0:
                    print(f"Episode {episode + 1}/{n_episodes}")
        
        results[model_name] = step_rewards
    
    # Plot performance progression
    plt.figure(figsize=(12, 6))
    
    # Define episode ranges for early, middle, and late training
    episode_ranges = [
        (0, n_episodes // 3, 'Early Training'),
        (n_episodes // 3, 2 * n_episodes // 3, 'Middle Training'),
        (2 * n_episodes // 3, n_episodes, 'Late Training')
    ]
    
    colors = ['#FFB6C1', '#87CEEB', '#98FB98']  # Light red, light blue, light green
    markers = ['o', 's', '^']  # Circle, square, triangle
    
    for name, data in results.items():
        for i, (start_ep, end_ep, label) in enumerate(episode_ranges):
            # Calculate mean performance for each step in this range
            range_data = data[:, start_ep:end_ep, :]  # (n_seeds, range_episodes, 6)
            
            # Calculate mean across seeds and episodes
            step_means = range_data.mean(axis=(0, 1))
            
            # Calculate standard error properly for binomial data
            n_samples = n_seeds * (end_ep - start_ep)  # total number of trials for each step
            step_sems = np.sqrt((step_means * (1 - step_means)) / n_samples)
            
            # Plot
            steps = np.arange(1, 7)
            plt.errorbar(steps, step_means * 100, yerr=step_sems * 100,
                        label=f'{name} - {label}', 
                        marker=markers[i],
                        color=colors[i],
                        capsize=5,
                        markersize=8)
    
    plt.axhline(y=50, color='gray', linestyle='--', label='Random Choice')
    plt.xlabel('Step in Episode')
    plt.ylabel('Average Performance (%)')
    plt.title('Performance by Step in Episode\nProgression During Training')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('step_perf_eval.png')
    plt.show()
    
    # Print detailed statistics for each training phase
    print("\nDetailed Statistics by Training Phase:")
    for name, data in results.items():
        print(f"\n{name}:")
        for start_ep, end_ep, label in episode_ranges:  # Fixed this line
            print(f"\n{label}:")
            range_data = data[:, start_ep:end_ep, :]
            step_means = range_data.mean(axis=(0, 1)) * 100
            
            # Calculate standard error properly for binomial data
            n_samples = n_seeds * (end_ep - start_ep)
            step_sems = np.sqrt((step_means/100 * (1 - step_means/100)) / n_samples) * 100
            
            for step in range(6):
                print(f"Step {step + 1}: {step_means[step]:.1f}% ± {step_sems[step]:.1f}%")
    
    return results

if __name__ == "__main__":
    n_seeds = 1
    n_episodes = 1000
    
    models = {
        'HopfieldDQN': HopfieldDQNAgent,
        # Uncomment to compare with other models
        #'DQN': DQNAgent,
        # 'NeuronAstrocyteDQN': NeuronAstrocyteDQNAgent
    }
    
    results = evaluate_models(Harlow_1D, models, n_seeds=n_seeds, n_episodes=n_episodes)
