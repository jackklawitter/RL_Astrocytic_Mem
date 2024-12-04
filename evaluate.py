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
        model_results = []
        
        for seed in range(n_seeds):
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            env = env_class(verbose=False)
            agent = model_class(
                state_size=env.reset().size,
                n_actions=2
            )
            
            rewards, _ = train_dqn(env, agent, n_episodes=n_episodes)
            model_results.append(rewards)
            
        results[model_name] = np.array(model_results)
    
    def analyze_performance(data):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        final_perf = data[:, -100:].mean(axis=1)
        learning_speed = [(r > 0.7).argmax() if (r > 0.7).any() else n_episodes 
                         for r in data]
        
        return {
            'mean_curve': mean,
            'std_curve': std,
            'final_performance': final_perf.mean(),
            'final_std': final_perf.std(),
            'learning_speed': np.mean(learning_speed),
            'speed_std': np.std(learning_speed)
        }
    
    analysis = {name: analyze_performance(data) 
               for name, data in results.items()}
    
    model_names = list(results.keys())
    if len(model_names) > 1:
        final_perfs = [results[m][:, -100:].mean(axis=1) for m in model_names]
        stat, pval = stats.ttest_ind(final_perfs[0], final_perfs[1])
        
        print(f"Statistical comparison (t-test):")
        print(f"t-statistic: {stat:.3f}")
        print(f"p-value: {pval:.3f}")
    
    plt.figure(figsize=(12, 6))
    for name, res in analysis.items():
        plt.plot(res['mean_curve'], label=name)
        plt.fill_between(np.arange(len(res['mean_curve'])),
                        res['mean_curve'] - res['std_curve'],
                        res['mean_curve'] + res['std_curve'],
                        alpha=0.2)
    
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Model Comparison')
    plt.show()
    
    return analysis

if __name__ == "__main__":
    
    n_seeds = 1
    n_episodes = 100
    
    models = {
        'DQN': DQNAgent,
        'HopfieldDQN': HopfieldDQNAgent,
        'NeuronAstrocyteDQN': NeuronAstrocyteDQNAgent
    }
    
    results = evaluate_models(Harlow_1D, models,n_seeds=n_seeds, n_episodes=n_episodes)