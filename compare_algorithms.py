import pandas as pd
import matplotlib.pyplot as plt
import time
from agent import train as train_a3c
from agent1 import train as train_dqn
from agent2 import train as train_pg
from agent3 import train as train_ppo
from agent4 import train as train_sac

class AlgorithmTrainer:
    def __init__(self, train_function, name):
        self.train_function = train_function
        self.name = name
        self.metrics = {
            'scores': [],
            'timestamps': [],
            'episode_lengths': []
        }
        self.start_time = None
        
    def train(self, num_episodes):
        self.start_time = time.time()
        self.train_function(num_episodes, self.record_metrics)
        return self.calculate_stats()
    
    def record_metrics(self, score, episode_length):
        self.metrics['scores'].append(score)
        self.metrics['episode_lengths'].append(episode_length)
        self.metrics['timestamps'].append(time.time() - self.start_time)
    
    def calculate_stats(self):
        scores = self.metrics['scores']
        timestamps = self.metrics['timestamps']
        lengths = self.metrics['episode_lengths']
        
        # Calculate convergence
        window_size = 100
        if len(scores) >= window_size:
            moving_avg = pd.Series(scores).rolling(window=window_size).mean()
            for i in range(window_size, len(moving_avg)-50):
                if all(abs(moving_avg[i+j] - moving_avg[i])/moving_avg[i] < 0.05 
                       for j in range(50)):
                    convergence_episode = i
                    break
            else:
                convergence_episode = len(scores)
        else:
            convergence_episode = len(scores)
            
        return {
            'Algorithm': self.name,
            'Max Score': max(scores) if scores else 0,
            'Average Score': sum(scores)/len(scores) if scores else 0,
            'Training Time (min)': timestamps[-1]/60 if timestamps else 0,
            'Convergence Episode': convergence_episode,
            'Average Episode Length': sum(lengths)/len(lengths) if lengths else 0,
            'Score Std Dev': pd.Series(scores).std() if scores else 0
        }

def plot_comparison(df):
    """Visualize comparison results with multiple plot types"""
    plt.ioff()  # Turn off interactive mode
    try:
        # 1. Basic 2x2 comparison plot (as requested)
        plt.figure(figsize=(12, 8))
        basic_metrics = ['Max Score', 'Average Score', 'Training Time (min)', 'Convergence Episode']
        
        for i, metric in enumerate(basic_metrics, 1):
            plt.subplot(2, 2, i)
            df.plot.bar(x='Algorithm', y=metric, ax=plt.gca(), legend=False)
            plt.title(metric)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison_basic.png', dpi=300, bbox_inches='tight')
        print("Saved basic comparison plot to algorithm_comparison_basic.png")
        
        # 2. Comprehensive 2x3 score metrics comparison
        plt.figure(figsize=(15, 10))
        score_metrics = [
            ('Max Score', 'skyblue'),
            ('Average Score', 'lightgreen'),
            ('Score Std Dev', 'salmon'),
            ('Training Time (min)', 'gold'),
            ('Convergence Episode', 'orchid'),
            ('Average Episode Length', 'lightblue')
        ]
        
        for i, (metric, color) in enumerate(score_metrics, 1):
            plt.subplot(2, 3, i)
            df.plot.bar(x='Algorithm', y=metric, ax=plt.gca(), 
                       legend=False, color=color)
            plt.title(metric)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('algorithm_comparison_score_metrics.png', dpi=300, bbox_inches='tight')
        print("Saved score metrics comparison plot to algorithm_comparison_score_metrics.png")
        
        # 3. Learning curves comparison
        plt.figure(figsize=(12, 8))
        for algo in df['Algorithm']:
            trainer = AlgorithmTrainer(globals()[f"train_{algo.lower()}"], algo)
            trainer.train(100)  # Just to get the metrics
            plt.plot(trainer.metrics['scores'], label=algo)
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Learning Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        print("Saved learning curves to learning_curves.png")
        
    finally:
        plt.close('all')

def run_comparison(num_episodes=100):
    """Main comparison function with error handling"""
    results = []
    algorithms = [
        ("DQN", lambda n, cb: AlgorithmTrainer(train_dqn, "DQN").train(n)),
        ("A3C", lambda n, cb: AlgorithmTrainer(train_a3c, "A3C").train(n)),
        ("Policy Gradient", lambda n, cb: AlgorithmTrainer(train_pg, "Policy Gradient").train(n)),
        ("PPO", lambda n, cb: AlgorithmTrainer(train_ppo, "PPO").train(n)),
        ("SAC", lambda n, cb: AlgorithmTrainer(train_sac, "SAC").train(n))
    ]
    
    for name, train_func in algorithms:
        print(f"\n--- Training {name} ---")
        try:
            stats = train_func(num_episodes, None)
            results.append(stats)
            print(f"Completed {name}. Results: {stats}")
        except Exception as e:
            print(f"Error training {name}: {str(e)}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        
        # Save raw results
        df.to_csv('algorithm_comparison_results.csv', index=False)
        print("Saved raw results to algorithm_comparison_results.csv")
        
        # Generate plots
        plot_comparison(df)
        return df
    
    print("No results to compare")
    return None

if __name__ == '__main__':
    # Example usage with reduced episodes for testing
    test_mode = True  # Set to False for full run
    
    if test_mode:
        print("Running in test mode (500 episodes per algorithm)")
        results = run_comparison(500)
    else:
        print("Running full comparison (2000 episodes per algorithm)")
        results = run_comparison(2000)
    
    if results is not None:
        print("\nFinal Comparison Results:")
        print(results.to_string(index=False))