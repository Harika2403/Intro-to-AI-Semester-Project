import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import time
import os

class MetricsTracker:
    def __init__(self, window_size=100):
        self.scores = []
        self.episode_lengths = []
        self.timestamps = []
        self.start_time = time.time()
        self.window_size = window_size
        self.moving_avg_scores = []
        self.max_score = 0
        
    def record_episode(self, score, episode_length):
        self.scores.append(score)
        self.episode_lengths.append(episode_length)
        self.timestamps.append(time.time() - self.start_time)
        
        if score > self.max_score:
            self.max_score = score
            
        # Calculate moving average
        if len(self.scores) >= self.window_size:
            window_scores = self.scores[-self.window_size:]
            self.moving_avg_scores.append(np.mean(window_scores))
    
    def plot_metrics(self, algorithm_name):
        """Thread-safe metric plotting that saves to file instead of showing"""
        plt.ioff()  # Turn off interactive mode
        fig = plt.figure(figsize=(12, 6))
        
        try:
            # Plot scores
            ax1 = plt.subplot(1, 2, 1)
            ax1.plot(self.scores, label='Scores')
            if self.moving_avg_scores:
                ax1.plot(range(self.window_size-1, len(self.scores)), 
                        self.moving_avg_scores, label=f'MA ({self.window_size})')
            ax1.set_title(f'{algorithm_name} - Scores')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Score')
            ax1.legend()
            
            # Plot episode lengths
            ax2 = plt.subplot(1, 2, 2)
            ax2.plot(self.episode_lengths)
            ax2.set_title(f'{algorithm_name} - Episode Lengths')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Steps')
            
            plt.tight_layout()
            
            # Ensure directory exists
            os.makedirs('metrics_plots', exist_ok=True)
            plot_path = f'metrics_plots/{algorithm_name}_metrics.png'
            plt.savefig(plot_path)
            print(f"Saved metrics plot to {plot_path}")
            
        finally:
            plt.close(fig)
            plt.ion()  # Restore interactive mode if needed
    
    def get_summary_stats(self):
        """Calculate and return summary statistics"""
        avg_score = np.mean(self.scores) if self.scores else 0
        avg_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
        training_time = self.timestamps[-1] if self.timestamps else 0
        
        # Calculate convergence episode
        convergence_ep = len(self.scores)  # Default to full length if no convergence
        if len(self.moving_avg_scores) > 50:
            for i in range(len(self.moving_avg_scores)-50):
                window = self.moving_avg_scores[i:i+50]
                if np.std(window) < 0.1 * np.mean(window):  # <10% variation
                    convergence_ep = i + self.window_size
                    break
        
        return {
            'max_score': self.max_score,
            'avg_score': avg_score,
            'avg_episode_length': avg_length,
            'training_time': training_time,
            'convergence_episode': convergence_ep
        }

    def save_to_csv(self, algorithm_name):
        """Save metrics data to CSV for later analysis"""
        import pandas as pd
        data = {
            'episode': range(1, len(self.scores)+1),
            'score': self.scores,
            'episode_length': self.episode_lengths,
            'timestamp': self.timestamps
        }
        df = pd.DataFrame(data)
        os.makedirs('metrics_data', exist_ok=True)
        csv_path = f'metrics_data/{algorithm_name}_metrics.csv'
        df.to_csv(csv_path, index=False)
        print(f"Saved metrics data to {csv_path}")
        return csv_path