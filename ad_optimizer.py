import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizerConfig:
    """Configuration for the AdOptimizer"""
    n_ads: int
    context_dim: int
    learning_rate: float = 0.001
    batch_size: int = 32
    hidden_dims: List[int] = None
    exploration_rate: float = 0.1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    multi_objective: bool = True
    real_time_bidding: bool = True
    user_profiling: bool = True
    advanced_context: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [128, 64, 32]  # Deeper network

class AdvancedContextualModel(nn.Module):
    def __init__(self, input_dim: int, n_ads: int, hidden_dims: List[int], 
                 learning_rate: float, device: torch.device):
        super(AdvancedContextualModel, self).__init__()
        self.device = device
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_dims[0]),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dims[0], num_heads=4)
        
        # Deep layers
        layers = []
        prev_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.deep_layers = nn.Sequential(*layers)
        
        # Multi-objective output heads
        self.ctr_head = nn.Linear(prev_dim, n_ads)
        self.conversion_head = nn.Linear(prev_dim, n_ads)
        self.revenue_head = nn.Linear(prev_dim, n_ads)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Self-attention
        features = features.unsqueeze(0)  # Add sequence dimension
        features, _ = self.attention(features, features, features)
        features = features.squeeze(0)
        
        # Deep processing
        deep_features = self.deep_layers(features)
        
        # Multi-objective predictions
        return {
            'ctr': self.ctr_head(deep_features),
            'conversion': self.conversion_head(deep_features),
            'revenue': self.revenue_head(deep_features)
        }
    
    def update(self, context: np.ndarray, ad_idx: int, rewards: Dict[str, float]):
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        
        # Create target tensors for each objective
        targets = {
            'ctr': torch.zeros(1, self.ctr_head.out_features).to(self.device),
            'conversion': torch.zeros(1, self.conversion_head.out_features).to(self.device),
            'revenue': torch.zeros(1, self.revenue_head.out_features).to(self.device)
        }
        
        for objective, reward in rewards.items():
            targets[objective][0, ad_idx] = reward
        
        self.optimizer.zero_grad()
        predictions = self(context_tensor)
        
        # Calculate loss for each objective
        total_loss = 0
        for objective in ['ctr', 'conversion', 'revenue']:
            loss = self.criterion(predictions[objective], targets[objective])
            total_loss += loss
        
        total_loss.backward()
        self.optimizer.step()

class UserProfile:
    def __init__(self):
        self.interaction_history = []
        self.preferences = {}
        self.last_update = datetime.now()
    
    def update(self, ad_id: int, reward: float, context: np.ndarray):
        self.interaction_history.append({
            'ad_id': ad_id,
            'reward': reward,
            'context': context,
            'timestamp': datetime.now()
        })
        self._update_preferences()
    
    def _update_preferences(self):
        # Update user preferences based on interaction history
        recent_interactions = self.interaction_history[-100:]  # Last 100 interactions
        if not recent_interactions:
            return
        
        # Calculate preference scores for different ad categories
        for interaction in recent_interactions:
            ad_id = interaction['ad_id']
            reward = interaction['reward']
            if ad_id not in self.preferences:
                self.preferences[ad_id] = {'score': 0, 'count': 0}
            self.preferences[ad_id]['score'] += reward
            self.preferences[ad_id]['count'] += 1
    
    def get_preference_score(self, ad_id: int) -> float:
        if ad_id not in self.preferences:
            return 0.0
        pref = self.preferences[ad_id]
        return pref['score'] / max(pref['count'], 1)

class AdOptimizer:
    def __init__(self, config: OptimizerConfig):
        """
        Initialize the ad optimizer with advanced features
        
        Args:
            config: OptimizerConfig object containing initialization parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Thompson Sampling parameters
        self.alpha = np.ones(config.n_ads)
        self.beta = np.ones(config.n_ads)
        
        # UCB parameters
        self.reward_sums = np.zeros(config.n_ads)
        self.n_plays = np.zeros(config.n_ads)
        
        # Advanced contextual model
        self.context_model = AdvancedContextualModel(
            input_dim=config.context_dim,
            n_ads=config.n_ads,
            hidden_dims=config.hidden_dims,
            learning_rate=config.learning_rate,
            device=self.device
        )
        
        # User profiling
        self.user_profiles = {}
        
        # Performance tracking
        self.performance_history = []
        self.best_ctr = 0.0
        self.multi_objective_metrics = {
            'ctr': [],
            'conversion': [],
            'revenue': []
        }
        
        logger.info(f"Initialized AdOptimizer with {config.n_ads} ads and {config.context_dim} context dimensions")
    
    def _get_user_id(self, context: np.ndarray) -> str:
        """Generate a unique user ID from context"""
        # Use a hash of relevant context features as user ID
        user_features = context[:5]  # Use first 5 features for user identification
        return hashlib.md5(user_features.tobytes()).hexdigest()
    
    def select_ad(self, context: np.ndarray, user_id: Optional[str] = None) -> int:
        """
        Select an ad using the hybrid approach with advanced features
        
        Args:
            context: Context vector for the current user/impression
            user_id: Optional user identifier for personalization
            
        Returns:
            Selected ad index
        """
        # Get uncertainty level
        uncertainty = self._calculate_uncertainty()
        
        # Get contextual predictions
        context_tensor = torch.FloatTensor(context).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.context_model(context_tensor)
            contextual_pred = predictions['ctr'].cpu().numpy()[0]
        
        # Get user profile influence if available
        user_influence = np.zeros(self.config.n_ads)
        if self.config.user_profiling:
            user_id = user_id or self._get_user_id(context)
            if user_id in self.user_profiles:
                for ad_idx in range(self.config.n_ads):
                    user_influence[ad_idx] = self.user_profiles[user_id].get_preference_score(ad_idx)
        
        if uncertainty > self.config.exploration_rate:
            # High uncertainty: Thompson Sampling
            theta = np.random.beta(self.alpha, self.beta)
            combined_score = (
                0.5 * theta +  # Thompson Sampling
                0.3 * contextual_pred +  # Contextual prediction
                0.2 * user_influence  # User profile influence
            )
        else:
            # Low uncertainty: UCB
            ucb_values = self._calculate_ucb()
            combined_score = (
                0.5 * ucb_values +  # UCB
                0.3 * contextual_pred +  # Contextual prediction
                0.2 * user_influence  # User profile influence
            )
        
        selected_ad = np.argmax(combined_score)
        return int(selected_ad)
    
    def update(self, ad_idx: int, reward: float, context: np.ndarray, 
              user_id: Optional[str] = None, additional_rewards: Optional[Dict[str, float]] = None):
        """
        Update the model parameters based on observed rewards
        
        Args:
            ad_idx: Index of the selected ad
            reward: Observed reward (0 or 1 for clicks)
            context: Context vector for the current user/impression
            user_id: Optional user identifier for personalization
            additional_rewards: Optional dictionary of additional rewards (conversion, revenue)
        """
        # Convert ad_idx to integer
        ad_idx = int(ad_idx)
        
        # Update Thompson Sampling parameters
        self.alpha[ad_idx] += reward
        self.beta[ad_idx] += (1 - reward)
        
        # Update UCB parameters
        self.reward_sums[ad_idx] += reward
        self.n_plays[ad_idx] += 1
        
        # Update user profile
        if self.config.user_profiling:
            user_id = user_id or self._get_user_id(context)
            if user_id not in self.user_profiles:
                self.user_profiles[user_id] = UserProfile()
            self.user_profiles[user_id].update(ad_idx, reward, context)
        
        # Update contextual model
        rewards = {'ctr': reward}
        if additional_rewards:
            rewards.update(additional_rewards)
        self.context_model.update(context, ad_idx, rewards)
        
        # Track performance
        self._update_performance_history(reward, additional_rewards)
    
    def _calculate_uncertainty(self) -> float:
        """Calculate the current uncertainty level"""
        total_plays = np.sum(self.n_plays)
        if total_plays == 0:
            return 1.0
        return 1.0 / (1.0 + np.sqrt(total_plays))
    
    def _calculate_ucb(self) -> np.ndarray:
        """Calculate UCB values for all ads"""
        t = np.sum(self.n_plays)
        if t == 0:
            return np.zeros(self.config.n_ads)
            
        ucb_values = np.zeros(self.config.n_ads)
        for i in range(self.config.n_ads):
            if self.n_plays[i] > 0:
                mean_reward = self.reward_sums[i] / self.n_plays[i]
                confidence = np.sqrt(2 * np.log(t) / self.n_plays[i])
                ucb_values[i] = mean_reward + confidence
        return ucb_values
    
    def _update_performance_history(self, reward: float, additional_rewards: Optional[Dict[str, float]] = None):
        """Update performance tracking metrics"""
        self.performance_history.append(reward)
        current_ctr = np.mean(self.performance_history[-1000:])
        
        if additional_rewards:
            for metric, value in additional_rewards.items():
                self.multi_objective_metrics[metric].append(value)
        
        if current_ctr > self.best_ctr:
            self.best_ctr = current_ctr
            self._save_best_model()
    
    def _save_best_model(self):
        """Save the best performing model"""
        state = {
            'context_model': self.context_model.state_dict(),
            'alpha': self.alpha,
            'beta': self.beta,
            'reward_sums': self.reward_sums,
            'n_plays': self.n_plays,
            'best_ctr': self.best_ctr,
            'user_profiles': self.user_profiles
        }
        torch.save(state, 'best_model.pth')
        logger.info(f"Saved new best model with CTR: {self.best_ctr:.4f}")
    
    def load_best_model(self):
        """Load the best performing model"""
        try:
            state = torch.load('best_model.pth')
            self.context_model.load_state_dict(state['context_model'])
            self.alpha = state['alpha']
            self.beta = state['beta']
            self.reward_sums = state['reward_sums']
            self.n_plays = state['n_plays']
            self.best_ctr = state['best_ctr']
            self.user_profiles = state.get('user_profiles', {})
            logger.info(f"Loaded best model with CTR: {self.best_ctr:.4f}")
        except FileNotFoundError:
            logger.warning("No saved model found")

class AdDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.contexts = data.iloc[:, :-2].values
        self.ads = data.iloc[:, -2].values.astype(int)  # Ensure integer type
        self.rewards = data.iloc[:, -1].values
        
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.contexts[idx]),
            torch.LongTensor([self.ads[idx]]),
            torch.FloatTensor([self.rewards[idx]])
        )

def load_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and preprocess the ad optimization dataset
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        Tuple of (train_data, test_data)
    """
    # Load the dataset
    data = pd.read_csv(file_path)
    
    # Ensure ad_id column is integer
    data.iloc[:, -2] = data.iloc[:, -2].astype(int)
    
    # Preprocess the data
    scaler = StandardScaler()
    context_cols = data.columns[:-2]  # Assuming last two columns are ad_id and reward
    data[context_cols] = scaler.fit_transform(data[context_cols])
    
    # Split into train and test
    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    return train_data, test_data

def evaluate_model(optimizer: AdOptimizer, test_data: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate the ad optimizer on test data
    
    Args:
        optimizer: AdOptimizer instance
        test_data: Test dataset
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_reward = 0
    n_impressions = len(test_data)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        for _, row in test_data.iterrows():
            context = row.iloc[:-2].values
            ad_idx = int(row.iloc[-2])
            reward = row.iloc[-1]
            
            future = executor.submit(optimizer.select_ad, context)
            futures.append((future, ad_idx, reward))
        
        for future, ad_idx, reward in futures:
            selected_ad = future.result()
            if selected_ad == ad_idx:
                total_reward += reward
    
    ctr = total_reward / n_impressions
    return {
        'CTR': ctr,
        'Total Reward': total_reward,
        'Impressions': n_impressions
    }

def plot_performance_metrics(metrics_history: List[Dict[str, float]], save_path: str = 'performance_metrics.png'):
    """Plot performance metrics over time"""
    plt.figure(figsize=(12, 8))
    
    # Extract metrics
    ctrs = [m['CTR'] for m in metrics_history]
    rewards = [m['Total Reward'] for m in metrics_history]
    impressions = [m['Impressions'] for m in metrics_history]
    
    # Plot CTR
    plt.subplot(2, 2, 1)
    plt.plot(ctrs, label='CTR')
    plt.title('Click-Through Rate Over Time')
    plt.xlabel('Batch')
    plt.ylabel('CTR')
    plt.grid(True)
    
    # Plot Total Reward
    plt.subplot(2, 2, 2)
    plt.plot(rewards, label='Total Reward')
    plt.title('Total Reward Over Time')
    plt.xlabel('Batch')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    # Plot Impressions
    plt.subplot(2, 2, 3)
    plt.plot(impressions, label='Impressions')
    plt.title('Number of Impressions Over Time')
    plt.xlabel('Batch')
    plt.ylabel('Impressions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_ad_selection_distribution(optimizer: AdOptimizer):
    """Plot the distribution of ad selections"""
    plt.figure(figsize=(10, 6))
    
    # Calculate selection probabilities
    total_plays = np.sum(optimizer.n_plays)
    if total_plays > 0:
        selection_probs = optimizer.n_plays / total_plays
        plt.bar(range(len(selection_probs)), selection_probs)
        plt.title('Ad Selection Distribution')
        plt.xlabel('Ad Index')
        plt.ylabel('Selection Probability')
        plt.grid(True)
        plt.savefig('ad_selection_distribution.png')
        plt.close()

def plot_algorithm_comparison(optimizer: AdOptimizer):
    """Plot comparison between Thompson Sampling and UCB performance"""
    plt.figure(figsize=(12, 6))
    
    # Calculate mean rewards for each algorithm
    ts_rewards = optimizer.alpha / (optimizer.alpha + optimizer.beta)
    ucb_rewards = optimizer.reward_sums / np.maximum(optimizer.n_plays, 1)
    
    x = np.arange(len(ts_rewards))
    width = 0.35
    
    plt.bar(x - width/2, ts_rewards, width, label='Thompson Sampling')
    plt.bar(x + width/2, ucb_rewards, width, label='UCB')
    
    plt.title('Algorithm Performance Comparison')
    plt.xlabel('Ad Index')
    plt.ylabel('Mean Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('algorithm_comparison.png')
    plt.close()

def main():
    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    train_data, test_data = load_data('ad_data.csv')
    
    # Initialize optimizer
    config = OptimizerConfig(
        n_ads=train_data.iloc[:, -2].nunique(),
        context_dim=train_data.shape[1] - 2
    )
    optimizer = AdOptimizer(config)
    
    # Train the model and collect metrics
    logger.info("Training the model...")
    metrics_history = []
    batch_size = 1000  # Evaluate every 1000 impressions
    
    for i, (_, row) in enumerate(tqdm(train_data.iterrows(), total=len(train_data))):
        context = row.iloc[:-2].values
        ad_idx = int(row.iloc[-2])
        reward = row.iloc[-1]
        
        optimizer.update(ad_idx, reward, context)
        
        # Evaluate periodically
        if (i + 1) % batch_size == 0:
            metrics = evaluate_model(optimizer, test_data)
            metrics_history.append(metrics)
            logger.info(f"Batch {i+1}: CTR = {metrics['CTR']:.4f}")
    
    # Generate plots
    logger.info("Generating visualizations...")
    plot_performance_metrics(metrics_history)
    plot_ad_selection_distribution(optimizer)
    plot_algorithm_comparison(optimizer)
    
    # Print final metrics
    logger.info("\nFinal Evaluation Results:")
    final_metrics = metrics_history[-1]
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Save configuration
    with open('optimizer_config.json', 'w') as f:
        json.dump(config.__dict__, f, indent=4)

if __name__ == "__main__":
    main() 