import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_data(n_samples=10000, n_ads=10, n_context_features=20):
    """
    Generate sample data for ad optimization
    
    Args:
        n_samples: Number of samples to generate
        n_ads: Number of different ads
        n_context_features: Number of context features
        
    Returns:
        DataFrame with sample data
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate context features
    context_data = {}
    
    # User features
    context_data['user_age'] = np.random.randint(18, 65, n_samples)
    context_data['user_gender'] = np.random.randint(0, 2, n_samples)
    context_data['user_income'] = np.random.randint(20000, 150000, n_samples)
    context_data['user_location'] = np.random.randint(0, 5, n_samples)  # 5 different locations
    
    # Time features
    timestamps = [datetime.now() - timedelta(hours=random.randint(0, 24*30)) for _ in range(n_samples)]
    context_data['hour_of_day'] = [t.hour for t in timestamps]
    context_data['day_of_week'] = [t.weekday() for t in timestamps]
    context_data['is_weekend'] = [1 if t.weekday() >= 5 else 0 for t in timestamps]
    
    # Device features
    context_data['device_type'] = np.random.randint(0, 3, n_samples)  # 0: mobile, 1: desktop, 2: tablet
    context_data['browser_type'] = np.random.randint(0, 4, n_samples)  # 4 different browsers
    
    # Content features
    context_data['content_category'] = np.random.randint(0, 8, n_samples)  # 8 different categories
    context_data['content_length'] = np.random.randint(100, 5000, n_samples)
    context_data['has_video'] = np.random.randint(0, 2, n_samples)
    
    # Additional random features
    for i in range(n_context_features - len(context_data)):
        context_data[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Generate ad IDs
    ad_ids = np.random.randint(0, n_ads, n_samples)
    
    # Generate rewards based on context and ad combinations
    rewards = np.zeros(n_samples)
    
    # Define some patterns for better rewards
    for i in range(n_samples):
        # Base probability
        prob = 0.1
        
        # User age effect
        if 25 <= context_data['user_age'][i] <= 45:
            prob += 0.05
        
        # Time of day effect
        if 9 <= context_data['hour_of_day'][i] <= 17:
            prob += 0.03
        
        # Weekend effect
        if context_data['is_weekend'][i] == 1:
            prob += 0.02
        
        # Device effect
        if context_data['device_type'][i] == 0:  # Mobile
            prob += 0.04
        
        # Content category effect
        if context_data['content_category'][i] in [1, 3, 5]:  # Popular categories
            prob += 0.03
        
        # Ad-specific effects
        if ad_ids[i] in [2, 5, 8]:  # Better performing ads
            prob += 0.05
        
        # Add some noise
        prob += np.random.normal(0, 0.02)
        
        # Ensure probability is between 0 and 1
        prob = max(0, min(1, prob))
        
        # Generate reward
        rewards[i] = np.random.binomial(1, prob)
    
    # Create DataFrame
    df = pd.DataFrame(context_data)
    df['ad_id'] = ad_ids
    df['reward'] = rewards
    
    return df

def main():
    # Generate sample data
    print("Generating sample ad optimization data...")
    df = generate_sample_data()
    
    # Save to CSV
    output_file = 'ad_data.csv'
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Number of ads: {df['ad_id'].nunique()}")
    print(f"Average CTR: {df['reward'].mean():.4f}")
    print("\nFeature Statistics:")
    print(df.describe())

if __name__ == "__main__":
    main() 