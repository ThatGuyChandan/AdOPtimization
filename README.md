# Ad Optimization with Reinforcement Learning

This project implements an ad optimization system using a hybrid approach combining Thompson Sampling and Upper Confidence Bound (UCB) algorithms. The system learns to select the most effective ads based on user context and historical performance.

## Project Files

- `ad_optimizer.py`: Main implementation of the ad optimization system
- `generate_data.py`: Data generator for creating synthetic ad optimization data
- `requirements.txt`: List of Python dependencies

## Features

- Hybrid Thompson Sampling and UCB algorithm for optimal exploration-exploitation trade-off
- Contextual modeling using deep learning
- Real-time ad selection and optimization
- Performance monitoring and evaluation

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic ad data:
```bash
python generate_data.py
```
This will create a `ad_data.csv` file with simulated ad optimization data.

2. Run the ad optimizer:
```bash
python ad_optimizer.py
```

The optimizer will:
- Load and preprocess the data
- Train the model using the hybrid Thompson Sampling-UCB approach
- Evaluate the performance on test data
- Print the results including CTR and total reward

## How It Works

1. The system takes user context and available ads as input
2. It uses a hybrid approach to select the best ad:
   - Thompson Sampling for exploration when uncertainty is high
   - UCB for exploitation when sufficient data is available
3. The selected ad is shown to the user
4. The system observes the reward (click or no click)
5. The model parameters are updated based on the observed reward
6. The process repeats for each new impression

## Results

The system will output:
- Click-Through Rate (CTR)
- Total reward
- Number of impressions
- Other performance metrics

## Future Improvements

- Integration with real ad serving platforms
- Advanced contextual modeling
- Multi-objective optimization
- Privacy-preserving techniques 