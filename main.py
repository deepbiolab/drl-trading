import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src import Agent, Environment
from src import plot_behavior, plot_losses


def dqn(
    env,
    agent,
    n_episodes=2,
    window=10,
    max_t=1000,
    epsilon=1.0,
    eps_min=0.01,
    eps_decay=0.995,
):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        epsilon (float): starting value of epsilon, for epsilon-greedy action selection
        eps_min (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # list containing scores and losses from each episode
    scores = []
    losses = []
    max_avg_score = float("-inf")

    for i_episode in range(1, n_episodes + 1):
        # get init state
        state = env.reset()
        episode_score = 0
        episode_losses = []  # Reset episode losses

        # run each episode
        for t in range(max_t):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, info = env.step(action)
            loss = agent.step(state, action, reward, next_state, done)

            # Collect loss if not None
            if loss is not None:
                episode_losses.append(loss.item())

            state = next_state
            episode_score += reward

            if done:
                scores.append(episode_score)
                # Append mean loss for this episode
                if episode_losses:
                    losses.append(np.mean(episode_losses))

                print(
                    f"Episode {i_episode} | "
                    f"Total Return: {np.exp(info['total_profit']) - 1:.4f} | "
                    f"Total Winners: {np.exp(info['total_winners']) - 1:.4f} | "  # Convert log return
                    f"Total Losers: {np.exp(info['total_losers']) - 1:.4f} | "   # Convert log return
                    f"Average Loss: {losses[-1] if losses else 0}"
                )
                break

        # Update epsilon
        epsilon = max(eps_min, eps_decay * epsilon)

        # Print episode stats
        if len(scores) > window:
            avg_score = np.mean(scores[-window:])
            if avg_score > max_avg_score:
                max_avg_score = avg_score
                agent.save("checkpoints/checkpoint.pth")

        if i_episode % window == 0:
            print(
                "\rEpisode {}/{} | Max Average Score: {:.2f}".format(
                    i_episode, n_episodes, max_avg_score
                ),
            )

    plot_behavior(
        env,
        info["states_buy"],
        info["states_sell"],
        info["total_profit"],
    )
    plot_losses(losses, f"Episode {i_episode} DQN model loss")
    return scores, losses


def train(args):
    """Training function"""
    print("Starting training...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    data = pd.read_csv(args.data_path)
    state_features = ["Date", "Close", "BB_upper", "BB_lower"]
    data = data[state_features]

    # Split dataset
    training_rows = int(len(data.index) * args.train_split)
    train_df = data.loc[:training_rows].set_index("Date")
    test_df = data.loc[training_rows:].set_index("Date")

    # Create environment and agent
    env = Environment(train_df, window_size=args.window_size, verbose=args.verbose)

    agent = Agent(
        env,
        test_mode=False,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        alpha=args.alpha,
        lr=args.lr,
        update_step=args.update_step,
        seed=args.seed,
        device=device,
    )

    # Train the agent
    scores, losses = dqn(
        env,
        agent,
        n_episodes=args.n_episodes,
        window=args.window,
        max_t=train_df.shape[0],
        epsilon=args.epsilon,
        eps_min=args.eps_min,
        eps_decay=args.eps_decay,
    )

    print("Training completed!")
    return test_df  # Return test data for testing phase


def test(test_df, args):
    """Testing function"""
    print("Starting testing...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Create test environment
    test_env = Environment(test_df, window_size=args.window_size, verbose=True)

    # Create and load trained agent
    agent = Agent(
        test_env,
        test_mode=True,
        device=device,
    )
    agent.load(filename=f"checkpoints/{args.model_path}")

    # Run test episodes
    state = test_env.reset()
    for t in range(test_df.shape[0]):
        action = agent.select_action(state)
        next_state, reward, done, info = test_env.step(action)
        state = next_state

        if done:
            print("------------------------------------------")
            print(f"Total Return: {np.exp(info['total_profit']) - 1:.4f}")  # Convert log return
            print("------------------------------------------")

    # Plot final results
    plot_behavior(
        test_env,
        info["states_buy"],
        info["states_sell"],
        info["total_profit"],
        train=False,
    )
    print("Testing completed!")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DRL Trading Agent")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="datasets/AAPL_2009-2010_6m_features_1d.csv",
                      help='Path to the dataset')
    parser.add_argument('--train_split', type=float, default=0.8,
                      help='Train/test split ratio')
    
    # Training parameters
    parser.add_argument('--n_episodes', type=int, default=5,
                      help='Number of training episodes')
    parser.add_argument('--window', type=int, default=1,
                      help='Window size for averaging scores')
    parser.add_argument('--window_size', type=int, default=1,
                      help='Window size for state observation')
    parser.add_argument('--epsilon', type=float, default=1.0,
                      help='Initial epsilon for epsilon-greedy')
    parser.add_argument('--eps_min', type=float, default=0.01,
                      help='Minimum epsilon value')
    parser.add_argument('--eps_decay', type=float, default=0.995,
                      help='Epsilon decay rate')
    
    # Agent parameters
    parser.add_argument('--buffer_size', type=int, default=1000,
                      help='Size of replay buffer')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=0.95,
                      help='Discount factor')
    parser.add_argument('--alpha', type=float, default=1e-3,
                      help='Soft update parameter')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--update_step', type=int, default=4,
                      help='Steps between network updates')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    # Other parameters
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--model_path', type=str, default="checkpoint.pth",
                      help='Path to save/load model')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
                      default='both', help='Run mode')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Create output directories if they don't exist
    model_dir = Path('checkpoints')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode in ['train', 'both']:
        test_df = train(args)
        
    if args.mode in ['test', 'both']:
        if args.mode == 'test':
            # Load test data if only testing
            data = pd.read_csv(args.data_path)
            state_features = ["Date", "Close", "BB_upper", "BB_lower"]
            data = data[state_features]
            training_rows = int(len(data.index) * args.train_split)
            test_df = data.loc[training_rows:].set_index("Date")
        
        test(test_df, args)
