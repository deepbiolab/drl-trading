import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src import load_dataset, create_cv_folds
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
            episode_score += np.exp(reward) - 1

            if done:
                scores.append(episode_score)
                # Append mean loss for this episode
                if episode_losses:
                    losses.append(np.mean(episode_losses))

                print(
                    f"Episode {i_episode} | "
                    f"Total Return: {info['total_profit']:.4f} | "
                    f"Total Winners: {info['total_winners']:.4f} | "  
                    f"Total Losers: {info['total_losers']:.4f} | "  
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
        info['total_profit'],
    )
    plot_losses(losses, f"Episode {i_episode} DQN model loss")
    return scores, losses


def cross_validate(args):
    """Cross validation training function"""
    print("Starting cross validation...")
    
    # Load full dataset
    processed_data, _, _ = load_dataset(
        data_path=args.data_path,
        is_auto=False,
        indicator_params={
            "ma_windows": [5, 20],
            "bb_window": 20,
            "bb_std": 2,
            "vol_window": 20,
        },
        verbose=args.verbose
    )
    
    # Create CV folds
    folds = create_cv_folds(processed_data, n_folds=args.n_folds, 
                            required_cols=["Date", "Close", "BB_upper", "BB_lower"])
    cv_results = []
    
    # Train and evaluate on each fold
    for fold_idx, (train_df, val_df) in enumerate(folds):
        print(f"\nTraining Fold {fold_idx + 1}/{args.n_folds}")
        
        # Create environments
        train_env = Environment(train_df, window_size=args.window_size, verbose=False)
        val_env = Environment(val_df, window_size=args.window_size, verbose=False)
        
        # Initialize agent
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        agent = Agent(
            train_env,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            lr=args.lr,
            device=device,
        )
        
        # Train on this fold
        scores, losses = dqn(
            train_env,
            agent,
            n_episodes=args.n_episodes,
            window=args.window,
            max_t=len(train_df),
            epsilon=args.epsilon,
            eps_min=args.eps_min,
            eps_decay=args.eps_decay,
        )
        
        # Validate on validation set
        val_state = val_env.reset()
        for t in range(len(val_df)):
            action = agent.select_action(val_state)
            val_state, _, done, info = val_env.step(action)
            if done:
                break
        
        # Store results
        cv_results.append({
            'fold': fold_idx + 1,
            'train_profit': np.mean(scores),
            'val_profit': info['total_profit'],
            'train_loss': np.mean(losses) if losses else 0
        })
        
        # Save model for this fold
        agent.save(f"checkpoints/model_fold_{fold_idx + 1}.pth")
    
    # Print summary
    print("\nCross Validation Results:")
    df_results = pd.DataFrame(cv_results)
    print(df_results)
    print("\nMean Results:")
    print(f"Train Profit: {df_results['train_profit'].mean():.4f} ± {df_results['train_profit'].std():.4f}")
    print(f"Val Profit: {df_results['val_profit'].mean():.4f} ± {df_results['val_profit'].std():.4f}")
    print(f"Train Loss: {df_results['train_loss'].mean():.4f} ± {df_results['train_loss'].std():.4f}")


def train(args):
    """Training function"""
    # Create output directories if they don't exist
    model_dir = Path("checkpoints")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Starting training...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare test data
    _, train_df, test_df = load_dataset(
        data_path=args.data_path,
        is_auto=False,
        indicator_params={
            "ma_windows": [5, 20],
            "bb_window": 20,
            "bb_std": 2,
            "vol_window": 20,
        },
        train_ratio=args.train_split,
        required_cols=["Date", "Close", "BB_upper", "BB_lower"],
        verbose=args.verbose,
    )

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
    test_env = Environment(test_df, window_size=args.window_size, verbose=False)

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
            print(
                f"Total Return: {info['total_profit']:.4f}"
            )  
            print("------------------------------------------")

    # Plot final results
    plot_behavior(
        test_env,
        info["states_buy"],
        info["states_sell"],
        info['total_profit'],
        train=False,
    )
    print("Testing completed!")


def backtest(args):
    """Similarly like Testing function but for the purpose of backtesting"""
    print("Starting backtesting...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load backtest data
    backtest_df, model_data, _ = load_dataset(
        data_path=args.backtest_data,
        is_auto=False,
        indicator_params={
            "ma_windows": [5, 20],
            "bb_window": 20,
            "bb_std": 2,
            "vol_window": 20,
        },
        train_ratio=1.0,
        required_cols=["Date", "Close", "BB_upper", "BB_lower"],
        verbose=args.verbose,
    )

    trade_tracker = pd.DataFrame(
        columns=[
            "Buy Price",
            "Buy Timestamp",
            "Sell Price",
            "Sell Timestamp",
            "Buy Volume",
            "Buy MA20",
            "Buy STD20",
        ]
    )

    # Create test environment
    backtest_env = Environment(
        model_data,
        window_size=args.window_size, verbose=True
    )

    # Create and load trained agent
    agent = Agent(
        backtest_env,
        test_mode=True,
        device=device,
    )
    agent.load(filename=f"checkpoints/{args.model_path}")

    # Run test episodes
    state = backtest_env.reset()
    for t in range(backtest_df.shape[0]):
        action = agent.select_action(state)
        next_state, reward, done, info = backtest_env.step(action)
        state = next_state

        if action == 1:
            trade_tracker.loc[len(info["states_buy"]), "Buy Timestamp"] = backtest_df['Date'][t]
            trade_tracker.loc[len(info["states_buy"]), "Buy Price"] = info["buy_price"]
            trade_tracker.loc[len(info["states_buy"]), "Buy Volume"] = backtest_df["Volume"][t]
            trade_tracker.loc[len(info["states_buy"]), "Buy MA20"] = backtest_df["MA20"][t]
            trade_tracker.loc[len(info["states_buy"]), "Buy STD20"] = backtest_df["STD20"][t]

        if action == 2 and len(backtest_env.inventory) > 0:
            trade_tracker.loc[len(info["states_sell"]), "Sell Timestamp"] = backtest_df['Date'][t]
            trade_tracker.loc[len(info["states_sell"]), "Sell Price"] = info["sell_price"]

        if done:
            print("------------------------------------------")
            print(
                f"Total Return: {info['total_profit']:.4f}"
            )  
            print("------------------------------------------")

    # Plot final results
    plot_behavior(
        backtest_env,
        info["states_buy"],
        info["states_sell"],
        info['total_profit'],
        train=False,
    )

    # Save trade tracker to CSV
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    trade_tracker.to_csv(results_dir / "backtest_trades.csv")
    print("\nBacktest trade log saved to results/backtest_trades.csv")
    print("BackTesting completed!")
    return trade_tracker


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DRL Trading Agent")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "backtest"],
        default="train",
        help="Run mode: backtest, or train (train+test)",
    )

    # Data parameters
    parser.add_argument(
        "--data_path",
        type=str,
        default="datasets/AAPL_2009-2010_6m_raw_1d.csv",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--backtest_data",
        type=str,
        default="datasets/GOOG_2009-2010_6m_raw_1d.csv",
        help="Path to backtest data file",
    )
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Train/test split ratio"
    )
    parser.add_argument("--n_folds", type=int, default=5,
                      help="Number of folds for cross validation")
    parser.add_argument("--cv", action="store_true",
                      help="Whether to use cross validation")

    # Training parameters
    parser.add_argument(
        "--n_episodes", type=int, default=50, help="Number of training episodes"
    )
    parser.add_argument(
        "--window", type=int, default=1, help="Window size for averaging scores"
    )
    parser.add_argument(
        "--window_size", type=int, default=1, help="Window size for state observation"
    )
    parser.add_argument(
        "--epsilon", type=float, default=1.0, help="Initial epsilon for epsilon-greedy"
    )
    parser.add_argument(
        "--eps_min", type=float, default=0.01, help="Minimum epsilon value"
    )
    parser.add_argument(
        "--eps_decay", type=float, default=0.995, help="Epsilon decay rate"
    )

    # Agent parameters
    parser.add_argument(
        "--buffer_size", type=int, default=1000, help="Size of replay buffer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument(
        "--alpha", type=float, default=1e-3, help="Soft update parameter"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--update_step", type=int, default=4, help="Steps between network updates"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Other parameters
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoint.pth",
        help="Path to save/load model",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.cv:
        cross_validate(args)
    else:
        if args.mode == "train":
            test_df = train(args)
            test(test_df, args)

        if args.mode == "backtest":
            if args.backtest_data:
                trade_tracker = backtest(args)
