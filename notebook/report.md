

## **State Space**
The **state space** in reinforcement learning represents the environment in a simplified manner, enabling agents to interpret and interact effectively. In the context of financial markets, the state space is constructed using carefully selected market features that provide relevant information for the agent to make trading decisions.

### **Feature Selection Principles**
When designing the state space for a trading system, several key principles should guide feature selection:

1. **Relevance to Strategy**:
   - Features should directly align with the trading strategy objectives
   - Examples:
     * Momentum indicators for momentum-based strategies
     * Volatility indicators for volatility-based strategies
     * Price data for mean-reversion strategies

2. **Stationarity Requirements**:
   - Features should exhibit statistical stationarity to ensure reliable learning
   - Techniques for achieving stationarity:
     * Using returns instead of raw prices
     * Normalizing indicators to fixed ranges (e.g., RSI's 0-100 scale)
     * Applying difference transformations when needed

3. **Signal-to-Noise Ratio**:
   - Features should provide clear signals while minimizing noise
   - Methods to improve signal quality:
     * Appropriate smoothing techniques (e.g., moving averages)
     * Outlier detection and handling
     * Adaptive parameter selection based on market conditions

### **Market Features in Financial Data**
Market features form the foundation of trading models in reinforcement learning. These features are derived from raw financial data and provide insights into market behavior. Our preprocessing pipeline supports the following feature categories:

#### **1. Price-Based Features**
a) **Core Price Data**:
   - **OHLC (Open-High-Low-Close)**:
     * Captures price movement within trading periods
     * Useful for identifying candlestick patterns
     * Primary source for derivative indicators

b) **Return Measures**:
   - **Percentage Returns**: $(P_t - P_{t-1}) / P_{t-1}$
     * Useful for comparing price changes across different scales
     * More intuitive for human interpretation
   
   - **Log Returns**: $log(P_t / P_{t-1})$
     * Better statistical properties (more normally distributed)
     * Additive over time periods
     * Used in volatility calculations

#### **2. Volume-Based Features**
a) **Trading Volume**:
   - Raw trading volume
   - Normalized volume (relative to moving average)
   - Volume momentum indicators

b) **Price-Volume Relationships**:
   - **VWAP (Volume Weighted Average Price)**:
     * Provides fair price assessment
     * Useful for execution quality analysis
   - **Volume Profile**:
     * Distribution of trading volume across price levels
     * Identifies significant price levels

#### **3. Technical Indicators**
a) **Trend Indicators**:
   - **Moving Averages**:
     * Multiple timeframes (5, 20, 50, 200 days)
     * Adaptive window selection based on volatility
     * Types supported:
       - Simple Moving Average (SMA)
       - Exponential Moving Average (EMA)

b) **Volatility Indicators**:
   - **Bollinger Bands**:
     * Dynamic parameters based on market conditions:
       - Higher volatility: 2.5 std, 15-day window
       - Lower volatility: 1.5 std, 25-day window
     * Upper and lower bands for range identification

   - **Volatility Measures**:
     * Rolling window volatility (15-30 days)
     * Annualized volatility (252 trading days)
     * Adaptive window selection based on market characteristics

c) **Momentum Indicators**:
   - **RSI (Relative Strength Index)**:
     * 14-day standard period
     * Overbought/oversold identification
     * Range-bound nature provides stationarity

   - **Price Momentum**:
     * Multiple timeframes for different trading horizons
     * Normalized to ensure stationarity
     * Adaptive calculation based on market regime

> All these features are automatically processed through our pipeline with:
> - Missing value handling
> - Appropriate normalization
> - Automatic parameter selection based on data characteristics
> - Quality checks and validation
   

## **Action Space**
In reinforcement learning, **actions** are the outputs of the agent, representing the behavior it takes based on the current state. In the context of financial trading, actions typically include:  
- **Buy**: Purchase a stock.  
- **Sell**: Sell a stock.  
- **Hold**: Maintain the current position without any action.

The **action space** defines the set of all possible actions the agent can take. The design of the action space is critical to the model's performance and must align with the trading strategy.

### **Types of Action Spaces**:
#### 1. **Discrete Action Space**:
- Actions are finite and discrete, such as "buy 10 shares," "sell 10 shares," or "hold."  
- This type of action space is commonly used with algorithms like **Deep Q-Networks (DQN)**.  
- **DQN Setting for Discrete Actions**:
  - **Action Set**: Define a fixed set of actions, such as:
    - `Buy 1 unit`
    - `Sell 1 unit`
    - `Hold`
  - **Advantages**:
    - Simpler to implement and computationally efficient.
    - Suitable for environments where actions can be predefined and are limited in number.
  - **Limitations**:
    - Lack of flexibility in scenarios requiring fine-grained control, e.g., trading fractional shares or adjusting position sizes dynamically.

#### 2. **Continuous Action Space**:
- Actions are continuous, such as "buy between 0 and 100 shares" or "sell between 0 and 100 shares."  
- This type of action space is better suited for algorithms like **Deep Deterministic Policy Gradient (DDPG)**, which can handle continuous control problems.  
- **Future Plan with DDPG**:
  - **Action Set**: Allow the agent to select actions from a continuous range, such as:
    - `Buy x units` where $x \in [0, 100]$
    - `Sell y units` where $y \in [0, 100]$
  - **Advantages**:
    - Provides greater flexibility and precision in decision-making.
    - Ideal for scenarios requiring dynamic position sizing or fractional trading.
  - **Challenges**:
    - Requires more complex exploration strategies (e.g., adding noise to actions for exploration).
    - Computationally more intensive compared to discrete action spaces.


### **Key Factors in Action Space Design**:
1. **Defining the Action Set**:  
   - Actions should reflect realistic trading decisions and align with the strategy. For example, "buy 1 share" or "sell 1 share."

2. **Market Constraints**:  
   - Incorporate real-world constraints such as transaction costs, minimum order quantities, and liquidity. For example, if the minimum order size is 50 shares, the action set should be defined in multiples of 50.

3. **Action Granularity**:  
   - Finer granularity allows for more precise control but increases the size of the action space, leading to higher computational complexity. For example, "buy 10 shares" provides finer control than "buy 50 shares."


### **Adjusting Action Space in Real-Time Trading**
In live trading, the action space may need to adapt dynamically to market conditions:

1. **Calibrating Granularity and Action Combinations**:  
   - Adjust the granularity of actions based on changes in market conditions, such as stock price volatility. For instance, when stock prices rise, the model could be allowed to trade smaller quantities to reduce risk.

2. **Risk Management Actions**:  
   - Include stop-loss and take-profit actions as part of the action space:  
     - **Stop-Loss Action**: Automatically sell a stock when its loss reaches a defined threshold (e.g., a $10 loss) to minimize further losses.  
     - **Take-Profit Action**: Automatically sell a stock when its profit reaches a defined threshold (e.g., a $10 profit) to lock in gains.



## **Rewards Design**
The reward function is a crucial component in reinforcement learning that defines the value of each action. In financial trading, rewards are typically aligned with trading profits while considering various aspects of trading behavior.

### **Core Principles**
1. **Profit-Based Rewards**:
   - Primary driver of agent behavior
   - Two main approaches:
     * **Raw Profit**: Direct price difference between buy and sell
     * **Log Returns**: Natural logarithm of price changes

2. **Action-Specific Rewards**:
   - **Buy Actions**: Usually zero by default
   - **Hold Actions**: Can be adjusted to encourage/discourage holding periods
   - **Sell Actions**: Based on trading profit/returns

3. **Temporal Considerations**:
   - Short-term vs. long-term rewards
   - Trade frequency optimization
   - Position holding duration

### **Reward Calculation Methods**

#### **1. Raw Profit Approach**
```python
reward = sell_price - buy_price
```
- **Advantages**:
  * Simple and intuitive
  * Direct relationship to actual profits
- **Limitations**:
  * Asymmetric returns (50% loss requires 100% gain to recover)
  * Scale dependency
  * Non-additive nature

#### **2. Logarithmic Returns** (Recommended)
```python
reward = log(sell_price / buy_price)
```
- **Advantages**:
  * **Additivity**: Multiple trade returns can be summed directly
  * **Symmetry**: Order of trades doesn't affect cumulative returns
  * **Scale Independence**: Works across different price levels
  * Better statistical properties for learning

### **Reward Function Components**

#### **1. Base Trading Rewards**
- **Buy Actions**: $R_{buy} = 0$ (default)
- **Hold Actions**: $R_{hold} = 0$ (default)
- **Sell Actions**: $R_{sell} = log(P_{sell} / P_{buy})$

#### **2. Behavioral Adjustments**
Rewards can be modified to encourage specific trading behaviors:

1. **Trading Frequency**:
   ```python
   if action == 'buy':
       reward += buy_incentive  # Increase for more frequent trading
   ```

2. **Holding Duration**:
   ```python
   if action == 'hold':
       reward += hold_incentive  # Increase for longer holding periods
   ```

3. **Risk Management**:
   ```python
   if stop_loss_triggered:
       reward -= stop_loss_penalty
   if take_profit_triggered:
       reward += take_profit_bonus
   ```

### **Implementation Guidelines**

#### **1. Initial Setup**
1. Start with basic profit-based rewards:
   - Buy/Hold rewards = 0
   - Sell rewards = logarithmic returns

2. Monitor agent behavior:
   - Trading frequency
   - Position holding periods
   - Risk management effectiveness

#### **2. Iterative Adjustment**
1. **Trading Frequency Optimization**:
   - If too frequent: Reduce buy rewards or add trading costs
   - If too infrequent: Increase buy rewards

2. **Position Management**:
   - If holding too short: Increase hold rewards
   - If holding too long: Decrease hold rewards or add time decay

3. **Risk-Reward Balance**:
   - Adjust stop-loss and take-profit rewards
   - Implement position sizing rewards
   - Consider market volatility in reward scaling

### **Best Practices**

1. **Reward Scale**:
   - Keep rewards in a reasonable range (-1 to 1 or -10 to 10)
   - Normalize rewards if using different components
   - Maintain consistent scale across episodes

2. **Reward Components Balance**:
   ```python
   total_reward = (
       w1 * profit_reward +
       w2 * behavioral_adjustment +
       w3 * risk_management_reward
   )
   ```
   - Carefully tune weights (w1, w2, w3)
   - Ensure primary profit objective remains dominant

3. **Monitoring and Validation**:
   - Track reward statistics across episodes
   - Validate reward function against trading objectives
   - Adjust based on agent's learning progress

> **Note**: The reward function should be designed to align with your trading strategy's objectives while maintaining mathematical properties that facilitate learning (additivity, symmetry, and proper scaling).
