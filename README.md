# Reinforcement Learning-Based Trading System

![](assets/rl-trading.svg)

## **System Workflow**
To prevent the issue where pre-filtering logic (e.g., stop-loss or take-profit rules) continuously overrides the model's decision-making process, the following improved workflow is proposed:

1. **State Input**:  
   - The market state is fed into the system.

2. **Filtering Rules Check**:  
   - If a filtering rule (e.g., stop-loss or take-profit) is triggered, the corresponding action (e.g., "sell") is recorded.  
   - The updated state and the triggered action are then passed to the model.

3. **Model Decision**:  
   - The model makes a decision based on the updated state.  
   - If the model’s action conflicts with the action triggered by the filtering rule, the filtering rule's action takes priority.

4. **Action Execution**:  
   - The final action (either the rule-triggered action or the model’s action) is executed.

5. **State Update**:  
   - The market state is updated, and the system proceeds to the next decision-making cycle.


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


## **Adjusting Action Space in Real-Time Trading**
In live trading, the action space may need to adapt dynamically to market conditions:

1. **Calibrating Granularity and Action Combinations**:  
   - Adjust the granularity of actions based on changes in market conditions, such as stock price volatility. For instance, when stock prices rise, the model could be allowed to trade smaller quantities to reduce risk.

2. **Risk Management Actions**:  
   - Include stop-loss and take-profit actions as part of the action space:  
     - **Stop-Loss Action**: Automatically sell a stock when its loss reaches a defined threshold (e.g., a $10 loss) to minimize further losses.  
     - **Take-Profit Action**: Automatically sell a stock when its profit reaches a defined threshold (e.g., a $10 profit) to lock in gains.

