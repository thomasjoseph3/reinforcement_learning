# Cliff Walking RL Agents: SARSA & Q-learning

This project demonstrates two classic reinforcement learning (RL) algorithms—SARSA and Q-learning—on the OpenAI Gymnasium CliffWalking-v0 environment. It includes code for training, saving, and evaluating both agents, with visualizations using OpenCV.

## Project Structure

- `sarsa.py` — Trains a SARSA agent and saves the learned Q-table to `sarsa_qtable.pkl`.
- `qlearning.py` — Trains a Q-learning agent and saves the learned Q-table to `qlearning_qtable.pkl`.
- `evaluate.py` — Loads the SARSA Q-table and visually evaluates the agent's performance.
- `evaluate_qlearning.py` — Loads the Q-learning Q-table and visually evaluates the agent's performance.
- `requirements.txt` — Lists required Python packages (e.g., numpy, gymnasium, opencv-python).

## How to Use

### 1. Install Requirements
```
pip install -r requirements.txt
```

### 2. Train an Agent
- **SARSA:**
  ```
  python sarsa.py
  ```
  This will create `sarsa_qtable.pkl`.

- **Q-learning:**
  ```
  python qlearning.py
  ```
  This will create `qlearning_qtable.pkl`.

### 3. Evaluate an Agent
- **SARSA Evaluation:**
  ```
  python evaluate.py
  ```
- **Q-learning Evaluation:**
  ```
  python evaluate_qlearning.py
  ```
  Both scripts will open a window showing the agent's path for several episodes and print episode statistics.

## Key Concepts

- **SARSA (On-policy):** Updates Q-values using the action actually taken, reflecting the agent's real behavior (including exploration).
- **Q-learning (Off-policy):** Updates Q-values using the best possible next action, learning the optimal policy regardless of exploration.
- **CliffWalking-v0:** A gridworld where the agent must reach the goal while avoiding the cliff (which gives a large negative reward).

## Customization
- You can tune hyperparameters (`ALPHA`, `EPSILON`, `GAMMA`, `NUM_EPISODES`) in the training scripts to experiment with learning speed and behavior.
- The evaluation scripts can be modified to run more or fewer episodes, or to change the visualization speed.

## Requirements
- Python 3.x
- numpy
- gymnasium
- opencv-python
- pickle (standard library)

## References
- [OpenAI Gymnasium Documentation](https://gymnasium.farama.org/)
- Sutton & Barto, "Reinforcement Learning: An Introduction"

---

Feel free to explore and modify the code to deepen your understanding of RL algorithms!
