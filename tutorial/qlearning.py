import numpy as np
import gymnasium as gym # type: ignore
import pickle as pkl

cliff_env = gym.make("CliffWalking-v0", render_mode="ansi")
q_table = np.zeros((48, 4))

# Q-learning parameters
EPSILON = 0.1  # Exploration rate
ALPHA = 0.1    # Learning rate
GAMMA = 0.9    # Discount factor
NUM_EPISODES = 500


def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action

for episode in range(NUM_EPISODES):
    done = False
    (state, info) = cliff_env.reset()
    total_reward = 0
    episode_length = 0
    while not done:
        action = policy(state, EPSILON)
        next_state, reward, terminated, truncated, info = cliff_env.step(action)
        # Q-learning update: use max Q for next state (off-policy)
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[next_state][best_next_action] - q_table[state][action]
        )
        state = next_state
        total_reward += reward
        episode_length += 1
        done = terminated or truncated
    print(total_reward, "total reward", "episode", episode_length, "episode", episode)

cliff_env.close()
pkl.dump(q_table, open("qlearning_qtable.pkl", "wb"))
print("Q-learning training complete")
