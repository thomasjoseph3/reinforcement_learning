import numpy as np
import gymnasium as gym # type: ignore
import pickle as pkl
cliff_env = gym.make("CliffWalking-v0", render_mode="ansi")
q_table = np.zeros(shape=(48, 4))


def policy(state, explore: float = 0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action


# parameters
EPSILON = 0.09
ALPHA = 0.07
GAMMA = 0.9
NUM_EPISODES = 500

for episodes in range(NUM_EPISODES):

    done = False
    (state, info) = cliff_env.reset()
    total_reward = 0
    episodes_length = 0
    action = policy(state, EPSILON)
    while not done:
        next_state, reward, terminated, truncated, info = cliff_env.step(action)
        next_action = policy(next_state, EPSILON)
        q_table[state][action] += ALPHA * (
            reward + GAMMA * q_table[next_state][next_action] - q_table[state][action]
        )
        state = next_state
        action = next_action
        total_reward += reward
        episodes_length += 1
        done = terminated or truncated
    print(total_reward, "total reward", "episode", episodes_length, "episode", episodes)
        
cliff_env.close()
pkl.dump(q_table,open("sarsa_qtable.pkl","wb"))
print("training complete")

