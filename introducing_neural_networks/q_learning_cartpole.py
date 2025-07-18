import gymnasium as gym
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

# Initialize the environment
env = gym.make("CartPole-v1")

# Q-Network
net_input = Input(shape=(4,))
x = Dense(64, activation="relu")(net_input)
x = Dense(32, activation="relu")(x)
output = Dense(2, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)

# Hyperparameters
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500

# Policy function
def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action

# Training loop
for episode in range(NUM_EPISODES):

    done = False
    total_reward = 0
    episode_length = 0

    # ✅ Modified reset
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)

    while not done:
        # Choose action using ε-greedy policy
        action = policy(state, EPSILON)

        # ✅ Modified step
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated

        next_state = tf.convert_to_tensor([next_state], dtype=tf.float32)
        next_action = policy(next_state)  # greedy action for Q-learning

        # TD Target
        target = reward + GAMMA * q_net(next_state)[0][next_action]
        if done:
            target = reward

        # Gradient update
        with tf.GradientTape() as tape:
            current = q_net(state)
        grads = tape.gradient(current, q_net.trainable_weights)
        delta = target - current[0][action]
        for j in range(len(grads)):
            q_net.trainable_weights[j].assign_add(ALPHA * delta * grads[j])

        # Move to next state
        state = next_state
        total_reward += reward
        episode_length += 1

    print(f"Episode: {episode}, Length: {episode_length}, Rewards: {total_reward:.2f}, Epsilon: {EPSILON:.3f}")
    EPSILON /= EPSILON_DECAY

# Save the trained Q-network
q_net.save("q_learning_q_net.keras")  # Recommended
env.close()
