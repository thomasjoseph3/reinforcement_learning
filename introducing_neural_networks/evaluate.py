import gymnasium as gym
import cv2
import tensorflow as tf
from keras.models import load_model

# Load the environment and the trained model
env = gym.make("CartPole-v1", render_mode="rgb_array")
model = load_model("q_learning_q_net.keras")  # or .h5 if saved as .h5

# ε-greedy policy (explore=0 disables randomness for evaluation)
def policy(state, explore=0.0):
    action = tf.argmax(model(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    return action

# Run a few episodes to see the agent in action
for episode in range(5):
    done = False
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)

    while not done:
        # Render the frame and show using OpenCV
        frame = env.render()
        cv2.imshow("CartPole Agent", frame)
        cv2.waitKey(100)  # ~10 FPS

        # Select action and take step
        action = policy(state, explore=0.0)
        next_state, reward, terminated, truncated, _ = env.step(action.numpy())
        done = terminated or truncated

        # Update state
        state = tf.convert_to_tensor([next_state], dtype=tf.float32)

env.close()
cv2.destroyAllWindows()
