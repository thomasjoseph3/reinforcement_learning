import gymnasium as gym # type: ignore
import pickle as pkl
import numpy as np
import cv2 # type: ignore

cliff_env = gym.make("CliffWalking-v0", render_mode="ansi")
q_table = pkl.load(open("qlearning_qtable.pkl", "rb"))

def initialize_frame():
    width, height = 600, 200
    img = np.ones(shape=(height, width, 3)) * 255.0
    margin_horizontal = 6
    margin_vertical = 2
    for i in range(13):
        img = cv2.line(img, (49 * i + margin_horizontal, margin_vertical),
                       (49 * i + margin_horizontal, 200 - margin_vertical), color=(0, 0, 0), thickness=1)
    for i in range(5):
        img = cv2.line(img, (margin_horizontal, 49 * i + margin_vertical),
                       (600 - margin_horizontal, 49 * i + margin_vertical), color=(0, 0, 0), thickness=1)
    img = cv2.rectangle(img, (49 * 1 + margin_horizontal + 2, 49 * 3 + margin_vertical + 2),
                        (49 * 11 + margin_horizontal - 2, 49 * 4 + margin_vertical - 2), color=(255, 0, 255),
                        thickness=-1)
    img = cv2.putText(img, text="Cliff", org=(49 * 5 + margin_horizontal, 49 * 4 + margin_vertical - 10),
                      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
    frame = cv2.putText(img, text="G", org=(49 * 11 + margin_horizontal + 10, 49 * 4 + margin_vertical - 10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return frame

def put_agent(img, state):
    margin_horizontal = 6
    margin_vertical = 2
    row, column = np.unravel_index(indices=state, shape=(4, 12))
    cv2.putText(img, text="A", org=(49 * column + margin_horizontal + 10, 49 * (row + 1) + margin_vertical - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
    return img

def policy(state, explore=0.0):
    action = int(np.argmax(q_table[state]))
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=4, size=1))
    return action

NUM_EPISODES = 5
for episode in range(NUM_EPISODES):
    done = False
    total_reward = 0
    episode_length = 0
    frame = initialize_frame()
    state, info = cliff_env.reset()
    while not done:
        frame2 = put_agent(frame.copy(), state)
        cv2.imshow("Cliff Walking Q-learning", frame2)
        cv2.waitKey(250)
        action = policy(state)
        state, reward, terminated, truncated, info = cliff_env.step(action)
        episode_length += 1
        total_reward += reward
        done = terminated or truncated
    print("Episode:", episode, "Length:", episode_length, "Reward:", total_reward)
cliff_env.close()
