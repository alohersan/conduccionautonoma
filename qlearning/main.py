"""
In this project, we will use the existing environments provided by the
OpenAI Gymnasium library (https://github.com/Farama-Foundation/Gymnasium)
to create a reinforcement learning agent that learns to play the Taxi game with Q-learning.

- Gymnasium: provides the environment, animations, rewards, states, and other utilities.
- Q-learning: implemented from scratch by me.

"""
#Libraries
import random
import gym
import numpy as np

# Initialize the environment
environment = gym.make('Taxi-v3')

# Q-learning hyperparameters
# Learning rate
alpha = 0.9
# Discount factor
gamma = 0.95
# Exploration rate
epsilon = 1.0
epsilon_decay = 0.9995
epsilon_min = 0.01

# Total training episodes
num_episodes = 10000
# Max steps per episode
max_steps = 100

# Initialize Q-table: state space x action space
q_table = np.zeros([environment.observation_space.n, environment.action_space.n])


# Function to choose an action based on the current state
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return environment.action_space.sample()
    else:
        return np.argmax(q_table[state])


# Training loop
for episode in range(num_episodes):
    state, _ = environment.reset()
    done = False

    for step in range(max_steps):
        action = choose_action(state)
        next_state, reward, done, truncated, info = environment.step(action)

        # Q-learning update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

        state = next_state

        if done or truncated:
            break

    # Decay epsilon after each episode
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

# Reinitialize the environment with render mode to visualize
environment = gym.make('Taxi-v3', render_mode='human')

# Test the trained agent
for episode in range(10):
    state, _ = environment.reset()
    done = False
    print(f"Episode nº {episode}")

    for step in range(max_steps):
        environment.render()
        action = np.argmax(q_table[state])
        next_state, reward, done, truncated, info = environment.step(action)
        state = next_state

        if done or truncated:
            environment.render()
            print(f"Finished episode nº {episode+1} in {step + 1} steps with reward {reward}")
            break

environment.close()
