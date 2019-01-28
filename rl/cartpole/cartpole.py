"""
A pole is attached by an un-actuated joint to a cart, which moves
along a frictionless track. The system is controlled by applying a
force of +1 or -1 to the cart. The pendulum starts upright, and the
goal is to prevent it from falling over. A reward of +1 is provided
for every timestep that the pole remains upright. The episode ends
when the pole is more than 15 degrees from vertical, or the cart
moves more than 2.4 units from the center.

CartPole-v0 defines "solving" as getting average reward of 195.0
over 100 consecutive trials. This environment corresponds to the version
of the cart-pole problem described by Barto, Sutton, and Anderson [Barto83].
"""

# Inspired from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

# Import dependencies
import numpy as np
import gym
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# from https://github.com/gsurma/cartpole
from scores.score_logger import ScoreLogger
import os

ENV_NAME = "CartPole-v1"

# Hyperparamters
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01  # 1% of the time the agent will explore
EXPLORATION_DECAY = 0.995

N_EPISODES = 1001

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

# Set Parameters
env = gym.make(ENV_NAME)
score_logger = ScoreLogger(ENV_NAME)

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

output_dir = 'model_output/cartpole'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class DQNNN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(state_size, 24)
        self.dense2 = nn.Linear(24, 24)
        self.output = nn.Linear(23, action_size)

    def forward(self, x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return self.output(x)
    

# Define agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_MAX  # Init exploration rate
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(
            lr=LEARNING_RATE))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def experience_replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                target = reward + GAMMA * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.exploration_rate > EXPLORATION_MIN:
            self.exploration_rate *= EXPLORATION_DECAY

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


# Interact with our environment
agent = DQNAgent(state_size, action_size)
done = False

for episode_nb in range(N_EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    for time_step in range(5000):
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])

        agent.remember(state, action, reward, next_state, done)

        state = next_state

        if done:
            print('episode: {}/{}, score:{}, exploration rate: {:.2}'
                  .format(episode_nb, N_EPISODES,
                          time_step, agent.exploration_rate))
            score_logger.add_score(time_step, episode_nb)
            break

        if len(agent.memory) > BATCH_SIZE:
            agent.experience_replay(BATCH_SIZE)

        if episode_nb % 50 == 0:
            agent.save(output_dir + "weight_" +
                       '{: 04d}'.format(episode_nb) + ".hdf5")
