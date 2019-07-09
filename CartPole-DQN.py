import gym

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

EPISODES = 50000

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNAgent:
    def __init__(self, observation_space, action_space):
        self.action_space = action_space
        self.observation_space = observation_space

        self.memory = deque(maxlen=2000)
        self.exploration_rate = EXPLORATION_MAX

        self.createModel()

    def createModel(self):
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.observation_space, ), activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_space, activation='linear'))
        self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE), metrics=['mse'])

    def remember(self, state, action, reward, next_state, done):
        # list of previous experiences and observations
        # to re-train the model with the previous experiences
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # trains the neural net with experiences in the memory
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            # if done, make our target reward
            q_update = reward
            if not done:
                q_update = reward + GAMMA * np.amax(self.model.predict(next_state)[0])
            # make the agent to approximately map the current state
            # to future discounted reward. We'll call that q_values
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            # Train the Neural Net with the state and q_values
            self.model.fit(state, q_values, epochs=1, verbose=0)
        # decrease the rate of random actions
        if self.exploration_rate > EXPLORATION_MIN:
            self.exploration_rate *= EXPLORATION_DECAY

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def save(self, fname):
        self.model.save(fname)


def play(env_name):
    env = gym.make(env_name)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    agent = DQNAgent(observation_space, action_space)

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, 4])

        scores = []
        for t in range(500):
            action = agent.act(state)

            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])

            reward = reward if not done else -reward
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            if done:
                scores.append(t)
                break
            agent.replay()

        print('Episode: {}/{}, score: {}, e: {:.2}'.format(e, EPISODES, t, agent.exploration_rate))
        if e % 100 == 0:
            agent.save('model/model-dqn.h5')
            plt.plot(scores)
            plt.savefig('score_plot')

if __name__ == '__main__':
    play(env_name='CartPole-v1')
