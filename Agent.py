from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
import random


class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.learn_rate = 0.001
        self.gamma = 0.95
        # Exploration rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = self._build_model()
        #self.checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.h5',
        #                                    verbose=1, save_best_only=False)
        self.memory = deque(maxlen=2000)

    def _build_model(self):
        model = Sequential()
        # Input layer of state size(4) and Hidden Layer with 24 nodes
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learn_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            target_f = self.model.predict(state)
            target_f[0][action] = target

            # Train the NN with the state and target_f
            self.model.fit(state, target_f, epochs=1, verbose=0)

            # Need to do 'pip install h5py' to save model
            if os.path.exists('saved_models/weights.best.from_scratch.h5'):
                os.remove('saved_models/weights.best.from_scratch.h5')
            self.model.save_weights('saved_models/weights.best.from_scratch.h5')

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            print("Without learning")
            self.epsilon = 0
            # self.model.load_weights('saved_models/weights.best.from_scratch.hdf5')
