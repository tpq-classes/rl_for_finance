#
# Financial Q-Learning Agent
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import random
import numpy as np
from pylab import plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

plt.style.use('seaborn-v0_8')

def set_seeds(seed=100):
    ''' Function to set seeds for all
        random number generators.
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TradingBot:
    def __init__(self, hidden_units, learning_rate, learn_env, valid_env=None, val=True):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.val = val
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.98
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.batch_size = 128
        self.max_treward = 0
        self.averages = list()
        self.trewards = []
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
        self.memory = deque(maxlen=2000)
        self.model = self._build_model(hidden_units, learning_rate)

    def _build_model(self, hu, lr):
        ''' Method to create the DNN model.
        '''
        model = Sequential()
        model.add(Dense(hu, input_shape=(
            self.learn_env.lags, self.learn_env.n_features),
            activation='relu'))
        model.add(Dropout(0.3, seed=100))
        model.add(Dense(hu, activation='relu'))
        model.add(Dropout(0.3, seed=100))
        model.add(Dense(2, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=lr)
        )
        return model
        
    def act(self, state):
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()
        q = self.model(tf.convert_to_tensor(state, dtype=tf.float32), training=False)
        return int(tf.argmax(q[0, 0]).numpy())  # keep "first time step" behavior

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
    
        states = np.concatenate([b[0] for b in batch], axis=0).astype(np.float32)        # (B, lags, n_features)
        actions = np.array([b[1] for b in batch], dtype=np.int32)                        # (B,)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)                      # (B,)
        next_states = np.concatenate([b[3] for b in batch], axis=0).astype(np.float32)   # (B, lags, n_features)
        dones = np.array([b[4] for b in batch], dtype=np.bool_)                          # (B,)
    
        # Q(s, :)
        q_states = self.model(states, training=False).numpy()      # (B, lags, 2)
        # Q(s', :)
        q_next = self.model(next_states, training=False).numpy()   # (B, lags, 2)
    
        # keep original logic: use time index 0
        max_q_next = np.max(q_next[:, 0, :], axis=1)               # (B,)
    
        targets = q_states.copy()
        updated = rewards + (1.0 - dones.astype(np.float32)) * (self.gamma * max_q_next)
        targets[np.arange(self.batch_size), 0, actions] = updated
    
        self.model.train_on_batch(states, targets)
    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def validate(self, e, episodes):
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags, self.valid_env.n_features])
    
        for _ in range(10000):
            q = self.model(tf.convert_to_tensor(state, dtype=tf.float32), training=False)
            action = int(tf.argmax(q[0, 0]).numpy())
    
            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(next_state, [1, self.valid_env.lags, self.valid_env.n_features])
    
            if done:
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)
                if e % 15 == 0:
                    templ = 70 * '='
                    templ += '\nepisode: {:2d}/{} | VALIDATION | '
                    templ += 'treward: {:4d} | perf: {:5.3f} | eps: {:.2f}\n'
                    templ += 70 * '='
                    print(templ.format(e, episodes, treward, perf, self.epsilon))
                break

    def learn(self, episodes):
        ''' Method to train the DQL agent.
        '''
        for e in range(1, episodes + 1):
            state = self.learn_env.reset()
            state = np.reshape(state, [1, self.learn_env.lags,
                                       self.learn_env.n_features])
            for _ in range(10000):
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)
                next_state = np.reshape(next_state,
                                        [1, self.learn_env.lags,
                                         self.learn_env.n_features])
                self.memory.append([state, action, reward,
                                    next_state, done])
                state = next_state
                if done:
                    treward = _ + 1
                    self.trewards.append(treward)
                    av = sum(self.trewards[-25:]) / 25
                    perf = self.learn_env.performance
                    self.averages.append(av)
                    self.performances.append(perf)
                    self.aperformances.append(
                        sum(self.performances[-25:]) / 25)
                    self.max_treward = max(self.max_treward, treward)
                    templ = 'episode: {:2d}/{} | treward: {:4d} | '
                    templ += 'perf: {:5.3f} | av: {:5.1f} | max: {:4d}'
                    print(templ.format(e, episodes, treward, perf,
                                       av, self.max_treward), end='\r')
                    break
            if self.val:
                self.validate(e, episodes)
            if len(self.memory) > self.batch_size:
                self.replay()
        print()

def plot_treward(agent):
    ''' Function to plot the total reward
        per training eposiode.
    '''
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.averages) + 1)
    y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
    plt.plot(x, agent.averages, label='moving average')
    plt.plot(x, y, 'r--', label='regression')
    plt.xlabel('episodes')
    plt.ylabel('total reward')
    plt.legend()


def plot_performance(agent):
    ''' Function to plot the financial gross
        performance per training episode.
    '''
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.performances) + 1)
    y = np.polyval(np.polyfit(x, agent.performances, deg=3), x)
    plt.plot(x, agent.performances[:], label='training')
    plt.plot(x, y, 'r--', label='regression (train)')
    if agent.val:
        y_ = np.polyval(np.polyfit(x, agent.vperformances, deg=3), x)
        plt.plot(x, agent.vperformances[:], label='validation')
        plt.plot(x, y_, 'r-.', label='regression (valid)')
    plt.xlabel('episodes')
    plt.ylabel('gross performance')
    plt.legend()
