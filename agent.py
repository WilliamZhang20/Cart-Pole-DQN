import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

from matplotlib import pyplot as plt

class Agent: # Based on a DQN to map to the right moves
    def __init__(self, state_size, action_size):
        self.n_actions = action_size
        '''
        lr = learning rate
        gamma = discount factor
        decay = rate of decay of the exploration probability
        batch_size = size of samples in 1 round
        '''
        self.lr = 0.001
        self.gamma = 0.99
        self.exp_prob = 1.0
        self.decay = 0.005
        self.batch_size = 32

        # memory buffer, storing only 2000 at one time
        self.memory_buffer = list()
        self.max_memory_buffer = 2000

        # Sequentially grouping 2 hidden layers of 24 neurons each
        self.model = Sequential([
            Input(shape=(state_size,)),
            Dense(units=24, activation='relu'),
            Dense(units=24, activation='relu'),
            Dense(units=action_size, activation='linear'),
        ])
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr))

    # Compute the action / output of the deep NN
    def compute_action(self, current_state):
        # sample a random variable (only during training)
        # If less than the set exploration boundary probability, a random action is chosen
        # Otherwise, forward the input through DNN and choose action with highest Q value

        current_state = np.array(current_state)
        current_state = np.reshape(current_state, [1, -1])

        if np.random.uniform(0, 1) < self.exp_prob:
            return np.random.choice(range(self.n_actions))
        q_values = self.model.predict(current_state)[0]
        return np.argmax(q_values)
    
    # After each training episode, decrease exploration probability
    def update_exp_prob(self):
        self.exp_prob = self.exp_prob * np.exp(-self.decay)
        print(self.exp_prob)

    # Store experiences (TODO: maybe make them from random times too?)
    def store_episode(self, current_state, action, reward, next_state, done):
        # In a giant list
        self.memory_buffer.append({
            "current_state": current_state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done
        })

        if len(self.memory_buffer) > self.max_memory_buffer:
            self.memory_buffer.pop(0)
    
    # Train the DQN
    def train(self):
        if len(self.memory_buffer) < self.batch_size:
            return

        # use a random batch of recent experiences
        np.random.shuffle(self.memory_buffer)
        batch_sample = self.memory_buffer[0:self.batch_size]

        for experience in batch_sample:
            current_state = np.reshape(experience["current_state"], [1, -1])
            next_state = np.reshape(experience["next_state"], [1, -1])
            
            q_current_state = self.model.predict(current_state, verbose=0)
            q_target = experience["reward"]

            # Applying Bellman Optimality to get standard Q-Values
            if not experience["done"]:
                q_target = q_target + self.gamma * np.max(self.model.predict(next_state, verbose=0)[0])

            q_current_state[0][experience["action"]] = q_target

            self.model.fit(current_state, q_current_state, verbose=0, epochs=1)