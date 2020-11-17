# -*- coding: utf-8 -*-

import numpy as np
import random
from collections import deque
import tensorflow as tf #can be used for optimizations (see "DeepQLearning.py")
from tensorflow import keras
from tensorflow.keras import layers

#TODO: handle optimizations to reduce complexity (see "DeepQLearning.py")


class DQNAgent():
    
    def __init__(self, action_size, gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, batch_size, memory_len, update_model_target, split_channels, colour):
        self.action_size = action_size
        
        self.memory = deque(maxlen=memory_len)
        
        self.gamma=gamma
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.learning_rate = learning_rate
        self.games_q_avg = [0]
        
        self.split_channels=split_channels
        
        self.colour=colour
        
        self.update_model_target=update_model_target
        self.replayed_positions=0
        
        self.model = self._build_model()
        self.model_target=self._build_model()
        
    def __getstate__ (self):
        state = self.__dict__.copy()
        del state['model']
        del state['model_target']
        return state
        
    def __setstate__(self, d):
        self.__dict__ = d
        self.model = self._build_model()
        self.model_target=self._build_model()
    
    def _build_model(self):
        
        input_shape=(9,9,2) if self.split_channels else (9,9,1)
        
        inputs = layers.Input(shape=input_shape) #Input=board state (9x9) in one single channel or two (one for white pieces and one for black ones)

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 3, strides=1, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
        
        layer4 = layers.Flatten()(layer3)
    
        action = layers.Dense(self.action_size, activation="linear")(layer4)

        model = keras.Model(inputs=inputs, outputs=action)
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        
        return model
    
    def reshapeInput(self, state):
        
        if self.split_channels:
            channel_white = np.copy(state)
            channel_white = np.where(channel_white==-1, 0, channel_white)
            channel_black = np.copy(state)
            channel_black = np.where(channel_black>0, 0, channel_black)
            
            input_state=np.zeros((9, 9, 2))
            input_state[:,:,0] = np.reshape(channel_white, (9,9))
            input_state[:,:,1] = np.reshape(channel_black, (9,9))
            
            return np.expand_dims(input_state, axis=0)
        
        else:    
            return np.expand_dims(np.reshape(state, (9, 9, 1)), axis = 0)

    def add_q_avg(self, reward):        
        self.games_q_avg[-1] += reward

    def store_q_avg(self, n_moves):
        self.games_q_avg[-1] /= n_moves
        self.games_q_avg.append(0)

    def remember(self, state, action, reward, next_state, done, legal_moves, weight = 1):
        '''
        Save information about an executed move for replay
        '''
        for i in range(weight):
            self.memory.append((state, action, reward, next_state, done, legal_moves))
        
    def act(self, state, legal_moves, perfect=False):
        '''
        Choose a legal move by model prediction with 1-epsilon probability or randomly otherwise.
        '''
        if ~perfect and np.random.rand() <= self.epsilon:
            return random.choice(legal_moves)
        act_values = self.model.predict(self.reshapeInput(state))
        act_values_masked = act_values[0][legal_moves]
        
        best_reward = np.amax (act_values_masked) if self.colour=="W" else np.amin(act_values_masked)
        best_actions = np.where (act_values[0]==best_reward)

        return int(np.random.choice(best_actions[0]))

    def replay(self, batch_size):
        '''
        Train the model with past experience rewards
        '''
        minibatch = random.sample(self.memory,batch_size)
        
        for state, action, reward, next_state, done, legal_moves in minibatch:
            target = reward
            if not done:
                next_state_prediction_values = self.model_target.predict(self.reshapeInput(next_state))
                next_state_prediction_values = next_state_prediction_values[0][legal_moves] #mask for legal moves
                next_state_q_value = np.amax (next_state_prediction_values) if self.colour=="W" else np.amin(next_state_prediction_values)
                target = (reward + self.gamma * next_state_q_value)
            target_f = self.model.predict(self.reshapeInput(state))
            target_f[0][action] = target
            
            self.model.fit(self.reshapeInput(state), target_f, epochs=1, verbose=0) #train model
            #POSSIBLE IMPROVEMENT: replace this function with "manual" operations to reduce complexity (see "DeepQLearning.py")
            
            self.replayed_positions+=1
            if self.replayed_positions % self.update_model_target ==0:
                self.model_target.set_weights(self.model.get_weights()) #sync model_target periodically
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay #reduce exploration and increase exploitation
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min
            
        
            
    def load(self, name):
        '''
        Load weights
        '''
        self.model.load_weights(name)
        self.model_target.load_weights(name)
        
    def save(self, name):
        '''
        Save weights
        '''
        self.model.save_weights(name)
