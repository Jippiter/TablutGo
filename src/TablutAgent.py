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
        
        input_shape=(9,9,3) if self.split_channels else (9,9,1)
        
        inputs = layers.Input(shape=input_shape) #Input=board state (9x9) in one single channel or three (one for white pieces, one for the king and one for black pieces)

        # Convolutions on the frames on the screen
        layer1 = layers.Conv2D(32, 3, strides=1, activation="relu")(inputs)
        layer2 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer1)
        layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
        
        layer4 = layers.Flatten()(layer3)
        layer5 = layers.Dense(512, activation="relu")(layer4)
    
        action = layers.Dense(self.action_size, activation="linear")(layer5)

        model = keras.Model(inputs=inputs, outputs=action)
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0))
        
        return model
    
    def reshapeInput(self, state):
        
        if self.split_channels:
            channel_white = np.copy(state)
            channel_white = np.where(channel_white!=1, 0, channel_white)
            channel_black = np.copy(state)
            channel_black = np.where(channel_black!=-1, 0, channel_black)
            channel_black = np.where(channel_black==-1, 1, channel_black)
            channel_king = np.copy(state)
            channel_king = np.where(channel_king!=3, 0, channel_king)
            channel_king = np.where(channel_king==3, 1, channel_king)
            
            input_state=np.zeros((9, 9, 3))
            input_state[:,:,0] = np.reshape(channel_white, (9,9))
            input_state[:,:,1] = np.reshape(channel_black, (9,9))
            input_state[:,:,2] = np.reshape(channel_king, (9,9))


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
            
    def replayOptimized(self, batch_size):
        loss_function = keras.losses.Huber()
        optimizer=keras.optimizers.Adam(lr=self.learning_rate, clipnorm=1.0)
        
        # Get indices of samples for replay buffers
        minibatch = random.sample(self.memory,batch_size)

        # Using list comprehension to sample from replay buffer
        state_sample = []
        action_sample = []
        rewards_sample = []
        state_next_sample = []
        done_sample = []
        legal_moves_sample = []
        
        for sample in minibatch:
            state_sample.append(self.reshapeInput(sample[0]))
            state_next_sample.append(self.reshapeInput(sample[3]))
            action_sample.append(sample[1])
            rewards_sample.append(sample[2])
            done_sample.append(sample[4])
            legal_moves_sample.append(sample[5])

        state_sample = np.reshape(np.array([state_sample]),(batch_size,9,9,3))
        state_next_sample = np.reshape(np.array([state_next_sample]),(batch_size,9,9,3))
        # Build the updated Q-values for the sampled future states
        # Use the target model for stability
        future_rewards = self.model_target.predict(state_next_sample)
        predicted_q_values = np.zeros((batch_size))
        for i in range(batch_size):
            masked_rewards = future_rewards[i][legal_moves_sample[i]]
            predicted_q_values[i] = tf.reduce_max(masked_rewards, axis=0) if self.colour=="W" else tf.reduce_min(masked_rewards, axis=0)
        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_sample + done_sample * (self.gamma * predicted_q_values)


        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(action_sample, self.action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(state_sample)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = loss_function(updated_q_values, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
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
