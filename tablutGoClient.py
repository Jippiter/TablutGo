# -*- coding: utf-8 -*-

import json
from connectionHandler import ConnectionHandler, ConnectionException, ConnectionClosedException
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Configuration paramaters for the whole setup
gamma = 0.99  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  # Rate at which to reduce chance of random action being taken
batch_size = 32  # Size of batch taken from replay buffer
max_steps_per_episode = 10000

#TODO: here some info on our environment/state?

num_actions = 9*9*(8*8)



def create_q_model():
    # Network defined by the Deepmind paper
    inputs = layers.Input(shape=(9, 9, 4,))

    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(inputs)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)

    layer4 = layers.Flatten()(layer3)

    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    return keras.Model(inputs=inputs, outputs=action)

def CommandLineException(Exception):
	pass
# Just to test a move
def execute_move(connHandle, fro, to, player):
	move = {
	  "from":fro,
	  "to":to,
	  "turn":player
	}
	connHandle.send(json.dumps(move))

# Parse command-line arguments
if len(sys.argv) != 2:
	raise CommandLineException("Wrong number of console arguments!")
	exit()
player = sys.argv[1]
host = "localhost"
port = 0
if player == "WHITE":
	port = 5800
elif player == "BLACK":
	port = 5801
else:
	raise CommandLineException("Invalid argument for player!")
	exit()

# Start connection
connHandle = ConnectionHandler(host, port)
connHandle.send('TablutGo')

i = 1
while True:
	try:
		length, message = connHandle.recv()
		print("Received message of length " + str(length) + ", Printing raw data: ")
		print(message)
		if message:
			data = json.loads(message)
			print("Found turn: " + data['turn'])
			if data['turn'] == "BLACK":

				#TODO: Implement function
				state = readStateFromMessage()
				episode_reward = 0

				# this is going to become for every move we can make
				for timestep in range(1, max_steps_per_episode):
					frame_count += 1

					# Use epsilon-greedy for exploration
					if frame_count < epsilon_random_frames or epsilon > np.random.rand(1)[0]:
						# Take random action
						action = np.random.choice(num_actions)
					else:
						# Predict action Q-values
						# From environment state
						state_tensor = tf.convert_to_tensor(state)
						state_tensor = tf.expand_dims(state_tensor, 0)
						action_probs = model(state_tensor, training=False)
						# Take best action
						#TODO: find converstion back to move?
						action = tf.argmax(action_probs[0]).numpy()

					# Decay probability of taking random action
					epsilon -= epsilon_interval / epsilon_greedy_frames
					epsilon = max(epsilon, epsilon_min)


					# Apply the sampled action in our environment

					#TODO: send move back using connectionhandler
					#also need to get a reward from our move somhow
					state_next, reward, done, _ = sendMove(action)
					execute_move(connHandle, "d" + str(i), "d" + str(i + 1), player)
					i += 1

					#TODO: important that we somehow get the next state here, maybe another read?
					state_next = np.array(state_next)

					episode_reward += reward

					# Save actions and states in replay buffer
					action_history.append(action)
					state_history.append(state)
					state_next_history.append(state_next)
					done_history.append(done)
					rewards_history.append(reward)
					state = state_next

					# Update every fourth frame and once batch size is over 32
					#TODO: maybe rewrite this? every 4 steps backpropagation is also not too bad
					if frame_count % update_after_actions == 0 and len(done_history) > batch_size:
						# Get indices of samples for replay buffers
						indices = np.random.choice(range(len(done_history)), size=batch_size)

						# Using list comprehension to sample from replay buffer
						state_sample = np.array([state_history[i] for i in indices])
						state_next_sample = np.array([state_next_history[i] for i in indices])
						rewards_sample = [rewards_history[i] for i in indices]
						action_sample = [action_history[i] for i in indices]
						done_sample = tf.convert_to_tensor(
							[float(done_history[i]) for i in indices]
						)

						# Build the updated Q-values for the sampled future states
						# Use the target model for stability
						future_rewards = model_target.predict(state_next_sample)
						# Q value = reward + discount factor * expected future reward
						updated_q_values = rewards_sample + gamma * tf.reduce_max(
							future_rewards, axis=1
						)

						#TODO: this already handles a game end, maybe change this to see who won (1/-1)

						# If final frame set the last value to -1
						updated_q_values = updated_q_values * (1 - done_sample) - done_sample

						# Create a mask so we only calculate loss on the updated Q-values
						masks = tf.one_hot(action_sample, num_actions)

						with tf.GradientTape() as tape:
							# Train the model on the states and updated Q-values
							q_values = model(state_sample)

							# Apply the masks to the Q-values to get the Q-value for action taken
							q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
							# Calculate loss between new Q-value and old Q-value
							loss = loss_function(updated_q_values, q_action)

						# Backpropagation
						grads = tape.gradient(loss, model.trainable_variables)
						optimizer.apply_gradients(zip(grads, model.trainable_variables))

					if frame_count % update_target_network == 0:
						# update the the target network with new weights
						model_target.set_weights(model.get_weights())
						# Log details
						template = "running reward: {:.2f} at episode {}, frame count {}"
						print(template.format(running_reward, episode_count, frame_count))

					# Limit the state and reward history
					if len(rewards_history) > max_memory_length:
						del rewards_history[:1]
						del state_history[:1]
						del state_next_history[:1]
						del action_history[:1]
						del done_history[:1]

					if done:
						break

				# Update running reward to check condition for solving
				episode_reward_history.append(episode_reward)
				if len(episode_reward_history) > 100:
					del episode_reward_history[:1]
				running_reward = np.mean(episode_reward_history)

				episode_count += 1

				if running_reward > 40:  # Condition to consider the task solved
					print("Solved at episode {}!".format(episode_count))
					break



	except ConnectionException as e:
		print(e)
		break



connHandle.close()
