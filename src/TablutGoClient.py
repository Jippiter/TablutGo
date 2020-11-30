# -*- coding: utf-8 -*-

import json
from ConnectionHandler import ConnectionHandler, ConnectionException, ConnectionClosedException
import TablutEnvironment
import TablutAgent
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob

agent_path = "Valerio Final/"

class CommandLineException(Exception):
	pass

def send_move(connHandle, fro, to, player_color):
	move = {
	  "from": fro,
	  "to": to,
	  "turn": player_color
	}
	connHandle.send(json.dumps(move))


def JSON_to_local_state(data):
	board = data['board']
	new_state = np.ndarray((9, 9), dtype=int)
	for i in range(len(board[0])):
		for j in range(len(board)):
			if board[i][j] == "WHITE":
				new_state[i, j] = 1
			elif board[i][j] == "BLACK":
				new_state[i, j] = -1
			elif board[i][j] == "KING":
				new_state[i, j] = 3
            else:
                new_state[i, j] = 0
	return new_state, data['turn']

# Parse command-line arguments
if len(sys.argv) != 3:
	raise CommandLineException("Wrong number of console arguments!")
	exit()
player_color = sys.argv[1]
host = sys.argv[2]
# Init local environment
env = TablutEnvironment.Environment(reward_king_captured=0,
                                reward_king_escape=0,
                                reward_white_capture=0,
                                reward_black_capture=0,
                                reward_king_closer_edge=0,
                                reward_king_further_black=0, 
                                reward_king_freedom=0,
								reward_neutral_move=0,
                                board_path="../Resources/board.png",
                                draw_board=False)

# Init agent and set port based on command-line color
port = 0
agent = None
if player_color.lower() == "white":
	port = 5800
	agent = TablutAgent.DQNAgent(action_size=9*9*16,
                                     gamma=0,
                                     epsilon=0.1,
                                     epsilon_min=0.1,
                                     epsilon_decay=0,
                                     learning_rate=0,
                                     batch_size=0,
                                     memory_len=1,
                                     update_model_target=0,
                                     split_channels=True,
                                     colour="W")
	agent_weights = glob.glob("../Weights/" + agent_path + "*white*.hdf5")
	# Load weights
	if len(agent_weights) > 0:
		agent.load(agent_weights[-1])
	else:
		raise Exception("Weights are None!")
		exit()
elif player_color.lower() == "black":
	port = 5801
	agent = TablutAgent.DQNAgent(action_size=9*9*16,
                                 gamma=0,
                                 epsilon=0.1,
                                 epsilon_min=0.1,
                                 epsilon_decay=0,
                                 learning_rate=0,
                                 batch_size=0,
                                 memory_len=1,
                                 update_model_target=0,
                                 split_channels=True,
                                 colour="B")
	agent_weights = glob.glob("../Weights/" + agent_path + "*black*.hdf5")
	# Load weights
	if len(agent_weights) > 0:
		agent.load(agent_weights[-1])
	else:
		raise Exception("Weights are None!")
		exit()
else:
	raise CommandLineException("Invalid argument for player!")
	exit()

# Start connection
connHandle = ConnectionHandler(host, port)
connHandle.send('TablutGo')

state, legal_moves = env.reset()
end_turn = "WHITE"
# Game loop
while True:
	try:
		length, message = connHandle.recv()
		print("Received message of length {}".format(length))
		if message:
			# Sync local state with server state
			data = json.loads(message)
			new_state, new_turn = JSON_to_local_state(data)
			new_state, _, done, draw, legal_moves = env.set_state(new_state, new_turn)
			state = new_state
			end_turn = new_turn
			print("Turn is {}".format(data['turn']))
			if data['turn'] == player_color:
				print("Computing and sending action.")
				action=agent.act(state, legal_moves, True)
				fro, to=env.actionToCoordinates(action, server=True)
				send_move(connHandle, fro, to, player_color)
				print("Action sent!")
			else:
				print("Waiting...")
	except ConnectionException as e:
		print(e)
		if end_turn == "DRAW":
			result= "drew."
		elif end_turn == player_color + "WIN":
			result= "won!"
		else:
			result = "lost. :("
		print("We {} GG WP!".format(result))
		break

connHandle.close()
