# -*- coding: utf-8 -*-

import json
from connectionHandler import ConnectionHandler, ConnectionException, ConnectionClosedException
import sys

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
				execute_move(connHandle, "d" + str(i), "d" + str(i + 1), player)
				i += 1
	except ConnectionException as e:
		print(e)
		break
connHandle.close()
