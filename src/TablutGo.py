import TablutGoClient
import os
from ConnectionHandler import ConnectionHandler, ConnectionException, ConnectionClosedException
import TablutEnvironment
import TablutAgent
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import glob


class CommandLineException(Exception):
	pass

if len(sys.argv) != 2:
	raise CommandLineException("Wrong number of console arguments!")
	exit()
player_color = sys.argv[1]
host = sys.argv[2]

if(player_color == "WHITE"):
    os.system('TablutGoClient WHITE localhost')
else:
    os.system('TablutGoClient BLACK localhost')

