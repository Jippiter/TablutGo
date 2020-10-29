# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import socket
import json
import numpy as np

HOST = "localhost"
PORT = 5800


sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
received = sock.recv(1024)
print(received)