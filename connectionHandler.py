# -*- coding: utf-8 -*-

import socket

class ConnectionException(Exception):
	pass
class ConnectionClosedException(ConnectionException):
	pass

class ConnectionHandler:
	def __init__(self, host, port):
		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		try:
			self.socket.connect((host, port))
		except OSError as e:
			raise ConnectionException(e)

	def close(self):
		self.socket.close()

	def send(self, message):
		msg = message + '\r\n'
		length = len(msg)
		try:
			self.socket.sendall(length.to_bytes(4, 'big') + bytes(msg, 'utf-8'))
		except OSError as e:
			print(e)
			exit()

	def recv(self):
		length = b''
		while len(length) < 4:
			try:
				data = self.socket.recv(4 - len(length))			
			except OSError as e:
				raise ConnectionException(e)
			if data:
				length += data
			else:
				raise ConnectionClosedException("Connection aborted!");
		length = int.from_bytes(length, 'big')
		message = b''
		while len(message) < length:
			try:
				data = self.socket.recv(length - len(message))
			except OSError as e:
				raise ConnectionException(e)
			if data:
				message += data
			else:
				raise ConnectionClosedException("Connection aborted!");	
		message = message.decode('utf-8')
		return length, message
