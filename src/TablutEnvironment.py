# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image as Img, ImageDraw, ImageTk
from tkinter import *
import time

class Environment:
    
    def __init__(self, reward_king_captured, reward_king_escape, reward_white_capture, reward_black_capture, board_path, draw_board):
        self.current_state=None
        self.turn=None
        self.reached_states = None
        self.white_reward = None
        self.legal_moves=None
        
        self.reward_king_captured=reward_king_captured
        self.reward_king_escape=reward_king_escape
        self.reward_white_capture=reward_white_capture
        self.reward_black_capture=reward_black_capture
        self.board_path=board_path
        self.draw_board=draw_board
        
        self.WHITE=1
        self.BLACK=-1
        
        self.columns_dictionary = {
                    "a": 0,
                    "b": 1,
                    "c": 2,
                    "d": 3,
                    "e": 4,
                    "f": 5,
                    "g": 6,
                    "h": 7,
                    "i": 8
                }
        
        self.reversed_columns_dictionary = dict((reversed(item) for item in self.columns_dictionary.items()))
        
        self.rows_dictionary = {
                    "1": 8,
                    "2": 7,
                    "3": 6,
                    "4": 5,
                    "5": 4,
                    "6": 3,
                    "7": 2,
                    "8": 1,
                    "9": 0
                }
        
        self.reversed_rows_dictionary = dict((reversed(item) for item in self.rows_dictionary.items()))
        
        self.board = np.array([[0,0,0,1,1,1,0,0,0], #-1 = Throne
                              [0,0,0,0,1,0,0,0,0], #Positive numbers = camps
                              [0,0,0,0,0,0,0,0,0],
                              [2,0,0,0,0,0,0,0,3],
                              [2,2,0,0,-1,0,0,3,3],
                              [2,0,0,0,0,0,0,0,3],
                              [0,0,0,0,0,0,0,0,0],
                              [0,0,0,0,4,0,0,0,0],
                              [0,0,0,4,4,4,0,0,0]])
    
        if self.draw_board:
            self.window=Tk()
            self.window.title("Board")
            self.canvas = Canvas(self.window,width=450,height=450)
            self.canvas.pack()
    

    
    def actionToCoordinates(self,action, server=False):
        '''
        Decode action number into coordinates (from, to)
        '''
        from_square = action // 16
        from_row = from_square // 9
        from_column = from_square % 9
        
        to_square = action % 16
        direction = to_square // 8
        if direction is 0: #move on same column
            to_column = from_column
            to_row = (from_row + 1 + to_square % 8) % 9
        else: #move on same row
            to_row = from_row
            to_column = (from_column + 1 + to_square % 8) % 9
            
        if server:
            from_row = 8-from_row
            to_row = 8-to_row
            
        from_coordinates=self.reversed_columns_dictionary[from_column] + self.reversed_rows_dictionary[from_row]
        to_coordinates=self.reversed_columns_dictionary[to_column] + self.reversed_rows_dictionary[to_row]
        
        return (from_coordinates, to_coordinates)
    
    def squaresToAction(self, from_square, to_square):
        from_row, from_column = from_square
        to_row, to_column = to_square
        
        direction = 1 if from_row==to_row else 0
        if direction==0:
            movement = from_row - to_row
        else:
            movement = from_column - to_column
            
        if movement>0:
            movement = 9 - movement
        if movement<0:
            movement*=-1
        
        action = ((from_row * 9 + from_column) * 16) + (8 * direction + movement -1)
        
        return action
    
    def coordinatesToAction(self, coordinates):
        '''
        Encode coordinates from (from, to) form into action number [0,9x9x16]
        '''
        from_coordinates, to_coordinates = coordinates
        
        from_column = self.columns_dictionary[from_coordinates[0]]
        from_row = self.rows_dictionary[from_coordinates[1]]
        
        to_column = self.columns_dictionary[to_coordinates[0]]
        to_row = self.rows_dictionary[to_coordinates[1]]
        
        from_square = (from_row, from_column)
        to_square = (to_row, to_column)
        
        return self.squaresToAction(from_square, to_square)
    
    def showState(self, state):
        '''
        Draw the current state on a board with pieces
        '''
        board = Img.open(self.board_path)
        draw = ImageDraw.Draw(board)
        size=board.width
        square_size=size/9
        centre=square_size/2
        radius=square_size/2 - 5
        
        for i in range (9):
            for j in range (9):
                piece=state[i][j]
                if piece==-1:
                    draw.ellipse((square_size*j + centre - radius, square_size*i + centre - radius, square_size*j + centre + radius, square_size*i + centre + radius),fill=(0, 0, 0), outline=(255, 255, 255))
                elif piece==1:
                    draw.ellipse((square_size*j + centre - radius, square_size*i + centre - radius, square_size*j + centre + radius, square_size*i + centre + radius),fill=(255, 255, 255), outline=(0, 0, 0))
                elif piece==3:
                    draw.ellipse((square_size*j + centre - radius, square_size*i + centre - radius, square_size*j + centre + radius, square_size*i + centre + radius),fill=(255, 255, 255), outline=(0, 0, 0))
                    draw.rectangle((square_size*j + centre - radius/4, square_size*i + centre - radius/2, square_size*j + centre + radius/4, square_size*i + centre + radius/2),fill=(0, 0, 0), outline=(0, 0, 0))
                    draw.rectangle((square_size*j + centre - radius/2, square_size*i + centre - radius/4, square_size*j + centre + radius/2, square_size*i + centre + radius/4),fill=(0, 0, 0), outline=(0, 0, 0))
        
        image = ImageTk.PhotoImage(board)
        self.canvas.delete("all")
        imagesprite = self.canvas.create_image(size/2,size/2,image=image)
        self.window.update()
        
    def reset(self):
        '''
        Reset environment to original state and variable configuration
        '''
        self.turn=self.WHITE
        self.reached_states=[]
        self.white_reward=0
        
        self.current_state=np.array([[0,0,0,-1,-1,-1,0,0,0],
                                    [0,0,0,0,-1,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [-1,0,0,0,1,0,0,0,-1],
                                    [-1,-1,1,1,3,1,1,-1,-1],
                                    [-1,0,0,0,1,0,0,0,-1],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,0,-1,0,0,0,0],
                                    [0,0,0,-1,-1,-1,0,0,0]])
        
        self.legal_moves=self.getAllLegalMoves(self.current_state,self.turn)
        
        if self.draw_board:
            self.showState(self.current_state)
    
        return self.current_state, self.legal_moves
    
    def set_state(self, state, turn):
        done=False
        draw=False
          
        if turn == "BLACK":
            self.turn=self.BLACK
        elif turn == "WHITE":
            self.turn=self.WHITE  
        self.current_state=state

        self.reached_states.append((self.current_state, self.turn))
        self.legal_moves=self.getAllLegalMoves(self.current_state, self.turn)     

        if self.isKingCaptured(state):
            done=True        
        elif self.isKingOnSafeSquare(state):
            done=True
        elif self.checkNoLegalMovesRemained(state):
            done=True
        elif self.stateReached(state, -self.turn):
            done=True
            draw=True

        return state, 0, done, draw, self.legal_moves

    def step(self, action):
        '''
        Execute a move.
        This function returns next-state, white_reward, done (game ended), draw-result and next possible legal_moves
        '''
        if not self.isLegalAction(self.current_state, action):
            raise Exception("Illegal move execption")
        
        from_coordinates, to_coordinates = self.actionToCoordinates(action)
        
        from_column = self.columns_dictionary[from_coordinates[0]]
        from_row = self.rows_dictionary[from_coordinates[1]]
        
        to_column = self.columns_dictionary[to_coordinates[0]]
        to_row = self.rows_dictionary[to_coordinates[1]]
        
        next_state = np.copy(self.current_state)
        next_state[to_row,to_column]=next_state[from_row,from_column]
        next_state[from_row,from_column]=0
        
        next_state, number_of_captures=self.applyCaptures(next_state, (to_row, to_column))
        
        if self.turn==self.WHITE:  
            self.legal_moves=self.getAllLegalMoves(next_state, self.BLACK)
        else:
            self.legal_moves=self.getAllLegalMoves(next_state, self.WHITE)
        
        done=False
        draw=False
        
        if self.isKingCaptured(next_state):
            done=True
            self.white_reward+=self.reward_king_captured
            
        elif self.isKingOnSafeSquare(next_state):
            done=True
            self.white_reward+=self.reward_king_escape
            
        elif self.checkNoLegalMovesRemained(next_state) and self.turn==self.WHITE:
            done=True
            self.white_reward+=self.reward_king_captured
            
        elif self.checkNoLegalMovesRemained(next_state) and self.turn==self.BLACK:
            done=True
            self.white_reward+=self.reward_king_escape
            
        elif self.stateReached(next_state, -self.turn):
            done=True
            draw=True
            #No reward nor punishment for a draw
            
        #POSSIBLE IMPROVEMENT: add rewards for other behaviours
            
        else:
            if self.turn==self.WHITE:
                self.white_reward+=self.reward_white_capture*number_of_captures
                self.turn=self.BLACK
            else:
                self.white_reward+=self.reward_black_capture*number_of_captures
                self.turn=self.WHITE
                
        self.current_state=next_state
        self.reached_states.append((self.current_state,self.turn))
        
        if self.draw_board:
            self.showState(self.current_state)
        
        return next_state, self.white_reward, done, draw, self.legal_moves
    
    def isLegalMove(self, state, from_row, from_column, to_row, to_column):
        '''
        Check whether a certain move on a specific state is legal or not
        '''
        
        if from_column==to_column and from_row==to_row:
            return False
        
        if from_column==to_column:
            add = 1 if to_row>from_row else 0
            for square in range(min(to_row, from_row)+add, max(to_row, from_row)+add):
                if state[square,from_column]!=0:
                    return False
                
                type_of_square = self.board[square,from_column]
                
                if self.board[from_row,from_column] > 0: #moving from camps
                   if type_of_square!=self.board[from_row,from_column] and type_of_square!=0:
                        return False
                else: #moving from throne or normal squares
                    if type_of_square!=0:
                        return False
        else:
            add = 1 if to_column>from_column else 0
            for square in range(min(to_column, from_column)+add, max(to_column, from_column)+add):
                if state[from_row,square]!=0:
                    return False
                
                type_of_square = self.board[from_row,square]
                
                if self.board[from_row,from_column] > 0: #moving from camps
                   if type_of_square!=self.board[from_row,from_column] and type_of_square!=0:
                        return False
                else: #moving from throne or normal squares
                    if type_of_square!=0:
                        return False
                    
        return True
        
    def isLegalAction(self, state, action):
        '''
        Check whether a certain action on a specific state is legal or not
        '''
        from_coordinates, to_coordinates = self.actionToCoordinates(action)
        
        from_column = self.columns_dictionary[from_coordinates[0]]
        from_row = self.rows_dictionary[from_coordinates[1]]
        
        to_column = self.columns_dictionary[to_coordinates[0]]
        to_row = self.rows_dictionary[to_coordinates[1]]
        
        return self.isLegalMove(state,from_row, from_column, to_row, to_column)
    
    def stateReached(self, state, turn):
        for reached_state in self.reached_states:
            state_comparison = reached_state[0]==state
            if state_comparison.all() and reached_state[1]==turn:
                return True
            
        return False
    
    def areEnemies(self, piece_1, piece_2):
        '''
        Return True if the given pieces are enemies
        '''
        return piece_1 * piece_2 < 0
    
    def areAllies(self, piece_1, piece_2):
        '''
        Return True if the given pieces are allies
        '''
        return piece_1 * piece_2 >0
    
    def isCampSquare(self, square):
        '''
        Return True if the given square is a camp
        '''
        row, column = square
        return self.board[row,column]>0
    
    def isThrone(self, square):
        '''
        Return True if the given square is the throne
        '''
        row, column = square
        return self.board[row,column]<0
    
    def isNearThrone(self, square):
        '''
        Return True if the given square is adjacent to the throne
        '''
        row, column = square
        row_throne = 4
        column_throne = 4
        
        distance = row-row_throne + column-column_throne
        
        return abs(distance)==1
    
    def getAllLegalMoves(self, state, turn):
        '''
        Return a list of all possible legal moves for a specific state
        '''
        legal_moves=[]
        for i in range(9):
             for j in range(9):
                 piece=state[i][j]
                 if (turn==self.WHITE and piece>0) or (turn==self.BLACK and piece<0) :
                     for k in range(9):
                         if self.isLegalMove(state, i, j, i, k):
                                 legal_moves.append(self.squaresToAction((i,j),(i,k)))
                         if self.isLegalMove(state, i, j, k, j):
                                 legal_moves.append(self.squaresToAction((i,j),(k,j)))
        
        return legal_moves
                         
        
    def applyCaptures(self, state, moved_square):
        '''
        Execute all the possible actions on a given state knowing last move.
        Return the updated state and the number of pieces which have been captured
        '''
        moved_row, moved_column=moved_square
        piece=state[moved_row,moved_column]
        
        next_state=np.copy(state)
        number_of_captures = 0
        
        enemies=[]
        opposites=[]
        
        #UP
        if moved_row>=2:
            enemies.append((moved_row-1, moved_column))
            opposites.append((moved_row-2, moved_column))
        
        #DOWN
        if moved_row<=6:
            enemies.append((moved_row+1, moved_column))
            opposites.append((moved_row+2, moved_column))
            
        #LEFT
        if moved_column>=2:
            enemies.append((moved_row, moved_column-1))
            opposites.append((moved_row, moved_column-2))
            
        #RIGHT
        if moved_column<=6:
            enemies.append((moved_row, moved_column+1))
            opposites.append((moved_row, moved_column+2))
        
        for i in range(len(enemies)):
            enemy = enemies[i]
            opposite = opposites[i]
            
            if self.areEnemies(piece, state[enemy[0],enemy[1]]) and not self.isCampSquare(enemy):
                if self.areAllies(piece, state[opposite[0],opposite[1]]) or self.isCampSquare(opposite) or self.isThrone(opposite):
                    if state[enemy[0],enemy[1]] != 3:
                        next_state[enemy[0],enemy[1]]=0
                        number_of_captures+=1


                else:

                    if self.isThrone(enemy):
                        if state[enemy[0] + 1, enemy[1]] + state[enemy[0] - 1, enemy[1]] + state[enemy[0], enemy[1] + 1] + state[enemy[0], enemy[1] - 1] == -4:
                            next_state[enemy[0], enemy[1]] = 0
                            number_of_captures += 1


                    elif self.isNearThrone(enemy):
                        if state[enemy[0] + 1, enemy[1]] + state[enemy[0] - 1, enemy[1]] + state[enemy[0], enemy[1] + 1] + state[enemy[0], enemy[1] - 1] == -3:
                            next_state[enemy[0], enemy[1]] = 0
                            number_of_captures += 1
                                
                        else:
                            next_state[enemy[0],enemy[1]]=0
                            number_of_captures+=1
        
        return next_state, number_of_captures
        
    def checkEndOfCheckers(self, state):
        '''
        Check whether all black checkers have been captured
        '''
        return not -1 in state
        
    def isKingCaptured(self, state):
        '''
        Check if the king has been captured
        '''
        return not 3 in state
        
    def isKingOnSafeSquare(self, state):
        '''
        Check if the king has reached one of the winning square
        '''
        kingPosition=np.where(state==3)
        row=kingPosition[0][0]
        column=kingPosition[1][0]
            
        return row==0 or row==8 or column==0 or column==8
        
    def checkNoLegalMovesRemained(self, state):
        '''
        Check if a player has no possible legal moves
        '''
        return len(self.legal_moves)==0
