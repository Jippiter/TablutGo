# -*- coding: utf-8 -*-

import os
import glob
import TablutEnvironment
import TablutAgent

def compare(player_1_path, player_2_path, games=100, player_1_split_channels=True, player_2_split_channels=True):

    #Initialize
    
    parameters_player_1 = open("../Weights/" + player_1_path + "Parameters.txt","r")
    print("Player 1 parameters:")
    print (parameters_player_1.read())
    parameters_player_2 = open("../Weights/" + player_2_path + "Parameters.txt","r")
    print("\nPlayer 2 parameters: ")
    print (parameters_player_2.read() + "\n")
    
    show_board = True #set True to watch the games on a board (this does not affect performances)
    
    board_path = "../Resources/Board.png"
    
    player_1 = TablutAgent.DQNAgent(action_size=9*9*16, 
                                       gamma=0, 
                                       epsilon=0, 
                                       epsilon_min=0, 
                                       epsilon_decay=0, 
                                       learning_rate=0, 
                                       batch_size=0, 
                                       memory_len=1, 
                                       update_model_target=0,
                                       split_channels=player_1_split_channels,
                                       colour="W")
    
    player_2 = TablutAgent.DQNAgent(action_size=9*9*16, 
                                       gamma=0, 
                                       epsilon=1, 
                                       epsilon_min=1, 
                                       epsilon_decay=1, 
                                       learning_rate=0, 
                                       batch_size=0, 
                                       memory_len=1, 
                                       update_model_target=0,
                                       split_channels=player_2_split_channels,
                                       colour="B")
    
    env = TablutEnvironment.Environment(reward_king_captured=0, 
                                        reward_king_escape=0, 
                                        reward_white_capture=0, 
                                        reward_black_capture=0,
                                        reward_king_closer_edge=0,
                                        reward_king_further_black=0,
                                        reward_king_freedom=0,
                                        reward_neutral_move=0,
                                        board_path=board_path, 
                                        draw_board=show_board)
    
    player_1_weights_white = glob.glob("../Weights/" + player_1_path + "*white*.hdf5")
    player_1_weights_black = glob.glob("../Weights/" + player_1_path + "*black*.hdf5")
    player_2_weights_white = glob.glob("../Weights/" + player_2_path + "*white*.hdf5")
    player_2_weights_black = glob.glob("../Weights/" + player_2_path + "*black*.hdf5")
    
    #Round one: player one has white
    
    if len(player_1_weights_white)>0 and len(player_2_weights_black)>0:
        player_1.load(player_1_weights_white[-1])
        player_2.load(player_2_weights_black[-1])
        
    player_1_white_score=0
        
    for game in range(games):
        state, legal_moves = env.reset()
        
        moves=0
        
        while True:
            moves+=1
            
            action = player_1.act(state, legal_moves)
            
            next_state, reward, done, draw, legal_moves = env.step(action)
            
            state = next_state
            
            if done:
                result = "White won" if not draw else "Draw"
                print ("Game n.{} has ended: ".format(game+1) + result + " after {} moves".format(moves))
                
                if draw:
                    player_1_white_score+=0.5
                else:
                    player_1_white_score+=1
                
                break
            
            action = player_2.act(state, legal_moves)
            
            next_state, reward, done, draw, legal_moves = env.step(action)
            
            state = next_state
            
            if done:
                result = "Black won" if not draw else "Draw"
                print ("Game n.{} has ended: ".format(game+1) + result + " after {} moves".format(moves))
                break
            
    print("Player 1 score as white: {}%".format(player_1_white_score / games * 100))
    
    #Round two: player one has black
    
    if len(player_1_weights_black)>0 and len(player_2_weights_white)>0:
        player_1.load(player_1_weights_black[-1])
        player_2.load(player_2_weights_white[-1])
        
    player_1_black_score=0
        
    for game in range(games):
        state, legal_moves = env.reset()
        
        moves=0
        
        while True:
            moves+=1
            
            action = player_2.act(state, legal_moves)
            
            next_state, reward, done, draw, legal_moves = env.step(action)
            
            state = next_state
            
            if done:
                result = "White won" if not draw else "Draw"
                print ("Game n.{} has ended: ".format(game+1) + result + " after {} moves".format(moves))
                break
            
            action = player_1.act(state, legal_moves)
            
            next_state, reward, done, draw, legal_moves = env.step(action)
            
            state = next_state
            
            if done:
                result = "Black won" if not draw else "Draw"
                print ("Game n.{} has ended: ".format(game+1) + result + " after {} moves".format(moves))
                
                if draw:
                    player_1_black_score+=0.5
                else:
                    player_1_black_score+=1
                
                break
            
    print("Player 1 score as black: {}%".format(player_1_black_score / games * 100))
    
    print("Player 1 scored: {}".format(player_1_white_score + player_1_black_score), "/{}".format(games*2))

#Example
compare("Gaetano new CNN - three channels fast/", "Gaetano new CNN - three channels fast/", player_1_split_channels=True, player_2_split_channels=True)