# -*- coding: utf-8 -*-

import os
import glob
import TablutEnvironment
import TablutAgent

def saveParameters(path,gamma, epsilon_min, epsilon_decay, learning_rate, batch_size, split_input_channels, update_model_target, reward_king_captured, reward_king_escape, reward_white_capture, reward_black_capture):
    '''
    Save the hyperparameters into a txt file
    '''
    file = open(path + "Parameters.txt", "w")
    file.write("gamma = " + str(gamma) + "\n")
    file.write("epsilon-min = " + str(epsilon_min) + "\n")
    file.write("epsilon decay = " + str(epsilon_decay) + "\n")
    file.write("learning rate = " + str(learning_rate) + "\n\n")
    file.write("batch size = " + str(batch_size) + "\n")
    file.write("split input channels = " + str(split_input_channels) + "\n")
    file.write("update model target steps = " + str(update_model_target) + "\n\n")
    file.write("reward king captured = " + str(reward_king_captured) + "\n")
    file.write("reward king escape = " + str(reward_king_escape) + "\n")
    file.write("reward white capture = " + str(reward_white_capture) + "\n")
    file.write("reward black capture = " + str(reward_black_capture))

    file.close()

#Parameters

gamma = 0.95 #discount factor
epsilon = 1.0 #exploration probability (random move choice)
epsilon_min = 0.05 #lower bound for epsilon
epsilon_decay = 0.9995 #speed for epsilon decay at each learning step (replay)
learning_rate = 0.0005
batch_size = 32 #number of samples for replay
moves_before_replay = 5000 #play this number of moves to get some experience before starting the replay
memory_len=10000 #max number of last moves to keep in memory
split_input_channels = True #set to True to split CNN's board input state into two channels (white pieces and black ones)
action_size=9*9*16 #number of possible actions (moves); output for the CNN
number_of_games=10000 #ideal numbe of games to play before the algorithm stops (not important, as it can be manually stopped and executed again)
update_model_target= 500 #number of moves required to update weights on the model target

#These rewards refer to white's perspective
reward_king_captured=-100 #reward for capturing the king
reward_king_escape=100 #reward for reaching a winning square with the king
reward_white_capture=3 #reward for capturing a black piece
reward_black_capture=-5 #reward for capturing a white piece

show_board = True #set True to watch the games on a board (this operation does not affect performances)

#REMEMBER: keep the / at the end of the path
cnn_weights_path = "Second Test/" #Change folder name to start another training from zero; use this to make different tests with different hyperparameters

save_weights_step = 25 #Save the CNNs' weights after each multiple of this number

board_path = "Resources/Board.png"

#Initialize

agent_white = TablutAgent.DQNAgent(action_size=action_size, 
                                   gamma=gamma, 
                                   epsilon=epsilon, 
                                   epsilon_min=epsilon_min, 
                                   epsilon_decay=epsilon_decay, 
                                   learning_rate=learning_rate, 
                                   batch_size=batch_size, 
                                   memory_len=memory_len, 
                                   update_model_target=update_model_target,
                                   split_channels=split_input_channels,
                                   colour="W")

agent_black = TablutAgent.DQNAgent(action_size=action_size, 
                                   gamma=gamma, 
                                   epsilon=epsilon, 
                                   epsilon_min=epsilon_min, 
                                   epsilon_decay=epsilon_decay, 
                                   learning_rate=learning_rate, 
                                   batch_size=batch_size, 
                                   memory_len=memory_len, 
                                   update_model_target=update_model_target,
                                   split_channels=split_input_channels,
                                   colour="B")

env = TablutEnvironment.Environment(reward_king_captured=reward_king_captured, 
                                    reward_king_escape=reward_king_escape, 
                                    reward_white_capture=reward_white_capture, 
                                    reward_black_capture=reward_black_capture, 
                                    board_path=board_path, 
                                    draw_board=show_board)

output_dir = "Weights/" + cnn_weights_path

if not os.path.exists(output_dir):
    starting_game_number=0
    os.makedirs(output_dir)
    saveParameters(path=output_dir,
                   gamma=gamma,
                   epsilon_min=epsilon_min,
                   epsilon_decay=epsilon_decay,
                   learning_rate=learning_rate,
                   batch_size=batch_size,
                   split_input_channels=split_input_channels,
                   update_model_target=update_model_target,
                   reward_king_captured=reward_king_captured,
                   reward_king_escape=reward_king_escape,
                   reward_white_capture=reward_white_capture,
                   reward_black_capture=reward_black_capture)
    
else: #load weights
    weights_white = glob.glob(output_dir + "*white*.hdf5")
    weights_black = glob.glob(output_dir + "*black*.hdf5")
    if len(weights_white)>0 and len(weights_black)>0: #if they exist, load last (best) weights
        agent_white.load(weights_white[-1])
        agent_black.load(weights_black[-1])
        number = weights_white[-1][-9:]
        starting_game_number = int(number[0:4]) #restart from the last played game count
        print("Weights loaded")
        
replay_mode = False
random_games=0

#Start

print("Playing random games to gain experience...")
#Deep Q-Learning algorithm applied with two agents (black and white) playing one against the other
for e in range(starting_game_number,number_of_games):
    
    state, legal_moves = env.reset()
    
    moves=0
    while True:
        moves+=1
        
        action = agent_white.act(state, legal_moves)
        
        next_state, reward, done, draw, legal_moves = env.step(action)
        
        agent_white.remember(state, action, reward, next_state, done, legal_moves)
        
        state = next_state
            
        if len(agent_white.memory) > moves_before_replay:
            if not replay_mode:
                replay_mode = True
                random_games=e
                print("Replay mode started...")
            agent_white.replay(batch_size)
            
        if done:
            result = "White won" if not draw else "Draw"
            headline = "Random game" if not replay_mode else "Game"
            print (headline, "n.{} has ended: ".format(e+1 - random_games) + result + " after {} moves".format(moves))
            break
        
        action = agent_black.act(state, legal_moves)
        
        next_state, reward, done, draw, legal_moves = env.step(action)
        
        agent_black.remember(state, action, reward, next_state, done, legal_moves)
        
        state = next_state
            
        if len(agent_black.memory) > moves_before_replay:
            if not replay_mode:
                replay_mode = True
                random_games=e
                print("Replay mode started...")
            agent_black.replay(batch_size)
            
        if done:
            result = "Black won" if not draw else "Draw"
            headline = "Random game" if not replay_mode else "Game"
            print (headline,"n.{} has ended: ".format(e+1 - random_games) + result + " after {} moves".format(moves))
            break
            
    if (e-random_games+1) % save_weights_step == 0 and replay_mode:
        agent_white.save(output_dir + "weights_white" + "{:04d}".format(e-random_games) + ".hdf5")
        agent_black.save(output_dir + "weights_black" + "{:04d}".format(e-random_games) + ".hdf5")
        
        print("Weights saved")