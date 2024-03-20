"""
Tic-Tac-Toe players using inheritance implementation by Kylie YIng
YouTube Kylie Ying: https://www.youtube.com/ycubed 
Twitch KylieYing: https://www.twitch.tv/kylieying 
Twitter @kylieyying: https://twitter.com/kylieyying 
Instagram @kylieyying: https://www.instagram.com/kylieyying/ 
Website: https://www.kylieying.com
Github: https://www.github.com/kying18 
Programmer Beast Mode Spotify playlist: https://open.spotify.com/playlist/4Akns5EUb3gzmlXIdsJkPs?si=qGc4ubKRRYmPHAJAIrCxVQ 
"""

import math
import random


class Player():
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        pass


class HumanPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        valid_square = False
        val = None
        while not valid_square:
            square = input(self.letter + '\'s turn. Input move (0-8): ')
            try:
                val = int(square)
                if val not in game.available_moves():
                    raise ValueError
                valid_square = True
            except ValueError:
                print('Invalid square. Try again.')
        return val


class RandomComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square


class SmartComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minimax(game, self.letter)['position']
        return square

    def minimax(self, state, player):
        max_player = self.letter  # yourself
        other_player = 'O' if player == 'X' else 'X'

        # first we want to check if the previous move is a winner
        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                        state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # each score should maximize
        else:
            best = {'position': None, 'score': math.inf}  # each score should minimize
        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player)  # simulate a game after making that move

            # undo move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move  # this represents the move optimal next move

            if player == max_player:  # X is max player
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best
    
import math

class SmartComputerPlayerPruning(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minimaxPruning(game, self.letter, -math.inf, math.inf)['position']
        return square

    def minimaxPruning(self, state, player, alpha, beta):
        max_player = self.letter  # yourself
        other_player = 'O' if player == 'X' else 'X'

        # first we want to check if the previous move is a winner
        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                        state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # each score should maximize
        else:
            best = {'position': None, 'score': math.inf}  # each score should minimize
        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimaxPruning(state, other_player, alpha, beta)  # simulate a game after making that move

            # undo move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move  # this represents the move optimal next move

            if player == max_player:  # X is max player
                if sim_score['score'] > best['score']:
                    best = sim_score
                alpha = max(alpha, best['score'])
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
                beta = min(beta, best['score'])
                
            if alpha >= beta:  # pruning condition
                break
        return best
    
from typing import Dict, List
import numpy as np
class Qlearning(Player):
    def __init__(self, letter, alpha=0.9, gamma=0.95, q_init=0.6):
        """
        Called when creating a new TQPlayer. Accepts some optional parameters to define its learning behaviour
        :param alpha: The learning rate needs to be larger than 0 and smaller than 1
        :param gamma: The reward discount. Needs to be larger than 0  and should be smaller than 1. Values close to 1
            should work best.
        :param q_init: The initial q values for each move and state.
        """
        self.side = None
        self.qtable = {}  # type: Dict[int, np.ndarray]
        self.move_history = []  # type: List[(int, int)]
        self.learning_rate = alpha
        self.value_discount = gamma
        self.q_init_val = q_init
        self.epsilon = 0.1
        self.current_state =None
        super().__init__(letter)

    def get_move(self, game):
        return self.make_move(self.letter,None, game)

    def keywithmaxval(self, d):
     """ a) create a list of the dict's keys and values; 
         b) return the key with the max value
         
         
     Based on https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary"""  
     k=list(d.keys())
     # boltzmann
     v = np.array(list(d.values()))
     x = k[int(random.choice(np.argwhere(v == np.amax(v))))]  # If there are multiple max values, choose randomly
     return x
    
    def make_move(self, move, letter, game):
        qplayer = self.letter  # yourself
        other_player = 'O' if letter == 'X' else 'X'
        # first we want to check if the previous move is a winner
        if game.current_winner == other_player:
            return {'position': None, 'score': 1 * (game.num_empty_squares() + 1) if other_player == qplayer else -1 * (
                        game.num_empty_squares() + 1)}
        elif not game.empty_squares():
            return {'position': None, 'score': 0}
        

        # Make a choice what move to take next
        possible_moves = game.available_moves()
        #self.current_state = game.get_current_state()
        self.current_state = game.get_state()

        # If the current_state does not exist in the qtable, insert it
        if self.current_state not in self.qtable:
            # New entry in the qtable, init to zero. 
            #self.board_to_state[self.current_state] = game.get_board()
            action_vs_qvalue = dict()
            for action in possible_moves:
                action_vs_qvalue[action] = 0
            #self.state_list.append(self.current_state)  # For plotting later
            self.qtable[self.current_state] = action_vs_qvalue
            
        # Insert epsilon choice here, exploit or explore
        if random.uniform(0, 1) < self.epsilon:
            x = random.choice(possible_moves)   # Random choice
        else:  # Exploit our qtable
            x = self.keywithmaxval(self.qtable[self.current_state])  # Optimal choice
        
        return x  
    
