#Credit: https://github.com/kying18/tic-tac-toe

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
    def reset(self):
        self.current_state = None
        self.action = None

class DefaultComputerPlayer(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        # Check for winning move or move to prevent opponent winning
        for possible_move in game.available_moves():
            game.make_move(possible_move, self.letter)
            if game.current_winner == self.letter:
                return possible_move
            game.make_move(possible_move, 'O' if self.letter == 'X' else 'X')
            if game.current_winner == ('O' if self.letter == 'X' else 'X'):
                game.board[possible_move] = ' ' 
                return possible_move
            game.board[possible_move] = ' '

        square = random.choice(game.available_moves())
        return square

    def reset(self):
        self.current_state = None
        self.action = None


class Minimax(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minimax(game, self.letter)['position']
        return square

    def minimax(self, state, player):
        max_player = self.letter 
        other_player = 'O' if player == 'X' else 'X'

        # check if winner
        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                        state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # maximize
        else:
            best = {'position': None, 'score': math.inf}  # minimize
        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimax(state, other_player) 

            # undo move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move 

            if player == max_player: 
                if sim_score['score'] > best['score']:
                    best = sim_score
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
        return best
    def reset(self):
        self.current_state = None
        self.action = None
    
import math

class MinimaxPruning(Player):
    def __init__(self, letter):
        super().__init__(letter)

    def get_move(self, game):
        if len(game.available_moves()) == 9:
            square = random.choice(game.available_moves())
        else:
            square = self.minimaxPruning(game, self.letter, -math.inf, math.inf)['position']
        return square

    def minimaxPruning(self, state, player, alpha, beta):
        max_player = self.letter 
        other_player = 'O' if player == 'X' else 'X'

        # check if winner
        if state.current_winner == other_player:
            return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
                        state.num_empty_squares() + 1)}
        elif not state.empty_squares():
            return {'position': None, 'score': 0}

        if player == max_player:
            best = {'position': None, 'score': -math.inf}  # maximize
        else:
            best = {'position': None, 'score': math.inf}  # minimize
        for possible_move in state.available_moves():
            state.make_move(possible_move, player)
            sim_score = self.minimaxPruning(state, other_player, alpha, beta)  

            # undo move
            state.board[possible_move] = ' '
            state.current_winner = None
            sim_score['position'] = possible_move  

            if player == max_player:  
                if sim_score['score'] > best['score']:
                    best = sim_score
                alpha = max(alpha, best['score'])
            else:
                if sim_score['score'] < best['score']:
                    best = sim_score
                beta = min(beta, best['score'])
                
            if alpha >= beta: 
                break
        return best
    def reset(self):
        self.current_state = None
        self.action = None
    
from typing import Dict, List
import numpy as np

class Qlearning(Player):
    def __init__(self, letter, qtable = None, alpha=0.9, gamma=0.95, q_init=0.6):
        self.side = None
        self.qtable = qtable if qtable is not None else {}
        self.learning_rate = alpha
        self.value_discount = gamma
        self.q_init_val = q_init
        self.epsilon = 0.1
        self.current_state =None
        super().__init__(letter)

    def get_move(self, game):
        return self.make_move(self.letter,None, game)

    def keywithmaxval(self, d):  
     k=list(d.keys())
     v = np.array(list(d.values()))
     x = k[int(random.choice(np.argwhere(v == np.amax(v))))]
     return x
    
    def make_move(self, move, letter, game):
        qplayer = self.letter  
        other_player = 'O' if letter == 'X' else 'X'
        if game.current_winner == other_player:
            return {'position': None, 'score': 1 * (game.num_empty_squares() + 1) if other_player == qplayer else -1 * (
                        game.num_empty_squares() + 1)}
        elif not game.empty_squares():
            return {'position': None, 'score': 0}
        
        possible_moves = game.available_moves()
        self.current_state = game.get_state()

        if self.current_state not in self.qtable:
            action_vs_qvalue = dict()
            for action in possible_moves:
                action_vs_qvalue[action] = 0
            self.qtable[self.current_state] = action_vs_qvalue
            
        if random.uniform(0, 1) < self.epsilon:
            x = random.choice(possible_moves) 
        else: 
            x = self.keywithmaxval(self.qtable[self.current_state])
        self.action = x
        self.current_state = game.get_state()
        return x  
    
    def reset(self):
        self.current_state = None
        self.action = None
