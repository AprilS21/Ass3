"""
Tic Tac Toe class + game play implementation by Kylie Ying
YouTube Kylie Ying: https://www.youtube.com/ycubed 
Twitch KylieYing: https://www.twitch.tv/kylieying 
Twitter @kylieyying: https://twitter.com/kylieyying 
Instagram @kylieyying: https://www.instagram.com/kylieyying/ 
Website: https://www.kylieying.com
Github: https://www.github.com/kying18 
Programmer Beast Mode Spotify playlist: https://open.spotify.com/playlist/4Akns5EUb3gzmlXIdsJkPs?si=qGc4ubKRRYmPHAJAIrCxVQ 
"""

from datetime import datetime
import logging
import math
import pickle
import sys
import time

import tqdm
from player import *

winner_global =' '

class TicTacToe():
    def __init__(self):
        self.board = self.make_board()
        self.current_winner = None

    @staticmethod
    def make_board():
        return [' ' for _ in range(9)]

    def print_board(self):
        for row in [self.board[i*3:(i+1) * 3] for i in range(3)]:
            print('| ' + ' | '.join(row) + ' |')

    @staticmethod
    def print_board_nums():
        # 0 | 1 | 2
        number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
        for row in number_board:
            print('| ' + ' | '.join(row) + ' |')

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.winner(square, letter):
                self.current_winner = letter
            return True
        return False
    
    def get_state(self):
        state = ''
        for x in self.board:
            state += x
        return state


    def winner(self, square, letter):
        # check the row
        row_ind = math.floor(square / 3)
        row = self.board[row_ind*3:(row_ind+1)*3]
        # print('row', row)
        if all([s == letter for s in row]):
            return True
        col_ind = square % 3
        column = [self.board[col_ind+i*3] for i in range(3)]
        # print('col', column)
        if all([s == letter for s in column]):
            return True
        if square % 2 == 0:
            diagonal1 = [self.board[i] for i in [0, 4, 8]]
            # print('diag1', diagonal1)
            if all([s == letter for s in diagonal1]):
                return True
            diagonal2 = [self.board[i] for i in [2, 4, 6]]
            # print('diag2', diagonal2)
            if all([s == letter for s in diagonal2]):
                return True
        return False

    def empty_squares(self):
        return ' ' in self.board

    def num_empty_squares(self):
        return self.board.count(' ')

    def available_moves(self):
        return [i for i, x in enumerate(self.board) if x == " "]



def play(game, x_player, o_player, print_game=True):
    global winner_global
    winner_global = ' '
    if print_game:
        game.print_board_nums()

    letter = 'X'
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)
        if game.make_move(square, letter):

            if print_game:
                print(letter + ' makes a move to square {}'.format(square))
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                winner_global = letter
                return letter  # ends the loop and exits the game
            letter = 'O' if letter == 'X' else 'X'  # switches player

        time.sleep(.8)

    if print_game:
        print('It\'s a tie!')



def train_qlearning_model2(num_games=100):
    """
    Trains the Q-learning model by playing a specified number of games.
    :param num_games: The number of games to play for training.
    """
    for game_num in range(num_games):
        print(f"Game {game_num + 1}")
        game = TicTacToe()
        winner = play(game, x_player, o_player, print_game=False)
        # Update Q-table based on the outcome of the game
        if winner == 'X':
            reward = 1
        elif winner == 'O':
            reward = -1
        else:
            reward = 0
        # Update Q-table for the last move made by the Q-learning player
        last_state = game.get_state()
        last_action = x_player.action
        # Ensure the last state exists in the Q-table with all possible actions
        if last_state not in x_player.qtable:
            x_player.qtable[last_state] = {action: 0 for action in range(9)} # Assuming 9 possible actions
        # Update the Q-value for the last action taken
        max_future_q = np.max(list(x_player.qtable[last_state].values()))
        current_q = x_player.qtable[last_state][last_action]
        new_q = (1 - x_player.learning_rate) * current_q + x_player.learning_rate * (reward + x_player.value_discount * max_future_q)
        x_player.qtable[last_state][last_action] = new_q
        # Decay epsilon for exploration vs. exploitation trade-off
        x_player.epsilon *= 0.995
        # Reset the game and Q-learning player for the next iteration
        game = TicTacToe()
        x_player.reset()

        # Save the Q-table after training
    with open('qtable.pkl', 'wb') as f:
        pickle.dump(x_player.qtable, f)


def load_qtable(file_path='qtable.pkl'):
    """
    Loads the Q-table from a file.
    :param file_path: The path to the file where the Q-table is stored.
    :return: The loaded Q-table.
    """
    with open(file_path, 'rb') as f:
        qtable = pickle.load(f)
    return qtable
import matplotlib.pyplot as plt

def plot_results(player1_wins, player2_wins, ties):
    labels = ['Player 1 Wins', 'Player 2 Wins', 'Ties']
    values = [player1_wins, player2_wins, ties]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'green', 'orange'])

    ax.set_xlabel('Results')
    ax.set_ylabel('Number of Games')
    ax.set_title('Results of Two Algorithms Playing a Game')

    plt.show()


if __name__ == '__main__':
    qtable = load_qtable()
    #x_player = Qlearning('X', qtable)
    #o_player = HumanPlayer('O')
    #x_player = SmartComputerPlayerPruning('X')
    x_player = RandomComputerPlayer('X')
    o_player = RandomComputerPlayer('O')
    #o_player = DefaultComputerPlayer('O')
    #o_player = SmartComputerPlayer('O')
    t = TicTacToe()
    #train_qlearning_model2(100)
    #o_player = HumanPlayer('O')
    wins_x = 0
    wins_o =0
    ties =0
    #play(t, x_player, o_player, print_game=True)
    for _ in range(10):
        play(t, x_player, o_player, print_game=False)
        #print(winner_global)
        t = TicTacToe()
        x_player.reset()
        o_player.reset()
        if winner_global == 'X':
            wins_x += 1
        elif  winner_global == 'O':
                wins_o+=1
        else:
            ties += 1

    plot_results(wins_x, wins_o, ties)

    



