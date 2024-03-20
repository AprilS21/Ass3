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
import sys
import time

import tqdm
from player import *


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
                return letter  # ends the loop and exits the game
            letter = 'O' if letter == 'X' else 'X'  # switches player

        time.sleep(.8)

    if print_game:
        print('It\'s a tie!')

# def update_qtable(player):
#     # Update the qtable
#     old_value = player.qtable[player.current_state][player.action]
#     try:
#         next_max = max(player.qtable[player.new_state_after_tree].values())
#     except KeyError:  # In case the tree for next state has not been made yet, simply return 0
#         next_max = 0
#     # Note that the reward we actually get is the reward after the tree has made its move. We then reverse that reward vs the lut to get our own. 
#     new_value = (1 - player.alpha) * old_value + player.alpha * (player.our_reward_lut[player.reward_after_tree] + player.gamma * next_max)
#     player.qtable[player.current_state][player.action] = new_value

# def train_q_learning(x_player, o_player, board):
#     # Create the board and player
#     start = datetime.now()
#     tictactoe = board

#     logging.info('Starting the training loop...')
#     no_episodes = 5     # Get the number of episodes to run from the input args
#     ep =0
#     while ep < no_episodes:
#         play(tictactoe, x_player, o_player, print_game=True)
#         #update_qtable(x_player)
#         tictactoe.make_board()
#         ep +=1

#     # logging.info('Saving the model...')
#     # with open('trained_player.pkl', 'wb') as f:
#     #     dill.dump(player_tree, f)

#     logging.info('Training finished...')
#     logging.info('done (%s)...' % (datetime.now() - start))
        

def play_and_train(game, x_player, o_player, print_game=True):
    if print_game:
        game.print_board_nums()

    letter = 'X'
    x_player.current_state = game.get_state()  # Initialize current state
    while game.empty_squares():
        if letter == 'O':
            square = o_player.get_move(game)
        else:
            square = x_player.get_move(game)
            #x_player.current_state =  game.get_state()
            #update_qtable(x_player, None, game)
            #x_player.make_move(square, letter, game)
        if game.make_move(square, letter):

            if print_game:
                print(letter + ' makes a move to square {}'.format(square))
                game.print_board()
                print('')

            if game.current_winner:
                if print_game:
                    print(letter + ' wins!')
                update_qtable(x_player, letter, game)  # Train the Q-learning model
                return letter  # ends the loop and exits the game
            letter = 'O' if letter == 'X' else 'X'  # switches player

        time.sleep(.8)

    if print_game:
        print('It\'s a tie!')
    update_qtable(x_player, 'tie', game)  # Train the Q-learning model after a tie


def update_qtable(player, game_result, game):
    # Extract relevant information
    state = player.current_state
    action = player.action
    qtable = player.qtable

    # Define rewards
    if game_result == player.letter:
        reward = 1  # Winning reward
    elif game_result == 'O' or game_result == 'X':
        reward = -1  # Losing reward
    else:
        reward = 0  # Tie reward

    # Update Q-value for the current state-action pair
    current_qvalue = qtable[state][action]
    max_next_qvalue = max(qtable[state].values()) if qtable[state] else 0  # If no actions yet, set to 0
    new_qvalue = current_qvalue + player.learning_rate * (reward + player.value_discount * max_next_qvalue - current_qvalue)
    qtable[state][action] = new_qvalue

    # Reset current state
    player.current_state = game.get_state()  # Get current state after making the move

# Training the Q-learning model on 100 games
def train_qlearning_model():
    x_player = Qlearning('X')
    o_player = RandomComputerPlayer('O')
    game = TicTacToe()
    for _ in range(100):
        play_and_train(game, x_player, o_player, print_game=False)



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




if __name__ == '__main__':
    x_player = Qlearning('X')
    #o_player = HumanPlayer('O')
    #o_player = SmartComputerPlayerPruning('O')
    o_player = RandomComputerPlayer('O')
    t = TicTacToe()
    train_qlearning_model2(5)
    #play(t, x_player, o_player, print_game=True)


