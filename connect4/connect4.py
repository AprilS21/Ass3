#Credit: https://github.com/Weiwei-Wan/AI_Connect4_MiniMax-Q-learning/tree/main
import numpy as np

HEIGHT = 5
WIDTH = 6

class Board():
    def __init__(self):
        self.board = np.zeros((HEIGHT,WIDTH))

    def playerMove(self, y, player):
        for i in range(HEIGHT-1, -1, -1):
            if self.board[i,y]==0:
                self.board[i,y] = player
                break
    
    def getBoard(self):
        return self.board

    def printBoard(self):
        for row in range(HEIGHT):
            print("|", end="")
            for col in range(WIDTH):
                if self.board[row, col] == 0:
                    print(" ", end="|")
                elif self.board[row, col] == 1:
                    print("X", end="|")
                elif self.board[row, col] == 2:
                    print("O", end="|")
            print()
        print("-" * (2 * WIDTH + 1))

import random
from player import *
import matplotlib.pyplot as plt

def trainQLearning(trainNum):
    playerQ1 = PlayQlearning(1)
    playerQ2 = PlayQlearning(2)
    count = 0
    while count < trainNum:
        print(count)
        count += 1
        # random first player
        Player1First = random.choice([True, False])
        # initialize board
        board = Board()
        playerQ1.newGame()
        playerQ2.newGame()
        # play game to finish
        while checkWinner(board.getBoard()) == 0:
            # First Player
            playerQ = playerQ1 if Player1First else playerQ2
            y1 = playerQ.findBestMove(board.getBoard())
            board.playerMove(y1, 1 if Player1First else 2)
            # game finish
            if checkWinner(board.getBoard()) != 0:
                playerQ1.finalResult(checkWinner(board.getBoard()))
                playerQ2.finalResult(checkWinner(board.getBoard()))
                break
            else:
                # second player
                playerQ = playerQ2 if Player1First else playerQ1
                y2 = playerQ.findBestMove(board.getBoard())
                board.playerMove(y2, 2 if Player1First else 1)
                # game finish
                if checkWinner(board.getBoard()) != 0:
                    playerQ1.finalResult(checkWinner(board.getBoard()))
                    playerQ2.finalResult(checkWinner(board.getBoard()))
                    break
    return playerQ1, playerQ2

def play(player1, player2, loopNum, trainNum, print_Board=False):
    player1WinCount = 0
    player2WinCount = 0
    tiedCount = 0
    # if q-learning player
    if player1 == 2 or player2 == 2:
        playerQ1, playerQ2 = trainQLearning(trainNum)
    count = 0

    while count < loopNum:
        print(count)
        count += 1
        # random first player
        Player1First = random.choice([True, False])
        # initialize board
        board = Board()
        if player1 == 2 or player2 == 2:
            playerQ1.newGame()
            playerQ2.newGame()
        # play game to finish
        while checkWinner(board.getBoard()) == 0:
            # First Player
            player = player1 if Player1First else player2
            if player1 == 2 or player2 == 2:
                playerQ = playerQ1 if Player1First else playerQ2
        
            if player == 0:
                y1 = defaultPlayer(1 if Player1First else 2, board.getBoard())
            elif player == 1:
                y1 = miniMaxAlgoPruning(1 if Player1First else 2, board.getBoard()) 
            elif player == 2:
                y1 = playerQ.findBestMove(board.getBoard())
            elif player == 3:
                y1 = miniMaxAlgo(1 if Player1First else 2, board.getBoard())
            elif player == 4:
                y1 = randomPlayer(board.getBoard())

            board.playerMove(y1, 1 if Player1First else 2)
            if print_Board: 
                board.printBoard()
            # game finish
            if checkWinner(board.getBoard()) != 0:
                if checkWinner(board.getBoard()) == 3:
                    tiedCount += 1
                elif checkWinner(board.getBoard()) == 1:
                    player1WinCount += 1
                elif checkWinner(board.getBoard()) == 2:
                    player2WinCount += 1

                if player1 == 2:
                    playerQ1.finalResult(checkWinner(board.getBoard()))
                if player2 == 2:
                    playerQ2.finalResult(checkWinner(board.getBoard()))

                break
            else:
                # second player
                player = player2 if Player1First else player1
                if player1 == 2 or player2 == 2:
                    playerQ = playerQ2 if Player1First else playerQ1
                
                if player == 0:
                    y2 = defaultPlayer(2 if Player1First else 1, board.getBoard())
                elif player == 1:
                    y2 = miniMaxAlgoPruning(2 if Player1First else 1, board.getBoard()) 
                elif player == 2:
                    y2 = playerQ.findBestMove(board.getBoard())
                elif player == 3:
                    y2 = miniMaxAlgo(2 if Player1First else 1, board.getBoard())
                elif player == 4:
                    y2 = randomPlayer(board.getBoard())
                

                board.playerMove(y2, 2 if Player1First else 1)
                # game finish
                if checkWinner(board.getBoard()) != 0:
                    if checkWinner(board.getBoard()) == 3:
                        tiedCount += 1
                    elif checkWinner(board.getBoard()) == 1:
                        player1WinCount += 1
                    elif checkWinner(board.getBoard()) == 2:
                        player2WinCount += 1

                    if player1 == 2:
                        playerQ1.finalResult(checkWinner(board.getBoard()))
                    if player2 == 2:
                        playerQ2.finalResult(checkWinner(board.getBoard()))

                    break
    plot_results(player1WinCount, player2WinCount, tiedCount)

def plot_results(player1_wins, player2_wins, ties):
    labels = ['Player 1 Wins', 'Player 2 Wins', 'Ties']
    values = [player1_wins, player2_wins, ties]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['blue', 'green', 'orange'])

    ax.set_xlabel('Results')
    ax.set_ylabel('Number of Games')
    ax.set_title('Results of Two Algorithms Playing a Game')

    plt.show()

def main():
    defaultPlayer =0
    miniMaxPrune = 1
    miniMaxNotPrune = 3
    qlearning =2
    random = 4
    
    #play(player1=qlearning, player2=miniMaxPrune, loopNum=50, trainNum=100000, print_Board=False)
    #play(player1=qlearning, player2=miniMaxNotPrune, loopNum=50, trainNum=100000, print_Board=False)
    #play(player1=qlearning, player2=random, loopNum=50, trainNum=100000, print_Board=False)
    #play(player1=miniMaxNotPrune, player2=defaultPlayer, loopNum=50, trainNum=100000, print_Board=False)
    #play(player1=miniMaxPrune, player2=defaultPlayer, loopNum=50, trainNum=100000, print_Board=False)

    play(player1=defaultPlayer, player2=defaultPlayer, loopNum=2, trainNum=100000, print_Board=True)
    play(player1=miniMaxPrune, player2=defaultPlayer, loopNum=100, trainNum=100000, print_Board=False)
main()