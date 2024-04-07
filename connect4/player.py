#Credit: : https://github.com/Weiwei-Wan/AI_Connect4_MiniMax-Q-learning/tree/main
import pickle
import numpy as np
import random

HEIGHT = 5
WIDTH = 6

def checkWinner(the_board):
    # check column
    for i in range(HEIGHT-3):
        for j in range(WIDTH):
            if the_board[i][j]!=0 and the_board[i+1][j]==the_board[i][j] and the_board[i+2][j]==the_board[i][j] and the_board[i+3][j]==the_board[i][j]:
                return the_board[i][j]
    # check row
    for j in range(WIDTH-3):
        for i in range(HEIGHT):
            if the_board[i][j]!=0 and the_board[i][j+1]==the_board[i][j] and the_board[i][j+2]==the_board[i][j] and the_board[i][j+3]==the_board[i][j]:
                return the_board[i][j]
    # check diagonal left
    for i in range(3,HEIGHT):
        for j in range(WIDTH-3):
            if the_board[i][j]!=0 and the_board[i-1][j+1]==the_board[i][j] and the_board[i-2][j+2]==the_board[i][j] and the_board[i-3][j+3]==the_board[i][j]:
                return the_board[i][j]
    # check diagonal right
    for i in range(HEIGHT-3):
        for j in range(3):
            if the_board[i][j]!=0 and the_board[i+1][j+1]==the_board[i][j] and the_board[i+2][j+2]==the_board[i][j] and the_board[i+3][j+3]==the_board[i][j]:
                return the_board[i][j]
    # not finished
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if the_board[i][j] == 0:
                return 0
    # tied
    return 3

def randomPlayer(board):
    choices = []
    for j in range(WIDTH):
        for i in range(HEIGHT):
            if board[i][j] == 0:
                choices.append(j)
                break
        
    final_choice = random.choice(choices)
    return final_choice


def miniMaxAlgoPruning(the_player, board):
    global player, opponent
    player = 1
    opponent = 2
    if (the_player == 2):
        opponent = 1
        player = 2
    return findBestMovePruning(board)
def miniMaxAlgo(the_player, board):
    global player, opponent
    player = 1
    opponent = 2
    if (the_player == 2):
        opponent = 1
        player = 2
    return findBestMove(board)  

def miniMaxPruning(the_board, columnHeights, depth, isMax, alpha, beta, state_count):
    score = 0
    if checkWinner(the_board) == 3: # tied
        return 0
    elif checkWinner(the_board) == player: # win
        score = 10
        return score
    elif checkWinner(the_board) == opponent: # lose
        score = -10
        return score
    if depth == 5: # Check if the maximum depth is reached
        return 0

    #state_count[0] += 1  # Increment state count

    if isMax:	
        best = -1000
        for j in range(WIDTH):
            if columnHeights[j]<HEIGHT:
                columnHeights[j] += 1
                the_board[HEIGHT-int(columnHeights[j])][j] = player
                new_best = max(best, miniMaxPruning(the_board, columnHeights, depth + 1, not isMax, alpha, beta, state_count))
                the_board[HEIGHT-int(columnHeights[j])][j] = 0
                columnHeights[j] -= 1
                if new_best > best:
                    best = new_best
                if best >= beta:
                    return best
                if best > alpha:
                    alpha = best
        return new_best
    else:
        best = 1000
        for j in range(WIDTH):
            if columnHeights[j]<HEIGHT:
                columnHeights[j] += 1
                the_board[HEIGHT-int(columnHeights[j])][j] = opponent
                new_best = min(best, miniMaxPruning(the_board, columnHeights, depth + 1, not isMax, alpha, beta, state_count))
                the_board[HEIGHT-int(columnHeights[j])][j] = 0
                columnHeights[j] -= 1
                if new_best < best:
                    best = new_best
                if best <= alpha:
                    return best
                if best < beta:
                    beta = best
        return new_best

def miniMax(the_board, heightMap, depth, isMax):
    score = 0
    if checkWinner(the_board) == 3: # tied
        return 0
    elif checkWinner(the_board) == player: # win
        score = 10
        return score
    elif checkWinner(the_board) == opponent: # lose
        score = -10
        return score
    if depth == 5:
        return 0

    if isMax:	
        best = -1000
        for j in range(WIDTH):
            if heightMap[j]<HEIGHT:
                heightMap[j] += 1
                the_board[HEIGHT-int(heightMap[j])][j] = player
                best = max(best, miniMax(the_board, heightMap, depth + 1, not isMax))
                the_board[HEIGHT-int(heightMap[j])][j] = 0
                heightMap[j] -= 1
        return best
    else:
        best = 1000
        for j in range(WIDTH):
            if heightMap[j]<HEIGHT:
                heightMap[j] += 1
                the_board[HEIGHT-int(heightMap[j])][j] = opponent
                best = min(best, miniMax(the_board, heightMap, depth + 1, not isMax))
                the_board[HEIGHT-int(heightMap[j])][j] = 0
                heightMap[j] -= 1
        return best

def findBestMovePruning(the_board):
    bestVal = -1000
    bestMove = -1
    global state_count
    state_count =0
    columnHeights = np.zeros((WIDTH))
    for j in range(WIDTH):
        for i in range(HEIGHT):
            if the_board[HEIGHT-1-i][j] != 0:
                columnHeights[j] += 1
    for j in range(WIDTH):
        if columnHeights[j]<HEIGHT:
            columnHeights[j] += 1
            the_board[HEIGHT-int(columnHeights[j])][j] = player
            moveVal = miniMaxPruning(the_board, columnHeights, 0, False, -1000, 1000, state_count)
            the_board[HEIGHT-int(columnHeights[j])][j] = 0
            columnHeights[j] -= 1
            if (moveVal > bestVal):				
                bestMove = j
                bestVal = moveVal
    #print("STATES VISITED: " +state_count)
    return bestMove



def findBestMove(the_board):
    bestVal = -1000
    bestMove = -1
    columnHeights = np.zeros((WIDTH))
    for j in range(WIDTH):
        for i in range(HEIGHT):
            if the_board[HEIGHT-1-i][j] != 0:
                columnHeights[j] += 1
    for j in range(WIDTH):
        if columnHeights[j]<HEIGHT:
            columnHeights[j] += 1
            the_board[HEIGHT-int(columnHeights[j])][j] = player
            moveVal = miniMax(the_board, columnHeights, 0, False)
            the_board[HEIGHT-int(columnHeights[j])][j] = 0
            columnHeights[j] -= 1
            if (moveVal > bestVal):				
                bestMove = j
                bestVal = moveVal
    return bestMove

# rewards
WIN_VALUE = 1.0
LOSS_VALUE = 0.0 
TIED_VALUE = 0.5

alpha=0.9 # learning rate
gamma=0.95 # discount
initialise=0.6 # inititialise q table

class PlayQlearning():
    def __init__(self, player):
        self.QTable = {}
        self.history = []
        self.player = 1
        self.opponent = 2
        if (player == 2):
            self.opponent = 1
            self.player = 2

    def indexBoard(self, the_board):
        index = ""
        for i in range(HEIGHT):
            for j in range(WIDTH):
                index += str(the_board[i][j])
        return index

    def getQValue(self, board_index):
        if board_index in self.QTable:
            value = self.QTable[board_index]
        else:
            value = initialise*np.ones((WIDTH))
            self.QTable[board_index] = value
        return value
    
    def getQTable(self):
        return self.QTable

    def checkPosAvaliable(self, y, the_board):
        if y<0 or y>=WIDTH:
            return False
        if the_board[0][y] != 0:
            return False
        return True

    def findBestMove(self, the_board):
        boardIndex = self.indexBoard(the_board)
        qValue = self.getQValue(boardIndex)
        while True:
            maxIndex = np.argmax(qValue)
            if self.checkPosAvaliable(maxIndex, the_board):
                break
            else:
                qValue[maxIndex] = -1.0
        self.history.append((boardIndex, maxIndex))
        return maxIndex

    def finalResult(self, winner):
        if winner == 3:
            final_value = TIED_VALUE
        elif winner == self.player:
            final_value = WIN_VALUE
        elif winner == self.opponent:
            final_value = LOSS_VALUE

        self.history.reverse()
        next_max = -1.0
        for h in self.history:
            qValue = self.getQValue(h[0])
            if next_max < 0: 
                qValue[h[1]] = final_value
            else:
                qValue[h[1]] = qValue[h[1]] * (1.0-alpha) + alpha * gamma * next_max

            next_max = qValue.max()
    
    def newGame(self):
        self.history = []

    def load_qtable(file_path='qtable.pkl'):
        with open(file_path, 'rb') as f:
            qtable = pickle.load(f)
        print(qtable)
        return qtable


def defaultPlayer(the_player, board):
    # check row win
    for i in range(HEIGHT):
        # left
        for j in range(1, WIDTH-2):
            if board[i][j]!=0 and board[i][j+1]==board[i][j] and board[i][j+2]==board[i][j] and board[i][j-1]==0:
                if board[i-1][j-1]!=0:
                    return j-1 # i, j-1
        # right
        for j in range(3):
            if board[i][j]!=0 and board[i][j+1]==board[i][j] and board[i][j+2]==board[i][j] and board[i][j+3]==0:
                if board[i-1][j+3]!=0:
                    return j+3 #i, j+3
    # check column win
    for j in range(WIDTH):
        for i in range(1, HEIGHT-2):
            if board[i][j]!=0 and board[i+1][j]==board[i][j] and board[i+2][j]==board[i][j] and board[i-1][j]==0:
                return j #i-1, j
    # check diagonal win left
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if i<HEIGHT-1 and j>0 and i-2>0 and j+2<WIDTH and board[i][j]!=0 and board[i-1][j+1]==board[i][j] and board[i-2][j+2]==board[i][j] and board[i+1][j-1]==0 and (i+1==HEIGHT-1 or board[i+2][j-1]!=0):
                return j-1 # i+1, j-1
            if i-3>0 and j+3<WIDTH and board[i][j]!=0 and board[i-1][j+1]==board[i][j] and board[i-2][j+2]==board[i][j] and board[i-3][j+3]==0 and board[i-2][j+3]!=0:
                return j+3 # i-3, j+3
    # check diagonal win right
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if i>0 and j>0 and i+2<HEIGHT and j+2<WIDTH and board[i][j]!=0 and board[i+1][j+1]==board[i][j] and board[i+2][j+2]==board[i][j] and board[i-1][j-1]==0 and board[i][j-1]!=0:
                return j-1 # i-1, j-1
            if i+3<HEIGHT and j+3<WIDTH and board[i][j]!=0 and board[i+1][j+1]==board[i][j] and board[i+2][j+2]==board[i][j] and board[i+3][j+3]==0 and (i+3==HEIGHT-1 or board[i+4][j+3]!=0):
                return j+3 # i+3, j+3

    choices = []
    for j in range(WIDTH):
        for i in range(HEIGHT):
            if board[i][j] == 0:
                choices.append(j)
                break
        
    final_choice = random.choice(choices)
    return final_choice
    