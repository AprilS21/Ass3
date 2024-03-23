import numpy as np
import random

class Connect4:
    def __init__(self):
        self.board = np.zeros((6, 7), dtype=int)  # 6x7 board
        self.player = 1  # Player 1 starts
        self.moves = [0, 1, 2, 3, 4, 5, 6]  # Possible moves
        
    def reset(self):
        self.board = np.zeros((6, 7), dtype=int)
        self.player = 1

    def available_moves(self):
        return [col for col in range(7) if self.board[0][col] == 0]

    def make_move(self, col):
        for row in range(5, -1, -1):
            if self.board[row][col] == 0:
                self.board[row][col] = self.player
                break
        self.player = 3 - self.player  # Switch player

    def check_win(self, player):
        # Check rows
        for row in range(6):
            for col in range(4):
                if np.all(self.board[row, col:col+4] == player):
                    return True
        # Check columns
        for col in range(7):
            for row in range(3):
                if np.all(self.board[row:row+4, col] == player):
                    return True
        # Check diagonals
        for row in range(3):
            for col in range(4):
                if np.all(self.board[row:row+4, col:col+4].diagonal() == player):
                    return True
                if np.all(np.fliplr(self.board[row:row+4, col:col+4]).diagonal() == player):
                    return True
        return False

    def print_board(self):
        for row in range(6):
            print("|", end="")
            for col in range(7):
                if self.board[row][col] == 0:
                    print(" ", end="|")
                elif self.board[row][col] == 1:
                    print("X", end="|")
                else:
                    print("O", end="|")
            print()
        print("---------------")

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=1.0):
        self.epsilon = epsilon  # Exploration rate
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.q_table = dict()

    def get_Q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update(self, state, action, reward, next_state):
        old_value = self.get_Q_value(state, action)
        best_next_action = max([self.get_Q_value(next_state, a) for a in range(7)])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * best_next_action)
        self.q_table[(state, action)] = new_value

    def choose_action(self, state, available_moves):
        if random.random() < self.epsilon:
            return random.choice(available_moves)
        else:
            max_q = max([self.get_Q_value(state, a) for a in available_moves])
            best_moves = [a for a in available_moves if self.get_Q_value(state, a) == max_q]
            return random.choice(best_moves)

class DefaultComputerPlayer:
    def choose_action(self, state):
        return random.choice([col for col in range(7) if state[0][col] == 0])

class MiniMaxPlayer:
    def __init__(self, depth=3):
        self.depth = depth

    def choose_move(self, state):
        best_score = float('-inf')
        best_move = None
        for col in range(7):
            if state[0][col] == 0:
                row = self.get_next_open_row(state, col)
                if row is not None:
                    new_state = state.copy()
                    new_state[row][col] = 2  # Assuming it's the maximizing player's turn
                    score = self.minimax(new_state, self.depth, False)
                    if score > best_score:
                        best_score = score
                        best_move = col
        return best_move

    def minimax(self, state, depth, is_maximizing):
        if depth == 0 or self.game_over(state):
            return self.evaluate(state)

        if is_maximizing:
            best_score = float('-inf')
            for col in range(7):
                if state[0][col] == 0:
                    row = self.get_next_open_row(state, col)
                    if row is not None:
                        new_state = state.copy()
                        new_state[row][col] = 2
                        score = self.minimax(new_state, depth - 1, False)
                        best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for col in range(7):
                if state[0][col] == 0:
                    row = self.get_next_open_row(state, col)
                    if row is not None:
                        new_state = state.copy()
                        new_state[row][col] = 1
                        score = self.minimax(new_state, depth - 1, True)
                        best_score = min(score, best_score)
            return best_score

    def evaluate(self, state):
        # Simple evaluation function: count the number of possible winning moves for each player
        score = 0
        for col in range(7):
            for row in range(6):
                if state[row][col] == 0:
                    # Check horizontally
                    if col < 4:
                        if state[row][col + 1] == state[row][col + 2] == state[row][col + 3] == 2:
                            score += 1
                    # Check vertically
                    if row < 3:
                        if state[row + 1][col] == state[row + 2][col] == state[row + 3][col] == 2:
                            score += 1
                    # Check diagonally /
                    if row < 3 and col < 4:
                        if state[row + 1][col + 1] == state[row + 2][col + 2] == state[row + 3][col + 3] == 2:
                            score += 1
                    # Check diagonally \
                    if row < 3 and col > 2:
                        if state[row + 1][col - 1] == state[row + 2][col - 2] == state[row + 3][col - 3] == 2:
                            score += 1
        return score

    def get_next_open_row(self, state, col):
        for r in range(5, -1, -1):
            if state[r][col] == 0:
                return r
        return None

    def game_over(self, state):
        return any(self.check_win(state, player) for player in [1, 2])

    def check_win(self, state, player):
        # Check rows
        for row in range(6):
            for col in range(4):
                if np.all(state[row, col:col + 4] == player):
                    return True
        # Check columns
        for col in range(7):
            for row in range(3):
                if np.all(state[row:row + 4, col] == player):
                    return True
        # Check diagonals
        for row in range(3):
            for col in range(4):
                if np.all(state[row:row + 4, col:col + 4].diagonal() == player):
                    return True
                if np.all(np.fliplr(state[row:row + 4, col:col + 4]).diagonal() == player):
                    return True
        return False


def state_to_string(state):
    return ''.join([str(int(cell)) for row in state for cell in row])

def play_game(player1, player2, print_board=False):
    env = Connect4()
    state = env.board.copy()
    state_str = state_to_string(state)
    done = False

    while not done:
        if print_board:
            env.print_board()
        if env.player == 1:
            action = player1.choose_action(state_str, env.available_moves())
        else:
            if player2 == 'user':
                action = int(input("Your move (0-6): "))
            else:
                action = player2.choose_action(state)
        env.make_move(action)
        next_state = env.board.copy()
        next_state_str = state_to_string(next_state)

        if env.check_win(1):
            if print_board:
                env.print_board()
                print("Player 1 wins!")
            done = True
            return 1
        elif env.check_win(2):
            if print_board:
                env.print_board()
                print("Player 2 wins!")
            done = True
            return 2
        elif len(env.available_moves()) == 0:
            if print_board:
                env.print_board()
                print("It's a draw!")
            done = True
            return 0

if __name__ == "__main__":
    agent = QLearningAgent()
    computer = DefaultComputerPlayer()
    for _ in range(10):
        play_game(agent, computer)
    print("Finished training")
    print("Do you want to play against the trained agent? (yes/no):")
    choice = input()
    if choice.lower() == 'yes':
        play_game(agent, 'user', print_board=True)  # Play a game against the trained agent
    else:
        play_game(agent, computer, print_board=True)  # Play a game against the default computer player
