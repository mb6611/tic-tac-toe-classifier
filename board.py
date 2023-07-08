import numpy as np
import copy
import random

# Create a board

class Board:
  # initialization
  MAX_TILES = 9
  MAX_MOVES = 9

  def __init__(self):
    self.board = np.array([0] * 9)
    self.possible_moves = np.array([tile for tile in range(0, Board.MAX_TILES)])
    self.current_move = 1
    self.move_history = np.array([], dtype=int)
    self.win_state = 0
    self.parent_board = self

  # check win
  def check_win(self):
    MAX_LENGTH = 3  # max row/column length

    # boards to check
    row_board = np.reshape(self.board, (3,3)) # row board
    column_board = np.transpose(row_board) # column board

    # diagonals to check
    diagonal = np.diag(row_board) # standard board diagonal
    flipped_diagonal = np.diag(np.fliplr(row_board)) # left-right flipped board diagonal

    win_x = np.array([1,1,1]) # win conditions
    win_o = np.array([-1,-1,-1])
    result = False

    for i in range(0, MAX_LENGTH):
      row = row_board[i]
      column = column_board[i]

      if np.any([np.all(np.in1d(x, win_x)) for x in
                                    (row, column, diagonal, flipped_diagonal)]):
        result = True
        self.win_state = 1
      elif np.any([np.all(np.in1d(x, win_o)) for x in
                                    (row, column, diagonal, flipped_diagonal)]):
        result = True
        self.win_state = -1
    return result

  # check draw
  def check_draw(self):
    result = False
    if ((self.current_move > Board.MAX_MOVES) & (self.check_win() == False)):
      result = True
    return result

  # return board
  def return_board(self):
    return self.board

  # play one move
  def play(self, tile):
    if (self.current_move % 2 == 1): self.board[tile] = 1 # update board
                                                          # and move number
    else: self.board[tile] = -1
    self.current_move += 1

    for index in range(0, len(self.possible_moves)): # remove move
      if self.possible_moves[index] == tile:
        self.possible_moves = np.delete(self.possible_moves, index)
        break

    self.move_history = np.append(self.move_history, tile) # add move to history
    self.check_win()

  def play_sub_board(self, tile):

    sub_board = Board()
    sub_board.board = np.copy(self.board)
    sub_board.possible_moves = np.copy(self.possible_moves)
    sub_board.current_move = copy.copy(self.current_move)
    sub_board.move_history = np.copy(self.move_history)
    sub_board.win_state = copy.copy(self.win_state)

    sub_board.play(tile)
    return sub_board

  def flush_move(self, tile):

    flushed_board = Board() # copy board

    flushed_board.board = np.copy(self.board)
    flushed_board.board[tile] = 0

    flushed_board.possible_moves = np.copy(self.possible_moves)

    for possible_move in range(0, len(flushed_board.possible_moves)):

      if (flushed_board.possible_moves[possible_move] > tile) & (possible_move == 0): # flushed move is smallest tile
        flushed_board.possible_moves = np.insert(flushed_board.possible_moves,
                                                 0, [tile], axis=0)
        break

      elif (tile < flushed_board.possible_moves[possible_move]):
        flushed_board.possible_moves = np.insert(flushed_board.possible_moves,
                                                 possible_move, [tile], axis=0)
        break


    flushed_board.current_move = copy.copy(self.current_move) - 1

    flushed_board.move_history = np.delete(np.copy(self.move_history), -1)

    flushed_board.win_state = flushed_board.check_win()

    flushed_board.parent_board = self.parent_board

    return flushed_board, tile

  def flush_last_move(self):
    return self.flush_move(self.move_history[-1])

counter = 0
new_valid_games_counter = 0

class GameGenerator():

  # initialize a board
  def __init__ (self):
    self.board = Board()

  # generate some sub-boards
  def generate_some_sub_boards(self, board):

    global counter
    global new_valid_games_counter

    sub_boards = []
    if len(board.possible_moves) > 0: # generate sub_boards if moves exist

      possible_moves = []

      if len(board.possible_moves) > 5: # generate two sub_boards if possible
        possible_moves = random.sample(board.possible_moves.tolist(), 5)
        #print(possible_moves)
      else: # generate one sub_board otherwise
        possible_moves = copy.copy(board.possible_moves)
        #print(possible_moves)

      for move in possible_moves:
        sub_board = board.play_sub_board(move)
        sub_boards.append(sub_board) #####

    return sub_boards

  # generate a sub-boards
  def generate_sub_boards(self, board):

    global counter

    sub_boards = []
    if len(board.possible_moves) > 0:
      for move in board.possible_moves:
        sub_board = board.play_sub_board(move)
        sub_boards.append(sub_board) #####

    return sub_boards

  # generate valid games
  def generate_valid_games(self, board):

    global counter
    global new_valid_games_counter

    # win condition
    if board.check_win():
      #print("Returning win...")
      counter += 1
      print(counter)
      return [board]

    # draw condition
    elif board.check_draw():
      counter += 1
      print(counter)
      #print("Returning draw...")
      return [board]

    # another move is possible condition
    else:
      #print("Generating another board...")
      sub_boards = self.generate_some_sub_boards(board)
      valid_games = []
      for sub_board in sub_boards: # set the board to the sub-board and
                                   # generate moves for that board
          valid_games.extend(self.generate_valid_games(sub_board))
      #print(valid_games)
      return valid_games