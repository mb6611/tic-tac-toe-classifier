import pandas as pd
import numpy as np
import os
from board import Board, GameGenerator

# reaed .pkl files

current_directory = os.getcwd()+'/tic_tac_toe_classifier/'
data_features = (pd.read_pickle(current_directory + 'features.pkl')[0].to_numpy()).tolist()

data_labels = pd.read_pickle(current_directory + 'labels.pkl').to_numpy()
data_labels = np.reshape(data_labels, (len(data_labels),))

# build data features from DataFrames
data_features_current_move = []

for element in data_features:
  data_features_current_move.append(element.current_move)

draw_data_features = []
draw_data_next_move = []

for element in range(0, len(data_features)):
  if data_features[element].parent_board.win_state == 0:
    draw_data_features.append(data_features[element])
    draw_data_next_move.append(data_labels[element])

draw_move_history_data = []
draw_current_move_data = []
draw_win_state_data = []

for element in draw_data_features:
  draw_move_history_data.append(element.move_history)
  draw_current_move_data.append(element.current_move)
  draw_win_state_data.append(element.win_state)

# pad move_history with zeros

result = []
for element in draw_move_history_data:
  result.append(np.pad(element, (0, 10 - len(element))))

draw_move_history_data = result