import tensorflow as tf
import numpy as np
import pandas as pd
from board import Board, GameGenerator
from load_data import draw_move_history_data, draw_current_move_data, draw_data_next_move
from keras.optimizers import SGD

# define NN topology
input = tf.keras.layers.Input(shape=(10,), dtype=tf.float32, name='move_history')
input2 = tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='current_move')
embedded = tf.keras.layers.Embedding(input_dim=10, output_dim=2, input_length=None)(input)
embedded2 = tf.keras.layers.Embedding(input_dim=10, output_dim=2, input_length=None)(input2)
combined = tf.keras.layers.Concatenate(axis=1)([embedded, embedded2])
flattened = tf.keras.layers.Flatten()(combined)
dense = tf.keras.layers.Dense(18, activation='relu')
x = dense(flattened)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(18, activation='relu')(x)
outputs = {
    'one': tf.keras.layers.Dense(1, name='one', activation='sigmoid')(x),
    'two': tf.keras.layers.Dense(1, name='two', activation='sigmoid')(x),
    'three': tf.keras.layers.Dense(1, name='three', activation='sigmoid')(x),
    'four': tf.keras.layers.Dense(1, name='four', activation='sigmoid')(x),
    'five': tf.keras.layers.Dense(1, name='five', activation='sigmoid')(x),
    'six': tf.keras.layers.Dense(1, name='six', activation='sigmoid')(x),
    'seven': tf.keras.layers.Dense(1, name='seven', activation='sigmoid')(x),
    'eight': tf.keras.layers.Dense(1, name='eight', activation='sigmoid')(x),
    'nine': tf.keras.layers.Dense(1, name='nine', activation='sigmoid')(x),
}

# build model
model = tf.keras.Model(inputs=[input,input2], outputs=outputs, name='tic-tac-toe-model')
model.summary()

#tf.keras.utils.plot_model(model, 'tic-tac-toe-model.png', show_shapes=True)


# prepare training dataset for model input
data = {'move_history': draw_move_history_data,
        'current_move': draw_current_move_data
}

train_df = pd.DataFrame(data)

labels = {"next_move": draw_data_next_move}
train_labels_df = pd.DataFrame(labels)



current_move_data = train_df['current_move'].to_numpy()
draw_move_history_data = np.reshape(draw_move_history_data, (46860, 10))
current_move_data = np.reshape(current_move_data, (len(current_move_data), 1))

array = np.reshape(draw_move_history_data[0], (1, 10))
array2 = np.reshape(current_move_data[0], (1,))
print(model.predict([draw_move_history_data, current_move_data]))





# compile model
opt = SGD(learning_rate=0.00001)

model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)


inputs = [draw_move_history_data, current_move_data]

history = model.fit(
    [draw_move_history_data, current_move_data],
    train_labels_df.values,
    batch_size=10,
    epochs=20,
    validation_split=0.2)

epochs = history.epoch
hist = pd.DataFrame(history.history)

print(hist)
#loss = hist["val_accuracy"]
