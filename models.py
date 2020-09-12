import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def LSTM(sequence_length, embedding_layer):
    return keras.Sequential(
        [
            keras.Input(shape=(sequence_length,)),
            embedding_layer,
            layers.SpatialDropout1D(0.3),
            layers.LSTM(100, return_sequences=True),
            layers.GlobalMaxPool1D(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="sigmoid")
        ]
    )

def GRU(sequence_length, embedding_layer):
    return keras.Sequential(
        [
            keras.Input(shape=(sequence_length,)),
            embedding_layer,
            layers.SpatialDropout1D(0.3),
            layers.GRU(100, return_sequences=True),
            layers.GlobalMaxPool1D(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="sigmoid")
        ]
    )
    
def BLSTM(sequence_length, embedding_layer):
    return keras.Sequential(
        [
            keras.Input(shape=(sequence_length,)),
            embedding_layer,
            layers.SpatialDropout1D(0.3),
            layers.Bidirectional(layers.LSTM(100, return_sequences=True)),
            layers.GlobalMaxPool1D(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="sigmoid")
        ]
    )
    
def BGRU(sequence_length, embedding_layer):
    return keras.Sequential(
        [
            keras.Input(shape=(sequence_length,)),
            embedding_layer,
            layers.SpatialDropout1D(0.3),
            layers.Bidirectional(layers.GRU(100, return_sequences=True)),
            layers.GlobalMaxPool1D(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="sigmoid")
        ]
    )

def BGRU_CNN(sequence_length, embedding_layer):
    return keras.Sequential(
        [
            keras.Input(shape=(sequence_length,)),
            embedding_layer,
            layers.SpatialDropout1D(0.3),
            layers.Bidirectional(layers.GRU(100, return_sequences=True)),
            layers.Conv1D(100, 3, activation="relu"),
            layers.GlobalMaxPool1D(),
            layers.Dense(50, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(6, activation="sigmoid")
        ]
    )