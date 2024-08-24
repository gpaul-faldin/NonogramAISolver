from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Dropout
from tensorflow.keras import regularizers

def create_10x10_nonogram_cnn():
    inputs = Input(shape=(20, 10, 1))
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(100, activation='sigmoid')(x)

    
    outputs = Reshape((10, 10))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


def create_5x5_nonogram_cnn():
    inputs = Input(shape=(10, 5, 1))  # 10 rows of tips (5 for rows, 5 for columns), 5 columns (max 2 tips per row/column)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(25, activation='sigmoid')(x)  # 5x5 = 25 output neurons
    
    outputs = Reshape((5, 5))(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model


