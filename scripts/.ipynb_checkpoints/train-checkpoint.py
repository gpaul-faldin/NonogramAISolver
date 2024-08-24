import tensorflow as tf
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from modelArchitecture import create_5x5_nonogram_cnn
from dataPreProcess import load_and_preprocess_data


def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU is available. Found {len(gpus)} GPU(s):")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("No GPU found. Training will proceed on CPU.")
    return bool(gpus)

def test_model_on_puzzle(model, puzzle_data):
    # Preprocess the puzzle data
    combined = np.array(puzzle_data['combined']).reshape(10, 5)
    combined = combined / np.max(combined)
    X = combined[np.newaxis, :, :, np.newaxis]
    
    # Make prediction
    prediction = model.predict(X)[0]
    prediction = (prediction > 0.5).astype(int)
    
    # Get actual solution
    solution = np.array(puzzle_data['solution']).reshape(5, 5)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(prediction, cmap='binary')
    ax1.set_title('Model Prediction')
    
    ax2.imshow(solution, cmap='binary')
    ax2.set_title('Actual Solution')
    
    plt.show()
    
    accuracy = np.mean(prediction == solution)
    print(f"Puzzle solving accuracy: {accuracy:.2f}")

def pick_random_puzzle():
    with open(f'data/5x5.json', 'r') as file:
        random.seed(time.time())
        file_data = json.load(file)
        puzzle = random.choice(file_data)
        return puzzle

def main():
    # Check GPU availability
    gpu_available = check_gpu()
    print(f"GPU available: {gpu_available}")

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data()

    # Create and compile the model
    model = create_5x5_nonogram_cnn()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
        history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=100,
                            validation_data=(X_val, y_val),
                            verbose=1)

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    test_model_on_puzzle(model, pick_random_puzzle())

    # Optionally, save the model
    model.save('5x5_nonogram_model.keras')

if __name__ == "__main__":
    main()