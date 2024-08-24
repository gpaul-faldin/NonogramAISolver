import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import time
import random
import matplotlib.pyplot as plt

# Path to the trained Keras model file
model_path = '5x5_nonogram_model.keras'

def pick_random_puzzle():
    """Selects a random puzzle from a JSON file."""
    with open(f'data/5x5.json', 'r') as file:
        random.seed(time.time())
        file_data = json.load(file)
        puzzle = random.choice(file_data)
        return puzzle

def test_model_on_puzzle(model, puzzle_data):
    """Tests the model on a given puzzle and visualizes the result."""
    
    # Preprocess the puzzle data
    combined = np.array(puzzle_data['combined']).reshape(10, 5)
    combined = combined / np.max(combined)  # Normalizing data
    X = combined[np.newaxis, :, :, np.newaxis]  # Shape: (1, 10, 5, 1)
    
    # Make prediction using the model
    prediction = model.predict(X)[0]  # Get the first (and only) batch
    prediction = (prediction > 0.5).astype(int)  # Convert to binary (0 or 1)
    
    # Get actual solution from puzzle data
    solution = np.array(puzzle_data['solution']).reshape(5, 5)
    
    # Visualize the model's prediction and actual solution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(prediction, cmap='binary', vmin=0, vmax=1)
    ax1.set_title('Model Prediction')
    ax1.grid(True)
    ax1.set_xticks(np.arange(-0.5, 5, 1))
    ax1.set_yticks(np.arange(-0.5, 5, 1))
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.tick_params(length=0)
    
    ax2.imshow(solution, cmap='binary', vmin=0, vmax=1)
    ax2.set_title('Actual Solution')
    ax2.grid(True)
    ax2.set_xticks(np.arange(-0.5, 5, 1))
    ax2.set_yticks(np.arange(-0.5, 5, 1))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.tick_params(length=0)
    
    plt.show()
    
    # Calculate and print the accuracy
    accuracy = np.mean(prediction == solution)
    print(f"Puzzle solving accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    # Load the Keras model
    model = load_model(model_path)
    
    # Pick a random puzzle
    puzzle = pick_random_puzzle()
    
    # Test the model on the puzzle and visualize the results
    test_model_on_puzzle(model, puzzle)
