import tensorflow as tf
import numpy as np
import json
import time
import random
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modelArchitecture import create_5x5_nonogram_cnn, create_10x10_nonogram_cnn
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

def test_model_on_puzzle(model, puzzle_data, size='5x5'):
    if size == '5x5':
        # Preprocess the 5x5 puzzle data
        combined = np.array(puzzle_data['combined']).reshape(10, 5)  # Adjust if needed based on actual data format
        combined = combined / np.max(combined)
        X = combined[np.newaxis, :, :, np.newaxis]
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction = (prediction > 0.5).astype(int)
        
        # Get actual solution
        solution = np.array(puzzle_data['solution']).reshape(5, 5)
    
    elif size == '10x10':
        # Preprocess the 10x10 puzzle data
        combined = np.array(puzzle_data['combined']).reshape(20, 10)  # Adjust if needed based on actual data format
        combined = combined / np.max(combined)
        X = combined[np.newaxis, :, :, np.newaxis]
        
        # Make prediction
        prediction = model.predict(X)[0]
        prediction = (prediction > 0.5).astype(int)
        
        # Get actual solution
        solution = np.array(puzzle_data['solution']).reshape(10, 10)
    
    else:
        raise ValueError("Unsupported nonogram size. Please use '5x5' or '10x10'.")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(prediction, cmap='binary')
    ax1.set_title(f'Model Prediction ({size})')
    
    ax2.imshow(solution, cmap='binary')
    ax2.set_title(f'Actual Solution ({size})')
    
    plt.show()
    
    # Compute and print accuracy
    accuracy = np.mean(prediction == solution)
    print(f"Puzzle solving accuracy: {accuracy:.2f}")

def pick_random_puzzle(size='5x5'):
    with open(f'data/{size}.json', 'r') as file:
        random.seed(time.time())
        file_data = json.load(file)
        puzzle = random.choice(file_data)
        return puzzle

def plot_learning_curves(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def main():
    # Check GPU availability
    gpu_available = check_gpu()
    size = "10x10"
    print(f"GPU available: {gpu_available}")

    # Load and preprocess data
    X_train, X_val, y_train, y_val = load_and_preprocess_data(size)

    # Create and compile the model
    if size == '5x5':
        model = create_5x5_nonogram_cnn()
    elif size == '10x10':
        model = create_10x10_nonogram_cnn()
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{size}_best_model.keras', monitor='val_accuracy', save_best_only=True)

    # Train the model
    with tf.device('/GPU:0' if gpu_available else '/CPU:0'):
        history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=3000,
                            validation_data=(X_val, y_val),
                            verbose=1,
                            callbacks=[early_stopping, model_checkpoint])

    # Load the best model (if checkpointing)
    model = tf.keras.models.load_model(f'{size}_best_model.keras')
    plot_learning_curves(history)

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    test_model_on_puzzle(model, pick_random_puzzle(size), size)

    # Optionally, save the model
    model.save(f'{size}_nonogram_model.keras')

if __name__ == "__main__":
    main()