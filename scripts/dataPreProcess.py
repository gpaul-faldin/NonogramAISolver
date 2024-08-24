import numpy as np
import json
from sklearn.model_selection import train_test_split

def preprocess_5x5_data(data):
    X = []
    y = []
    
    for puzzle in data:
        combined = np.array(puzzle['combined']).reshape(10, 5)
        combined = combined / np.max(combined)
        X.append(combined)
        y.append(np.array(puzzle['solution']).reshape(5, 5))
    
    return np.array(X), np.array(y)

def preprocess_10x10_data(data):
    X = []
    y = []
    
    for puzzle in data:
        combined = np.array(puzzle['combined']).reshape(20, 10)
        combined = combined / np.max(combined)
        X.append(combined)
        y.append(np.array(puzzle['solution']).reshape(10, 10))
    
    return np.array(X), np.array(y)

def load_and_preprocess_data(size='5x5'):
    with open(f'data/{size}.json', 'r') as f:
        data = json.load(f)
    if size == '5x5':
        X, y = preprocess_5x5_data(data)
    elif size == '10x10':
        X, y = preprocess_10x10_data(data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_and_preprocess_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")