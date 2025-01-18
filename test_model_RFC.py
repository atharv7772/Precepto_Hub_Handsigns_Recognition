import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# Load the trained Random Forest model
try:
    with open('./annotated_data/random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print("Error: 'random_forest_model.pkl' not found.")
    exit()

# Load the test data
try:
    data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))
    print("Test data loaded successfully.")
except FileNotFoundError:
    print("Error: 'hand_gesture_data.pkl' not found.")
    exit()

# Extract test data and labels
test_data = data_dict['data']
test_labels = data_dict['labels']

# Ensure consistent data lengths
data_lengths = [len(sample) for sample in test_data]
if len(set(data_lengths)) > 1:
    print(f"Inconsistent test data lengths detected: {set(data_lengths)}")
    max_length = max(data_lengths)
    print(f"Padding all test samples to the maximum length: {max_length}")

    # Pad or truncate data to ensure consistent shape
    padded_data = []
    for sample in test_data:
        if len(sample) < max_length:
            # Pad with zeros
            sample += [0] * (max_length - len(sample))
        elif len(sample) > max_length:
            # Truncate extra elements
            sample = sample[:max_length]
        padded_data.append(sample)
    
    test_data = np.array(padded_data)
else:
    test_data = np.array(test_data)

test_labels = np.array(test_labels)
print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

# Make predictions using the loaded model
predictions = model.predict(test_data)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(test_labels, predictions)
print(f"Random Forest Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Print predictions for the first 10 samples
print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", test_labels[:10])
