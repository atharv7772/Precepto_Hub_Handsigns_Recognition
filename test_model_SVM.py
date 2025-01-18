import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# Load the trained SVM model
try:
    with open('./svm_model.pkl', 'rb') as model_file:
        svm_model = pickle.load(model_file)
    print("SVM model loaded successfully.")
except FileNotFoundError:
    print("Error: 'svm_model.pkl' not found.")
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

# Expected number of landmarks per image (21 landmarks with x, y coordinates)
expected_length = 42  # 21 landmarks * 2 (x and y)

# Function to pad or truncate data
def pad_or_truncate(sample, expected_length):
    if len(sample) < expected_length:
        return sample + [0] * (expected_length - len(sample))  # Pad with zeros
    return sample[:expected_length]  # Truncate extra elements

# Ensure all test samples have the expected length
processed_test_data = [pad_or_truncate(sample, expected_length) for sample in test_data]

# Convert to numpy arrays
test_data = np.array(processed_test_data)
test_labels = np.array(test_labels)

print(f"Test data shape: {test_data.shape}, Test labels shape: {test_labels.shape}")

# Make predictions using the SVM model
predictions = svm_model.predict(test_data)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(test_labels, predictions)
print(f"SVM Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Display some sample predictions
print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", test_labels[:10])
