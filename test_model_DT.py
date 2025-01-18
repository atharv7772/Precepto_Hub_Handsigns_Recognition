import pickle
import numpy as np
from sklearn.metrics import accuracy_score

# Load the trained Decision Tree model
with open('decision_tree_model.pkl', 'rb') as model_file:
    dt_model = pickle.load(model_file)

# Load the test data
data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))

# Extract dataset and labels
test_data = data_dict['data']
test_labels = data_dict['labels']

# Expected number of landmarks per image (21 landmarks with x, y coordinates)
expected_length = 42  # 21 landmarks * 2 (x and y)

# Function to pad the data
def pad_data(data_aux, expected_length):
    if len(data_aux) < expected_length:
        # Pad with zeros if the length is shorter than expected
        data_aux.extend([0] * (expected_length - len(data_aux)))
    return data_aux[:expected_length]  # Ensure it's not longer than expected

# Process the test data
processed_test_data = [pad_data(d, expected_length) for d in test_data]

# Convert to numpy arrays
processed_test_data = np.array(processed_test_data)
test_labels = np.array(test_labels)

# Make predictions using the trained model
predictions = dt_model.predict(processed_test_data)

# Calculate accuracy on the test set
test_accuracy = accuracy_score(test_labels, predictions)
print(f"Decision Tree Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Print predictions for the first 10 samples
print("First 10 predictions:", predictions[:10])
print("First 10 actual labels:", test_labels[:10])
