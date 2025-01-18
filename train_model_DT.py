import pickle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data from the pickle file
data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))

# Extract dataset and labels
data = data_dict['data']
labels = data_dict['labels']

# Expected number of landmarks per image (21 landmarks with x, y coordinates)
expected_length = 42  # 21 landmarks * 2 (x and y)

# Function to pad the data
def pad_data(data_aux, expected_length):
    if len(data_aux) < expected_length:
        # Pad with zeros if the length is shorter than expected
        data_aux.extend([0] * (expected_length - len(data_aux)))
    return data_aux[:expected_length]  # Ensure it's not longer than expected

# Process the data to ensure each entry has the expected length
processed_data = [pad_data(d, expected_length) for d in data]

# Convert to numpy arrays
data = np.array(processed_data)
labels = np.array(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(x_train, y_train)

# Make predictions
y_predict_dt = dt_model.predict(x_test)

# Calculate accuracy
accuracy_dt = accuracy_score(y_test, y_predict_dt)
print(f"Decision Tree Model Accuracy: {accuracy_dt * 100:.2f}%")

# Save the model
with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)

print("Decision Tree model saved as 'decision_tree_model.pkl'.")
