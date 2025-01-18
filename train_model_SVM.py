import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the data from the pickle file
try:
    data_dict = pickle.load(open('./hand_gesture_data.pkl', 'rb'))
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'hand_gesture_data.pkl' not found.")
    exit()

# Extract data and labels
data = data_dict['data']
labels = data_dict['labels']

# Expected number of landmarks per image (21 landmarks with x, y coordinates)
expected_length = 42  # 21 landmarks * 2 (x and y)

# Function to pad or truncate data
def pad_or_truncate(sample, expected_length):
    if len(sample) < expected_length:
        return sample + [0] * (expected_length - len(sample))  # Pad with zeros
    return sample[:expected_length]  # Truncate extra elements

# Ensure all data samples have the expected length
processed_data = [pad_or_truncate(sample, expected_length) for sample in data]

# Convert to numpy arrays
data = np.array(processed_data)
labels = np.array(labels)

print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
print("Data split into training and testing sets.")

# Initialize the SVM model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the SVM model
print("Training SVM model...")
svm_model.fit(x_train, y_train)
print("SVM model trained successfully.")

# Make predictions
y_predict_svm = svm_model.predict(x_test)

# Calculate and display accuracy
accuracy_svm = accuracy_score(y_test, y_predict_svm)
print(f"SVM Model Accuracy: {accuracy_svm * 100:.2f}%")

# Save the trained model for future use
with open('./svm_model.pkl', 'wb') as model_file:
    pickle.dump(svm_model, model_file)
print("SVM model saved as 'svm_model.pkl'.")
