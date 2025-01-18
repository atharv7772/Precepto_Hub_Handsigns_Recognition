import os
import pickle
import mediapipe as mp
import cv2

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # Corrected drawing_utils

# Configure the Hands model
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Path to the dataset folder
data_dir = './dataset'

# Lists to store landmark data and corresponding labels
data = []
labels = []

# Directory to save annotated images
annotated_data_dir = './annotated_data'
if not os.path.exists(annotated_data_dir):
    os.makedirs(annotated_data_dir)

# Loop through each class (A-Z)
for dir_ in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, dir_)

    # Loop through each image in the class folder
    for img_path in os.listdir(label_dir):
        data_aux = []  # To store the normalized landmarks for one image
        x_ = []        # List to collect x-coordinates
        y_ = []        # List to collect y-coordinates

        img = cv2.imread(os.path.join(label_dir, img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                # Normalize landmarks by subtracting the minimum value
                min_x, max_x = min(x_), max(x_)
                min_y, max_y = min(y_), max(y_)

                for x, y in zip(x_, y_):
                    data_aux.append((x - min_x) / (max_x - min_x))  # Normalizing x coordinates
                    data_aux.append((y - min_y) / (max_y - min_y))  # Normalizing y coordinates

                # Optionally draw landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            data.append(data_aux)
            labels.append(dir_)

            # Save annotated image (optional)
            save_path = os.path.join(annotated_data_dir, f'{dir_}_{img_path}')
            cv2.imwrite(save_path, img)

# Save the processed data and labels to a pickle file
with open('hand_gesture_data.pkl', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data preprocessing completed and saved as 'hand_gesture_data.pkl'.")
