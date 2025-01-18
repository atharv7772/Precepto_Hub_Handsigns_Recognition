import os
import cv2

# Define the base directory to store data
data_dir = os.path.join(os.getcwd(), "dataset")

# Number of classes (26 letters: A-Z)
classes = 26
# Number of images to be captured for each alphabet
class_images = 500

# Initiating webcam, 0 is for the default webcam
cap = cv2.VideoCapture(0)

# Loop through all classes (A-Z)
for j in range(classes):
    # Check if a directory exists for the current class, if not, create one
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))
    
    print(f'Collecting data for class {chr(65 + j)}')

    # Loop to capture the images for each class
    count = 0
    while count < class_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        cv2.putText(frame, f'Capturing images for class {chr(65 + j)}: {count}/{class_images}', 
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Capturing', frame)

        # Save the image when the Enter key is pressed
        if cv2.waitKey(1) & 0xFF == 13:  # Enter key
            img_name = os.path.join(data_dir, str(j), f"{chr(65 + j)}_{count}.png")
            cv2.imwrite(img_name, frame)
            count += 1

    print(f"Data collection for class {chr(65 + j)} completed.")

cap.release()
cv2.destroyAllWindows()
