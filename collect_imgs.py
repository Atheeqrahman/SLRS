import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 6
dataset_size = 100

# Attempt to open the camera capture
cap = cv2.VideoCapture(0)  # Use index 0 for default camera

# Check if the camera capture was successful
if not cap.isOpened():
    print("Error: Failed to open camera capture")
    exit()

# Loop over each class
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Prompt the user to prepare for data collection
    print('Press "Q" when ready...')
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect data for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        
        # Check if frame capture was successful
        if not ret:
            print("Error: Failed to capture frame")
            break

        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)

        counter += 1

# Release the camera capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
