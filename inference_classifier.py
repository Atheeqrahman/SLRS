# Idea is to give a sign for space and it will be addded

import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Open the camera capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {0: 'G', 1: 'A', 2: 'B', 3: 'C', 4: ' ', 5: 'E'}

# Variable to store the detected word
detected_word = ""

# Variable to keep track of last detection time
last_detection_time = time.time()

# Variable to keep track of last word detection time
last_word_detection_time = time.time()

while True:
    # Capture frame from the camera
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Error: Failed to capture frame.")
        break

    H, W, _ = frame.shape

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe Hands model
    results = hands.process(frame_rgb)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Extract hand landmarks and normalize coordinates
        data_aux = []
        x_ = []
        y_ = []
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        # Make prediction using the model
        prediction = model.predict([np.asarray(data_aux)])
        
        # Convert prediction to character label
        detected_character = labels_dict[int(prediction[0])]
        
        # Check if 2 seconds have passed since last detection
        if time.time() - last_detection_time >= 2:
            # Update the detected word with the newly detected character
            detected_word += detected_character
            # Update the last detection time
            last_detection_time = time.time()
            # Update the last word detection time
            last_word_detection_time = time.time()
    
    # Check if 5 seconds have passed since last word detection
    if time.time() - last_word_detection_time >= 5:
        # Print the final detected word
        print(detected_word)
        # Update the last word detection time
        last_word_detection_time = time.time()

    # Display detected word on the frame
    cv2.putText(frame, detected_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display annotated frame
    cv2.imshow('frame', frame)
    
    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
