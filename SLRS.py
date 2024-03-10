import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import google.generativeai as genai
import pyttsx3
import cv2

# Set up Google Gemini API key
genai.configure(api_key="YOUR_API_KEY")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 120)  # speaking rate
voices = engine.getProperty('voices')
# Selecting a male voice
engine.setProperty('voice', voices[0].id)  # 0 for male; 1 for female
# Adjusting pitch and volume for a bit of bass
engine.setProperty('pitch', 50)  # Adjust pitch (50 is the default)
engine.setProperty('volume', 0.9)  # Adjust volume (0 to 1, 1 is the default)

# Initialize Google Gemini model
model = genai.GenerativeModel('gemini-pro')

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model_hand = model_dict['model']

# Open the camera capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define labels dictionary
labels_dict = {0: 'Gopal', 1: 'B', 2: 'HI', 3: 'D', 4: ' ', 5: 'What is the wether out there'}

# Variable to store the detected word
detected_word = ""

# Variable to keep track of last detection time
last_detection_time = time.time()

# Variable to keep track of last word detection time
last_word_detection_time = time.time()

# Variable to keep track of last blank screen time
last_blank_screen_time = time.time()

# Time to wait for a blank screen before resetting detected_word (in seconds)
blank_screen_timeout = 5


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    global detected_word, last_detection_time, last_word_detection_time, last_blank_screen_time

    wake_call = "Gopal"

    # Listen for the wake call
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

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

            prediction = model_hand.predict([np.asarray(data_aux)])
            detected_character = labels_dict[int(prediction[0])]

            if time.time() - last_detection_time >= 2:
                detected_word += detected_character
                last_detection_time = time.time()
                last_word_detection_time = time.time()

        else:
            if time.time() - last_detection_time >= blank_screen_timeout:
                detected_word = ""
                last_blank_screen_time = time.time()

        if time.time() - last_word_detection_time >= 5:
            print(detected_word)
            speak_text(detected_word)
            last_word_detection_time = time.time()

        cv2.putText(frame, detected_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if detected_word.lower().endswith(wake_call.lower()):
            print("Gopal: Hey there! How can I help you today?")
            speak_text("Hey there! How can I help you today?")
            detected_word = ""
            break

    # Conversation loop
    while True:
        user_input = detected_word

        if "that's all" in user_input.lower():
            print("Gopal: Bye Bye! Have a great day")
            speak_text("Bye Bye! Have a great day")
            break

        elif any(word in user_input.lower() for word in ["what is your name", "who are you"]):
            ai_response = "Hi, I'm Gopal the bot. Speed 1 terahertz, memory 1 zigabyte."
            print("Gopal:", ai_response)
            speak_text(ai_response)

        else:
            print("Gopal:", end=' ')
            if user_input.strip():  # Check if user_input is not empty or whitespace
                response = model.generate_content(user_input)
                response_text = response.text
                print(response_text)
                speak_text(response_text)
            else:
                print("Gopal: Sorry, I didn't catch that. Could you please repeat?")


if __name__ == "__main__":
    main()

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
