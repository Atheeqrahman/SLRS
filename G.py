import google.generativeai as genai
import pyttsx3

# Set up Google Gemini API key
genai.configure(api_key="AIzaSyD1bWsTKwHwUNhvyFzI5tdM-InXf7FGJJ8")

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


def speak_text(text):
    engine.say(text)
    engine.runAndWait()


def main():
    wake_call = "Gopal"

    # Listen for the wake call
    while True:
        user_input = input("You: ")

        if wake_call.lower() in user_input.lower():
            print("Gopal: Hey there How can I help you today?")
            speak_text("Hey There How can I help you today?")
            break
        else:
            print("Gopal: Waiting for wake call...")

    # Conversation loop
    while True:
        user_input = input("You: ")

        if "that's all" in user_input.lower():
            print("Gopal: Bye Bye! Have a great day")
            speak_text("Bye Bye! Have a great day")
            break

        elif any(word in user_input.lower() for word in ["what is your name", "who are you"]):
            ai_response = "Hi, I'm Gopal the bot......  Speed 1 terahertz,      memory 1 zigabyte."
            print("Gopal: Hi, I'm Gopal the bot. Speed 1 terahertz, memory 1 zigabyte.")
            speak_text(ai_response)

        else:
            print("Gopal:", end=' ')
            response = model.generate_content(user_input)
            response_text = response.text  # Extract text content from response object
            print(response_text)  # Print the extracted text content
            speak_text(response_text)


if __name__ == "__main__":
    main()
