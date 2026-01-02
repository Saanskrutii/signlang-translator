import cv2, mediapipe as mp, pyttsx3
print("OpenCV:", cv2.__version__)
print("Mediapipe:", mp.__version__)
engine = pyttsx3.init(); engine.say("Text to speech is working"); engine.runAndWait()
print("TTS OK")
