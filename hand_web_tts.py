# src/hand_web_tts.py
import streamlit as st
import cv2
import mediapipe as mp
import pyttsx3
import tempfile

st.title("Hand Detection Web App")

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# MediaPipe hand detector
hands = mp.solutions.hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
draw = mp.solutions.drawing_utils

# Upload video (optional) or use webcam
st.write("Press Start to use your webcam for hand detection.")
start = st.button("Start Camera")

if start:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Streamlit frame for video

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access camera")
            break

        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            engine.say("Hand detected")
            engine.runAndWait()

        # Convert frame to RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

        # Break if user closes browser or stops script (Streamlit doesn't have a simple 'q' key)
