# src/streamlit_hand_tts.py

import cv2
import mediapipe as mp
import streamlit as st
import threading
import os
import numpy as np

# Streamlit page config
st.set_page_config(page_title="Hand Detector with TTS", layout="wide")
st.title("Hand Detector with Speech Feedback")

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_draw = mp.solutions.drawing_utils

# Streamlit placeholder for video
frame_placeholder = st.empty()

# TTS function using macOS 'say' (non-blocking)
def speak(text):
    os.system(f'say "{text}" &')

# Track if hand was already detected
hand_detected = False

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to capture camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if not hand_detected:
                hand_detected = True
                threading.Thread(target=speak, args=("Hand detected",), daemon=True).start()
        else:
            hand_detected = False

        # Convert BGR to RGB for Streamlit
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
finally:
    cap.release()
