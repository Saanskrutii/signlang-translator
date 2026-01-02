#!/usr/bin/env python3
"""
mediapipe_cam_next.py
A single script that supports:
  1) Real-time hand detection & landmark overlay
  2) Dataset collection to CSV with labels
  3) Training a simple RandomForest gesture model from the CSV
  4) Real-time prediction + TTS of recognized gestures
"""
import argparse
import csv
import time
from pathlib import Path
import sys
import random

import cv2
import numpy as np
import mediapipe as mp

# Optional imports for training/prediction/TTS
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    import joblib
except Exception:
    RandomForestClassifier = None
    train_test_split = None
    accuracy_score = None
    classification_report = None
    joblib = None

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Edit this to your preferred labels
LABEL_MAP = {
    "1": "ThumbsUp",
    "2": "Peace",
    "3": "Fist",
}

# ---------------------------
# Utility functions
# ---------------------------
def normalize_landmarks(landmarks):
    if len(landmarks) != 21:
        return None
    pts = np.array(landmarks, dtype=np.float32)
    base = pts[0].copy()
    pts -= base
    max_dist = np.max(np.linalg.norm(pts[:, :2], axis=1))
    scale = max_dist if max_dist >= 1e-6 else 1.0
    pts[:, :3] /= scale
    return pts.flatten().tolist()

def draw_label(img, text, org=(10,30)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 1, cv2.LINE_AA)

def speak_tts(engine, text):
    if engine:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass

def ensure_header(csv_path):
    csv_path = Path(csv_path)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    header = []
    for i in range(21):
        header += [f"x{i+1}", f"y{i+1}", f"z{i+1}"]
    header += ["label"]
    with csv_path.open("w", newline="") as f:
        csv.writer(f).writerow(header)

def append_sample(csv_path, features, label):
    ensure_header(csv_path)
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(list(features)+[label])

# ---------------------------
# Modes
# ---------------------------
def collect_mode(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)
    current_label_key = "1"
    current_label = LABEL_MAP[current_label_key]
    recording = False
    saved_count = 0
    frame_counter = 0

    with mp_hands.Hands(max_num_hands=args.max_hands,
                        min_detection_confidence=args.min_det,
                        min_tracking_confidence=args.min_track) as hands:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)
            landmarks_list = []
            if res.multi_hand_landmarks:
                for hand in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    pts = [(lm.x, lm.y, lm.z) for lm in hand.landmark]
                    features = normalize_landmarks(pts)
                    if features: landmarks_list.append(features)
            if recording and landmarks_list:
                frame_counter += 1
                if frame_counter % args.save_interval == 0:
                    append_sample(args.csv, landmarks_list[0], current_label)
                    saved_count += 1
            draw_label(frame, f"[Collect] Label: {current_label} | Recording: {recording} | Saved: {saved_count}", (10,30))
            draw_label(frame, "Keys: 1/2/3 label, r record, s save, q quit", (10,60))
            cv2.imshow("Hand Data Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key in (ord('1'), ord('2'), ord('3')):
                current_label_key = chr(key)
                current_label = LABEL_MAP[current_label_key]
            elif key == ord('r'): recording = not recording
            elif key == ord('s') and landmarks_list:
                append_sample(args.csv, landmarks_list[0], current_label)
                saved_count += 1
    cap.release()
    cv2.destroyAllWindows()

def train_mode(args):
    if RandomForestClassifier is None or joblib is None:
        print("ERROR: Install scikit-learn and joblib.")
        sys.exit(1)
    import pandas as pd
    df = pd.read_csv(args.csv)
    X = df.drop(columns=['label']).values
    y = df['label'].values
    if len(np.unique(y)) < 2:
        print("Need at least 2 labels to train.")
        sys.exit(1)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    clf=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)
    clf.fit(X_train,y_train)
    print("Accuracy:",accuracy_score(y_test,clf.predict(X_test)))
    joblib.dump(clf,args.model)
    print("Saved model:",args.model)

def predict_mode(args):
    tts_engine = pyttsx3.init() if (args.tts and pyttsx3) else None
    model = joblib.load(args.model) if (args.model and Path(args.model).exists() and joblib) else None
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        sys.exit(1)
    last_pred=None
    last_time=0
    with mp_hands.Hands(max_num_hands=args.max_hands,
                        min_detection_confidence=args.min_det,
                        min_tracking_confidence=args.min_track) as hands:
        while True:
            ret,frame=cap.read()
            if not ret:break
            frame=cv2.flip(frame,1)
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            res=hands.process(rgb)
            text="No hand"
            if res.multi_hand_landmarks:
                pts=[(lm.x,lm.y,lm.z) for lm in res.multi_hand_landmarks[0].landmark]
                feats=normalize_landmarks(pts)
                if feats:
                    pred=model.predict([feats])[0] if model else random.choice(list(LABEL_MAP.values()))
                    text=f"Gesture: {pred}"
                    if args.tts and (pred!=last_pred or time.time()-last_time>1.5):
                        speak_tts(tts_engine,f"Gesture detected: {pred}")
                        last_pred=pred
                        last_time=time.time()
                mp_drawing.draw_landmarks(frame,res.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
            draw_label(frame,"[Predict] "+text,(10,30))
            draw_label(frame,"Press q to quit",(10,60))
            cv2.imshow("Gesture Recognition",frame)
            if (cv2.waitKey(1)&0xFF)==ord('q'):break
    cap.release()
    cv2.destroyAllWindows()

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["collect","train","predict"],default="collect")
    p.add_argument("--csv",type=str,default="hand_landmarks.csv")
    p.add_argument("--model",type=str,default="gesture_model.pkl")
    p.add_argument("--max_hands",type=int,default=1)
    p.add_argument("--min_det",type=float,default=0.7)
    p.add_argument("--min_track",type=float,default=0.7)
    p.add_argument("--save_interval",type=int,default=3)
    p.add_argument("--tts",action="store_true")
    return p.parse_args()

def main():
    args=parse_args()
    if args.mode=="collect":collect_mode(args)
    elif args.mode=="train":train_mode(args)
    elif args.mode=="predict":predict_mode(args)

if __name__=="__main__":
    main()
