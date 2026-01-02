import cv2
import mediapipe as mp
import pyttsx3

print("✅ All packages working fine!")

# Quick webcam test (press 'q' to exit)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Webcam Test", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

