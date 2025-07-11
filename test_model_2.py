import numpy as np
import cv2
import pickle
import time
import os
import csv
from datetime import datetime

# Setup
frameWidth = 640
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
cooldown_seconds = 3

# Directories
save_dir = "DetectedSigns"
log_file = "detection_log.csv"
os.makedirs(save_dir, exist_ok=True)

# Load trained CNN model
with open("model_trained.p", "rb") as f:
    model = pickle.load(f)

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

if not cap.isOpened():
    print("Error: Webcam not accessible.")
    exit()

# Preprocessing
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Class labels
def getClassName(classNo):
    labels = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vechiles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
        'No vechiles', 'Vechiles over 3.5 metric tons prohibited', 'No entry',
        'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
        'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
        'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
        'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vechiles over 3.5 metric tons'
    ]
    return labels[classNo] if 0 <= classNo < len(labels) else "Unknown"

# CSV logging
def log_detection(sign_name, probability):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, sign_name, round(probability * 100, 2)])

# Create log file if not exists
if not os.path.exists(log_file):
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Detected Sign", "Probability (%)"])

last_detect_time = {}

# Loop
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read from webcam.")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 1)
    edges = cv2.Canny(frame_blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > max_area and len(approx) >= 4:
                best_cnt = cnt
                max_area = area

    if best_cnt is not None:
        x, y, w, h = cv2.boundingRect(best_cnt)
        roi = frame[y:y+h, x:x+w]

        try:
            img = cv2.resize(roi, (32, 32))
            img = preprocessing(img)
            img = img.reshape(1, 32, 32, 1)

            predictions = model.predict(img)
            classIndex = int(np.argmax(predictions))
            probabilityValue = float(np.max(predictions))

            if probabilityValue > threshold:
                className = getClassName(classIndex)
                now = time.time()

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{className}", (x, y - 10), font, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"{round(probabilityValue*100, 2)}%", (x, y + h + 20), font, 0.6, (0, 255, 0), 2)

                if (className not in last_detect_time) or (now - last_detect_time[className] > cooldown_seconds):
                    last_detect_time[className] = now

                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{className}_{timestamp}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, frame)

                    log_detection(className, probabilityValue)

                    print(f"[Detected] {className} | Probability: {round(probabilityValue * 100, 2)}%")
                    print(f"[Saved] {filename}")
        except Exception as e:
            print("Error processing ROI:", e)

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
