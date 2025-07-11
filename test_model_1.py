import numpy as np
import cv2
import pickle
import threading
import time
import os
import csv
from datetime import datetime
from playsound import playsound

# Setup
frameWidth = 1080
frameHeight = 480
brightness = 180
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX
buzzer_sound = "alert.mp3"
cooldown_seconds = 3

# Directories
save_dir = "DetectedSigns"
log_file = "detection_log.csv"
os.makedirs(save_dir, exist_ok=True)

# Load trained CNN model
with open("model_trained.p", "rb") as f:
    model = pickle.load(f)

# Webcam setup
cap = cv2.VideoCapture(1)
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
# sending new image to compare with the cnn to give output probability
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

# Class labels
def getClassName(classNo):
    if   classNo == 0: return 'Speed Limit 20 km/h'
    elif classNo == 1: return 'Speed Limit 30 km/h'
    elif classNo == 2: return 'Speed Limit 50 km/h'
    elif classNo == 3: return 'Speed Limit 60 km/h'
    elif classNo == 4: return 'Speed Limit 70 km/h'
    elif classNo == 5: return 'Speed Limit 80 km/h'
    elif classNo == 6: return 'End of Speed Limit 80 km/h'
    elif classNo == 7: return 'Speed Limit 100 km/h'
    elif classNo == 8: return 'Speed Limit 120 km/h'
    elif classNo == 9: return 'No passing'
    elif classNo == 10: return 'No passing for vechiles over 3.5 metric tons'
    elif classNo == 11: return 'Right-of-way at the next intersection'
    elif classNo == 12: return 'Priority road'
    elif classNo == 13: return 'Yield'
    elif classNo == 14: return 'Stop'
    elif classNo == 15: return 'No vechiles'
    elif classNo == 16: return 'Vechiles over 3.5 metric tons prohibited'
    elif classNo == 17: return 'No entry'
    elif classNo == 18: return 'General caution'
    elif classNo == 19: return 'Dangerous curve to the left'
    elif classNo == 20: return 'Dangerous curve to the right'
    elif classNo == 21: return 'Double curve'
    elif classNo == 22: return 'Bumpy road'
    elif classNo == 23: return 'Slippery road'
    elif classNo == 24: return 'Road narrows on the right'
    elif classNo == 25: return 'Road work'
    elif classNo == 26: return 'Traffic signals'
    elif classNo == 27: return 'Pedestrians'
    elif classNo == 28: return 'Children crossing'
    elif classNo == 29: return 'Bicycles crossing'
    elif classNo == 30: return 'Beware of ice/snow'
    elif classNo == 31: return 'Wild animals crossing'
    elif classNo == 32: return 'End of all speed and passing limits'
    elif classNo == 33: return 'Turn right ahead'
    elif classNo == 34: return 'Turn left ahead'
    elif classNo == 35: return 'Ahead only'
    elif classNo == 36: return 'Go straight or right'
    elif classNo == 37: return 'Go straight or left'
    elif classNo == 38: return 'Keep right'
    elif classNo == 39: return 'Keep left'
    elif classNo == 40: return 'Roundabout mandatory'
    elif classNo == 41: return 'End of no passing'
    elif classNo == 42: return 'End of no passing by vechiles over 3.5 metric tons'

# Sound playback
def play_buzzer():
    threading.Thread(target=playsound, args=(buzzer_sound,), daemon=True).start()

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

last_buzzer_time = {}

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

    # Assume the largest contour is the traffic sign
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:  # minimum area filter
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

                # Draw box and text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{className}", (x, y - 10), font, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"{round(probabilityValue*100, 2)}%", (x, y + h + 20), font, 0.6, (0, 255, 0), 2)

                if (className not in last_buzzer_time) or (now - last_buzzer_time[className] > cooldown_seconds):
                    last_buzzer_time[className] = now

                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{className}_{timestamp}.jpg"
                    filepath = os.path.join(save_dir, filename)
                    cv2.imwrite(filepath, frame)

                    log_detection(className, probabilityValue)
                    play_buzzer()

                    print(f"[Detected] {className} | Probability: {round(probabilityValue * 100, 2)}%")
                    print(f"[Saved] {filename}")
        except Exception as e:
            print("Error processing ROI:", e)

    # Display
    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
