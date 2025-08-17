import os
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Load model and labels
model = tf.keras.models.load_model('model.h5')
labels = sorted(os.listdir('dataset/asl_alphabet_train'))

# Init text-to-speech
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Init webcam
cap = cv2.VideoCapture(0)

prev_label = ""
label_count = 0
stable_label = ""
word = ""
recent_letters = []
last_time = time.time()

while True:
    ret, frame = cap.read()
    x1, y1, x2, y2 = 50, 50, 350, 350
    roi = frame[y1:y2, x1:x2]

    # Preprocess ROI
    img = cv2.resize(roi, (64, 64))
    img = img / 255.0
    img = img.reshape(1, 64, 64, 3)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred[0])
    label = labels[idx]
    confidence = np.max(pred[0])

    # Stability check
    if label == prev_label:
        label_count += 1
    else:
        label_count = 0
    prev_label = label

    if label_count > 10 and confidence > 0.9:
        if label != stable_label:
            stable_label = label
            print(f"✔️ Added letter: {stable_label}")
            if stable_label.lower() == "space":
                word += " "
            elif stable_label.lower() == "del":
                word = word[:-1]
            elif stable_label.lower() != "nothing":
                word += stable_label
                recent_letters.append(stable_label)
                if len(recent_letters) > 10:
                    recent_letters.pop(0)

    # Draw input rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Overlay panel
    overlay = frame.copy()
    height, width, _ = frame.shape
    panel_height = 180
    cv2.rectangle(overlay, (0, height - panel_height), (width, height), (0, 0, 0), -1)
    alpha = 0.6
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Show predictions and word
    cv2.putText(frame, f"Letter: {stable_label}", (20, height - 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(frame, f"Word: {word}", (20, height - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"Recent: {' '.join(recent_letters)}", (20, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    # Key Buttons - repositioned
    buttons = [("s = Speak", 20), ("r = Reset", 160), ("q = Quit", 300)]
    for text, x in buttons:
        cv2.rectangle(frame, (x, height - 40), (x + 120, height - 20), (50, 50, 50), -1)
        cv2.putText(frame, text, (x + 5, height - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Confidence meter
    center = (width - 80, height - 130)  # Adjusted upward
    radius = 30
    angle = int(confidence * 360)
    cv2.ellipse(frame, center, (radius, radius), 0, 0, angle, (0, 255, 255), -1)
    cv2.circle(frame, center, radius, (255, 255, 255), 2)

    # Green confidence text below meter
    cv2.putText(frame, f"Conf: {int(confidence * 100)}%", (center[0] - 40, center[1] + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - last_time)
    last_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

    # Footer: Credits - smaller and repositioned
    credit_text = "By Manisha & Vishal | DPGITM | BTech 6th Sem"
    cv2.putText(frame, credit_text, (20, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Show the frame
    cv2.imshow("🤖 Sign Language Translator", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s') and word:
        speak(word)
    elif key == ord('r'):
        print("🔄 Word reset.")
        word = ""
        stable_label = ""
        recent_letters = []

cap.release()
cv2.destroyAllWindows()
