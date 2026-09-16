import cv2
import numpy as np
import math
from pycaw.pycaw import AudioUtilities
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
import time

smooth_distance = 20
last_update = 0

# Initialize Pycaw to hook into the system's core audio endpoint
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

# Fetch system audio volume ranges (typically yields a range like -65.25 to 0.0 dB)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

# UI Setup variables
vol_percent = 0
bar_height = 400

# Begin capturing live video feed from primary webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7) as hands:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip frame horizontally for a more natural mirror view reflection
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Convert the color space from BGR to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        gesture_text = "VOLUME CONTROL"
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw the structural hand skeleton tracking dots onto the screen
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract pixel locations for required landmarks
                # Landmark 4 = Thumb Tip, Landmark 8 = Index Finger Tip
                lm_list = hand_landmarks.landmark
                x1, y1 = int(lm_list[4].x * w), int(lm_list[4].y * h)
                x2, y2 = int(lm_list[8].x * w), int(lm_list[8].y * h)
                
                # Extract positions for remaining finger tips to check for a fist structure
                y_index_base = lm_list[6].y
                y_middle_tip = lm_list[12].y
                y_ring_tip = lm_list[16].y
                y_pinky_tip = lm_list[20].y

                # Mute Detection: If fingers are folded down below their knuckle joints
                if y_middle_tip > y_index_base and y_ring_tip > y_index_base and y_pinky_tip > y_index_base:
                    gesture_text = "MUTE (Fist)"
                    volume.SetMute(1, None)
                    vol_percent = 0
                    bar_height = 400
                else:
                    # Unmute and control via pinch tracking distance calculation
                    volume.SetMute(0, None)
                    
                    # Calculate straight-line Euclidean distance between thumb and index tips
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Calculate distance
                    distance = math.hypot(x2 - x1, y2 - y1)
                    
                    # Fix these lines in your code:
                    vol_db = np.interp(distance, [20, 200], [minVol, maxVol])

                    volume.SetMasterVolumeLevel(vol_db, None)

                    vol_percent = np.interp(distance, [20, 200], [0, 100])
                    bar_height = np.interp(distance, [20, 200], [400, 150])

        # Draw the application dashboard interface panel overlays (mirroring your image)
        cv2.putText(frame, f"Gesture: {gesture_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Volume: {int(vol_percent)}%", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.putText(frame, "Pinch = Volume", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Fist = Mute", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Open Palm = Unmute", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Press Q to Exit", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw the dynamic vertical volume status bar indicator graphic
        cv2.rectangle(frame, (580, 150), (610, 400), (100, 100, 100), 3)
        cv2.rectangle(frame, (580, int(bar_height)), (610, 400), (0, 255, 0), cv2.FILLED)

        # Launch the display monitor window
        cv2.imshow('Finger Voice Control Dashboard', frame)
        
        # Break loop execution cleanly when the 'q' key is pressed down
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
