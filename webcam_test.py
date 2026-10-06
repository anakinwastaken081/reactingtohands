import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import math
from collections import deque


# Loads the hand tracking model from MediaPipe
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)
landmarker = vision.HandLandmarker.create_from_options(options)


# Landmark index constants
WRIST = 0
THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP = 4, 8, 12, 16, 20
FINGER_TIPS = [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
PALM_BASE = [WRIST, 5, 9, 13, 17]


# Skeleton connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),              # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),              # Index
    (0, 9), (9, 10), (10, 11), (11, 12),         # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),       # Ring
    (0, 17), (17, 18), (18, 19), (19, 20),       # Pinky
]


# Camera
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
print("Actual resolution:", cap.get(cv2.CAP_PROP_FRAME_WIDTH), "x",
      cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Count fingers function

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])
def is_waving(pts, idx):
    if idx not in wrist_history:
        wrist_history[idx] = deque(maxlen=WAVE_WINDOW)

    wrist_x = pts[WRIST][0]
    wrist_history[idx].append(wrist_x)

    if len(wrist_history[idx]) < WAVE_WINDOW:
        return False #not enought history yet to determine waving

    #Count direction reversals in the recent x positions
    MIN_MOVEMENT = 8 # Minimum movement in pixels to count as a direction change
    diffs= [wrist_history[idx][i+1] - wrist_history[idx][i] for i in range(len(wrist_history[idx])-1)]
    reversals = 0
    for i in range(1, len(diffs)):
        if diffs[i] == 0 or diffs[i-1] == 0:
            continue
        if (diffs[i] > 0) != (diffs[i-1] > 0):
            reversals += 1

    return reversals >= 8 # change for more strict waving detection

def is_fist(pts):
    palm_ref = pts[9]
    tip_ids = [THUMB_TIP, INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
    avg_tip_dist = sum(dist(pts[t], palm_ref) for t in tip_ids) / len(tip_ids)
    hand_size = dist(pts[9], pts[0])  # Distance from palm center to wrist
    return all(dist(pts[t], palm_ref) < hand_size * 0.9 for t in tip_ids)

def count_fingers(pts, handedness_label):
    fingers_up = []

    # Thumb "Up" if tip is farther from palm center.
    palm_ref = pts[9]
    fingers_up.append(dist(pts[THUMB_TIP], palm_ref) > dist(pts[2], palm_ref)* 1.1)

    tip_ids = [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
    for tip_id in tip_ids:
        fingers_up.append(pts[tip_id][1] < pts[tip_id - 2][1])  # Tip is above PIP joint

    return fingers_up.count(True)

fps_smooth = 0.0
wrist_history = {} # one deque per hand to store recent wrist positions for wave detection
WAVE_WINDOW = 20 # How many recent frames to look at for wave detection

# Main Loop
start = time.time()
last_ts = -1
prev_time = start
while True:
    success, frame = cap.read()
    if not success:
        print("No camera detected")
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Video mode needs a timestamp that always increases
    ts = int((time.time() - start) * 1000)
    if ts <= last_ts:
        ts = last_ts + 1
    last_ts = ts

    result = landmarker.detect_for_video(mp_image, ts)
    print("Hands detected:", len(result.hand_landmarks))

    # result.hand_landmarks is a list of hands (each with 21 landmarks)
    for idx, hand in enumerate(result.hand_landmarks):
        pts = [(int(p.x * w), int(p.y * h)) for p in hand]
        
        handedness_label = result.handedness[idx][0].category_name # Left or Right
        handedness_label = "Left" if handedness_label == "Right" else "Right"
       
        if is_fist(pts):
            shape_text = "Fist"
        else:
            finger_count = count_fingers(pts, handedness_label)
            shape_text = f"{handedness_label} Fingers: {finger_count}"
            
        motion_text = " | Waving" if is_waving(pts, idx) else " Not Waving"
        display_text = f"{handedness_label} {shape_text}{motion_text}"

        cv2.putText(frame, display_text, (10, 90 + idx * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 1) Draw skeleton
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

        # 2) Compute palm center as average of the base points
        palm_x = sum(pts[i][0] for i in PALM_BASE) // len(PALM_BASE)
        palm_y = sum(pts[i][1] for i in PALM_BASE) // len(PALM_BASE)
        palm_center = (palm_x, palm_y)

        # 3) Draw line from each fingertip to palm center (magenta)
        for tip_idx in FINGER_TIPS:
            cv2.line(frame, pts[tip_idx], palm_center, (255, 0, 255), 1)

        # 4) Draw dots
        for i, pt in enumerate(pts):
            if i in FINGER_TIPS:
                cv2.circle(frame, pt, 8, (0, 255, 255), -1)   # yellow
            elif i in PALM_BASE:
                cv2.circle(frame, pt, 5, (255, 0, 0), -1)     # blue
            else:
                cv2.circle(frame, pt, 3, (0, 0, 255), -1)     # red

        cv2.circle(frame, palm_center, 10, (255, 0, 0), -1)   # blue palm center

    # FPS Counter
    now = time.time()
    inst = 1 / max(now - prev_time, 1e-6)
    prev_time = now
    fps_smooth = inst if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * inst


    cv2.putText(frame, f"FPS: {int(fps_smooth)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


landmarker.close()
cap.release()
cv2.destroyAllWindows()