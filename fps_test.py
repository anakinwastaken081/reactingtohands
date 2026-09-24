import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# --- Test A: camera only (no MediaPipe) ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print("Camera FPS setting reported:", cap.get(cv2.CAP_PROP_FPS))

for _ in range(10):          # warm-up frames
    cap.read()

n = 0
t0 = time.time()
while time.time() - t0 < 5:
    ok, frame = cap.read()
    if not ok:
        break
    n += 1
print(f"A) Camera only: {n / (time.time() - t0):.1f} FPS")

# --- Test B: camera + MediaPipe (no drawing, no window) ---
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = vision.HandLandmarker.create_from_options(options)

start = time.time()
last_ts = -1
t_read = 0.0
t_detect = 0.0
n = 0
t0 = time.time()
while time.time() - t0 < 5:
    a = time.perf_counter()
    ok, frame = cap.read()
    b = time.perf_counter()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts = int((time.time() - start) * 1000)
    if ts <= last_ts:
        ts = last_ts + 1
    last_ts = ts
    landmarker.detect_for_video(mp_image, ts)
    c = time.perf_counter()
    t_read += b - a
    t_detect += c - b
    n += 1

print(f"B) Camera + MediaPipe: {n / (time.time() - t0):.1f} FPS")
print(f"   avg camera wait: {t_read / n * 1000:.1f} ms | avg convert+detect: {t_detect / n * 1000:.1f} ms")

landmarker.close()
cap.release()