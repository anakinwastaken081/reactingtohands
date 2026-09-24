import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

print("[1] Imports done")

options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
print("[2] Options built")

landmarker = vision.HandLandmarker.create_from_options(options)
print("[3] Landmarker created")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("[4] Camera opened:", cap.isOpened())

start = time.time()
last_ts = -1

while True:
    ret, frame = cap.read()
    print("[5] Frame read:", ret)
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    print("[6] Image prepared")

    ts = int((time.time() - start) * 1000)
    if ts <= last_ts:
        ts = last_ts + 1
    last_ts = ts

    result = landmarker.detect_for_video(mp_image, ts)
    print("[7] Detection done. Hands:", len(result.hand_landmarks))

    cv2.imshow("Test", frame)
    print("[8] imshow called")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

landmarker.close()
cap.release()
cv2.destroyAllWindows()