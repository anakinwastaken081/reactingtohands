import cv2
import mediapipe as mp



mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# This line of code grabs first camera it finds (0 = first camera)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    sucess, frame = cap.read()
    if not sucess:
     print("No camera detected")
     break

    # Convert the RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    #If hand detected
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        #Draws the hand skeleton
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        

    cv2.imshow("Webcam Test", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

cap.release()
cv2.destroyAllWindows()
   



