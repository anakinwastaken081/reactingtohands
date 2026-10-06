Reacting to Hands — Real-Time Hand Gesture Recognition

A real-time hand-tracking and gesture-recognition application built with Python, OpenCV, and MediaPipe's Hand Landmarker model. Tracks up to two hands from a webcam feed, counts extended fingers, detects a closed fist, and recognizes a waving motion — all live, frame by frame.

Status: In progress, AI-assisted development.

What it does:
Detects and draws a 21-point hand skeleton for up to 2 hands at once
Computes a palm-center reference point and draws lines from each fingertip to it
Counts extended fingers (0–5) per hand, correctly handling thumb orientation regardless of which side of the hand faces the camera
Detects a closed fist as its own state (more reliable than guessing individual finger positions when fingers are occluded)
Detects a waving gesture by tracking wrist position over a rolling window and counting direction reversals
Displays live FPS and per-hand status on screen
How it was built

This project was built through an iterative, test-driven process: run the program, observe real behavior, diagnose the cause, apply a fix, and retest before moving on. That process surfaced and resolved several real issues along the way, including:

A camera driver bottleneck that capped frame rate at a fixed 15 FPS regardless of lighting or resolution — isolated through structured A/B testing (varying exposure, resolution, and backend independently) and resolved by switching from the DirectShow to Media Foundation camera backend
An orientation-dependent thumb-detection bug, fixed by switching from a position-based rule to a distance-based one that works regardless of which side of the hand faces the camera
A multi-hand display bug where both hands' finger counts rendered on top of each other
Unreliable finger-counting when fingers were curled/occluded (e.g. a fist), addressed by detecting the overall hand shape instead of trusting individual occluded landmark positions

The implementation code was written with AI assistance (Claude); the testing strategy, bug diagnosis, requirement decisions, and verification of each fix were done hands-on.

Tech stack
Python 3
OpenCV — camera capture, image processing, drawing
MediaPipe (Tasks API, HandLandmarker) — hand detection and 21-point landmark tracking
Setup
Clone the repo and create a virtual environment:
bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   source venv/bin/activate     # macOS/Linux
Install dependencies:
bash
   pip install opencv-contrib-python mediapipe
Download the hand landmark model and place it in the project root as hand_landmarker.task: MediaPipe Hand Landmarker model
Run it:
bash
   python webcam_test.py
Press q to quit.
Known limitations / next steps
Single 2D camera limits gesture precision to the image plane; no true depth
Wave detection threshold is tuned by eye, not auto-calibrated per user
Next planned stage: a generic gesture-response system (e.g. sound/visual feedback) and additional static gestures (peace sign, thumbs up)
Notes on the included files
webcam_test.py — main application
fps_test.py / diagnostic.py — standalone scripts used during performance diagnosis (camera-only vs. camera+MediaPipe FPS comparisons)
hand_landmarker.task — pretrained MediaPipe model (not authored by this project)
