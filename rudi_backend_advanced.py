
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import tempfile

app = Flask(__name__)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video_file = request.files["video"]
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    video_file.save(temp_video_path)

    cap = cv2.VideoCapture(temp_video_path)
    total_frames = 0
    hands_detected = 0

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            total_frames += 1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            if results.multi_hand_landmarks:
                hands_detected += 1

    cap.release()
    activity_ratio = hands_detected / total_frames if total_frames else 0

    feedback = "Needs improvement"
    if activity_ratio > 0.75:
        feedback = "Excellent control and consistency!"
    elif activity_ratio > 0.5:
        feedback = "Good rhythm, but can be improved."

    return jsonify({
        "frames_analyzed": total_frames,
        "hands_detected": hands_detected,
        "motion_activity_score": f"{activity_ratio * 100:.2f}%",
        "feedback": feedback
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
