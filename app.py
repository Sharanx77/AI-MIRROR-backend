from flask import Flask, request, jsonify
import cv2, os, uuid, mimetypes, numpy as np
from deepfake_detector import is_deepfake
from face_emotion import detect_face_emotion
from voice_emotion import detect_voice_emotion
from emotion_analyzer import check_consistency
from pydub import AudioSegment

app = Flask(__name__)
UPLOAD_DIR = "tmp"; os.makedirs(UPLOAD_DIR, exist_ok=True)

# helper to grab first video frame
def _first_frame(path):
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read(); cap.release()
    if not ret:
        raise ValueError("Cannot read frame")
    return frame

@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    fname = f"{uuid.uuid4()}"; ext = os.path.splitext(file.filename)[1]
    saved = os.path.join(UPLOAD_DIR, fname + ext)
    file.save(saved)

    mime, _ = mimetypes.guess_type(saved)
    mime = mime or ""

    # ---------------- IMAGE ----------------
    if mime.startswith("image"):
        img_bytes = np.frombuffer(open(saved, "rb").read(), np.uint8)
        frame = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        deepfake_prob = is_deepfake(frame)
        face_em, face_conf = detect_face_emotion(frame)
        return jsonify({
            "mode": "image",
            "deepfake_probability": deepfake_prob,
            "face_emotion": face_em,
            "face_confidence": face_conf
        })

    # ---------------- VIDEO ----------------
    elif mime.startswith("video"):
        try:
            frame = _first_frame(saved)
        except ValueError:
            return "Cannot read video", 400

        deepfake_prob = is_deepfake(frame)
        face_em, face_conf = detect_face_emotion(frame)

        # audio extraction for voice emotion
        wav_path = saved.rsplit(".",1)[0] + ".wav"
        AudioSegment.from_file(saved).export(wav_path, format="wav")
        voice_em, voice_conf = detect_voice_emotion(wav_path)
        consistency = check_consistency(face_em, voice_em)

        return jsonify({
            "mode": "video",
            "deepfake_probability": deepfake_prob,
            "face_emotion": face_em,
            "face_confidence": face_conf,
            "voice_emotion": voice_em,
            "voice_confidence": voice_conf,
            "consistency": consistency
        })

    else:
        return "Unsupported file type", 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))