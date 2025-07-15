from deepface import DeepFace
# cache models to avoid reload penalty
analyzer = DeepFace
def detect_face_emotion(frame):
try:
result = analyzer.analyze(frame, actions=["emotion"],
enforce_detection=False, prog_bar=False)
return result[0]["dominant_emotion"], result[0]["emotion"].get(result[0]["dominant_emotion"], 0)/100
        except Exception:
        return "unknown", 0.0