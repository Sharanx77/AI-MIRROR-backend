import cv2
import numpy as np
# load your pretrained CNN weights here
MODEL_PATH = "cnn_deepfake.onnx"
net = cv2.dnn.readNet(MODEL_PATH)
def is_deepfake(frame: np.ndarray) -> float:
"""Return probability (0‑1) that frame is deepfake."""
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (224,224))
net.setInput(blob)
out = net.forward()
return float(out[0][0])