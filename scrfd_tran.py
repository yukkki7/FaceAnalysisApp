import sys, cv2, json
from insightface.app import FaceAnalysis

img_path = sys.argv[1]
app = FaceAnalysis(name='scrfd_500m', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
faces = app.get(cv2.imread(img_path))

results = []
for face in faces:
    box = face.bbox.astype(float).tolist()
    conf = float(face.det_score)
    results.append({"bbox": box, "conf": conf})

print(json.dumps(results))
