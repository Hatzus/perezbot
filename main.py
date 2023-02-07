import cv2
from roboflow import Roboflow
rf = Roboflow(api_key="cn28TNUzySjIEfqf10qU")
project = rf.workspace().project("perezbot")
model = project.version(1).model

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    rect, frame = cap.read()
    bild = rescale_frame(frame, percent=200)
    ret, frame = cap.read()

    predictions = model.predict(bild, confidence=50, overlap=50).json()
    if predictions["predictions"]:
        detections = []
        for prediction in predictions["predictions"]:
            x = prediction["x"]
            y = prediction["y"]
            w = prediction["width"]
            h = prediction["height"]
            class_name = prediction["class"]
            confidence = prediction["confidence"]

            detections.append({
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "class": class_name,
                "confidence": confidence
            })

        for i, detection in enumerate(detections):
            x = detection["x"]
            y = detection["y"]
            w = detection["width"]
            h = detection["height"]
            class_name = detection["class"]
            confidence = detection["confidence"]

            cv2.putText(bild, str(class_name), (int(int(x)-int(int(w)/int(2))), int(int(int(int(y) + int(h))-int(int(h)/int(2)))-int(int(h)+int(10)))), cv2.FONT_HERSHEY_PLAIN, 2, (200, 200, 50), 2)
            cv2.rectangle(bild, (int(int(x)-int(int(w)/int(2))), int(int(int(int(y) + int(h))-int(int(h)/int(2)))-int(h))), (int(int(x)+int(int(w)/int(2))), int(int(int(int(y) + int(h)) + int(int(h) / int(2)))-int(h))), (200, 0, 50), 3)
            cv2.circle(bild, (int(x), int(y)) , 5, (0, 0, 200), 10)
    else:
        print("No predictions available")

    cv2.imshow("bild", bild)

    cv2.waitKey(1)

