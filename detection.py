import torch
import cv2
import numpy as np
import time
from pylibdmtx.pylibdmtx import decode
from PIL import Image


class CodeDetection():
    def __init__(self):
        self.model = self.load_model()
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:", self.device)

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5',
                               'custom',
                               path='datamatrix-best.pt',
                               )
        return model

    def score_frame(self, frame):
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord

    def plot_boxes(self, results, frame):
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                    row[3] * y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), bgr, 2)
        return frame

    def decode_dmx(self, frame):
        pass

    def __call__(self):
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            start_time = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow("img", frame)
            key = cv2.waitKey(1)
            timeout = 3000
            if key == ord("p"):
                cv2.waitKey(timeout)
                height, width, channels = frame.shape

                offset_x = 0
                offset_y = 0
                print('****START****')
                for result in results[1]:
                    x_min = int(result[0].numpy() * width)
                    y_min = int(result[1].numpy() * height)
                    x_max = int(result[2].numpy() * width)
                    y_max = int(result[3].numpy() * height)

                    x_min += offset_x
                    y_min += offset_y
                    x_max += offset_x
                    y_max += offset_y

                    # Crop the object from the original image
                    cropped_object = frame[y_min:y_max, x_min:x_max, :]

                    decoded = decode(cropped_object, timeout=timeout, min_edge=30, corrections=2)

                    print(decoded)

                    # cropped_image = Image.fromarray(cropped_object)
                    # cropped_image.save(f'cropped_image.jpg')
                print('****STOP****')
            elif key & 0xFF == ord('q'):
                break


detection = CodeDetection()
detection()
