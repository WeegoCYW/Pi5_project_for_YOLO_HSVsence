from flask import Flask, Response
import cv2
import logging
from ultralytics import YOLO

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# ✅ 載入模型
model_path = "y11n_batch4_e50_size320_renew.pt"
model = YOLO(model_path)

@app.route('/video')
def video_feed():
    logging.debug("Video feed route accessed")

    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logging.error("Camera not accessible.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error reading frame.")
                break

            # ✅ 模型推論，設定信心度閾值為 0.7
            #    YOLOv11 只會回傳信心度高於 0.7 的結果, verbose=False
            results = model(frame, conf=0.7)[0] # 取第一個結果

            # ✅ 繪製結果
            #    因為 results 物件已經過濾，所以這裡只會畫出高於閾值的偵測框
            annotated_frame = results.plot()

            # ✅ 編碼並輸出為串流格式
            _, img_encoded = cv2.imencode('.jpg', annotated_frame)
            img_bytes = img_encoded.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)