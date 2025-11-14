import cv2
import numpy as np
from flask import Flask, render_template, Response
import threading
import time

# --- 參數設定 ---
# 請根據實際測試，重新調整 REF_AREA 和 MAX_AREA
REF_AREA = 11000  # 設定未膨脹的參考面積，需要根據你的圖片調整
MAX_AREA = 56000 # 設定完全膨脹的最大面積，需要根據你的圖片調整

# 全域變數和鎖
camera = None
frame_event = threading.Event()
latest_frame = None

def capture_frames():
    """獨立的執行緒，負責持續從攝影機讀取影像。"""
    global latest_frame
    global camera

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("無法開啟攝影機，請檢查攝影機是否連接正確。")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        latest_frame = frame.copy()
        frame_event.set()
        frame_event.clear()
        time.sleep(0.01)

def calculate_inflation_percentage(current_area, ref_area=REF_AREA, max_area=MAX_AREA):
    if max_area - ref_area == 0:
        return 0
    percentage = (current_area - ref_area) / (max_area - ref_area) * 100
    return max(0, min(100, percentage))

def generate_frames():
    while True:
        frame_event.wait()
        
        if latest_frame is None:
            continue
        
        frame = latest_frame.copy()
        
        # --- 影像處理：針對白色透明袋子 (使用二值化 V5) ---
        
        # 1. 轉換為灰階
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. **極致加大高斯模糊**：使用 (31, 31) 來徹底消除細小的網格孔洞
        blur = cv2.GaussianBlur(gray, (31, 31), 0)
        
        # 3. **二值化處理 (Thresholding)**：將模糊後的影像區分為前景(袋子)和背景
        # 大幅降低閾值到 90，嘗試找回袋子的主體亮度
        # *** 閾值 90 是當前最關鍵的微調參數 ***
        _, thresh = cv2.threshold(blur, 130, 255, cv2.THRESH_BINARY)
        
        # 4. **形態學關閉操作**：連接間隙並移除小雜點
        # 加大到 (9, 9) 以獲得更強的輪廓連接效果
        kernel_morph = np.ones((9, 9), np.uint8) 
        processed_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_morph)
        
        # 5. 尋找輪廓：直接在處理後的二值圖上找輪廓
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- 處理與顯示 ---

        area = 0
        percentage = 0
        
        # 篩選輪廓：只考慮面積大於一定閾值的輪廓
        min_area_threshold = 1000 
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area_threshold]

        if valid_contours:
            # 找到最大的有效輪廓
            largest_contour = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            percentage = calculate_inflation_percentage(area)
            
            # 繪製輪廓
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Area: {area:.2f} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Inflation: {percentage:.2f}%", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

app = Flask(__name__)

@app.route('/area')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # HTML 顯示內容也更新以反映當前的偵測方法
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>夾鏈袋膨脹偵測 (白色透明袋子優化)</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                margin-top: 50px;
                background-color: #f0f0f0;
            }
            img {
                border: 2px solid #333;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>夾鏈袋膨脹偵測 (白色透明袋子優化)</h1>
        <p>當前使用「極致模糊 (31x31) + 二值化 (90) + 加大形態學關閉 (9x9)」來辨識白色透明袋子輪廓。</p>
        <img src="/area" alt="影像串流" width="640" height="480">
    </body>
    </html>
    """

if __name__ == '__main__':
    thread = threading.Thread(target=capture_frames)
    thread.daemon = True 
    thread.start()
    
    app.run(host='0.0.0.0', port=5002)
