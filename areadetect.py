from flask import Flask, render_template, Response
import cv2
import numpy as np
import os

app = Flask(__name__)

# --- 參數設定 ---
# 根據你的氣球實際大小，調整這些面積值
# 你可以先運行一次靜態偵測程式，觀察未膨脹和膨脹後的面積，來設定這些值。
REF_AREA = 2500  # 設定未膨脹的參考面積，需要根據你的圖片調整
MAX_AREA = 25000 # 設定完全膨脹的最大面積，需要根據你的圖片調整

# 設定氣球的 HSV 顏色範圍（黃色）
# 這些值需要根據你的實際光照條件微調
# LOWER_HSV = np.array([20, 100, 100])
# UPPER_HSV = np.array([40, 255, 255])

# 設定氣球的 HSV 顏色範圍（橘色）
# 橘色的 H 值大約在 5 到 25 之間。請根據實際光照條件微調。
# LOWER_HSV = np.array([5, 100, 100]) 
# UPPER_HSV = np.array([15, 255, 255])

# 設定氣球的 HSV 顏色範圍（粉色）
# 粉色的 H 值大約在 160 到 180 之間。請根據實際光照條件微調。
# LOWER_HSV = np.array([170, 55, 55]) 
# UPPER_HSV = np.array([180, 255, 255])

# 設定氣球的 HSV 顏色範圍（咖啡色）
# 咖啡色是低飽和度、低亮度的橘色/紅色。
LOWER_HSV = np.array([10, 50, 20])      # H: 鎖定橘色 (10-30), S, V 下限調低以捕捉暗色調
UPPER_HSV = np.array([30, 255, 200])    # 限制亮度上限 200，排除過亮的反光

# 開啟攝影機
camera = cv2.VideoCapture(0)
# 或者使用 IP 攝影機
# camera = cv2.VideoCapture("http://192.168.99.118:8080/video")

# --- 核心邏輯 ---

def calculate_inflation_percentage(current_area, ref_area=REF_AREA, max_area=MAX_AREA):
    """
    根據當前面積，與參考面積和最大面積計算膨脹百分比。
    使用線性插值，將面積範圍 [ref_area, max_area] 映射到百分比 [0, 100]。
    """
    # 確保不會除以零
    if max_area - ref_area == 0:
        return 0
    
    # 將當前面積值線性映射到 0-100%
    percentage = (current_area - ref_area) / (max_area - ref_area) * 100
    
    # 限制百分比在 0 到 100 之間
    return max(0, min(100, percentage))

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # 影像處理
        # 1. 轉換到 HSV 顏色空間
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. 建立遮罩以分離氣球
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        
        # 3. 形態學操作（可選，用於平滑遮罩）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 4. 在遮罩上尋找輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 預設面積和百分比
        area = 0
        percentage = 0
        
        if contours:
            # 找到最大的輪廓（即氣球）
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # 計算膨脹百分比
            percentage = calculate_inflation_percentage(area)
            
            # 繪製輪廓
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        
        # 繪製文字，顯示面積和百分比
        cv2.putText(frame, f"Area: {area:.2f} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Inflation: {percentage:.2f}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 傳輸 MJPEG 串流
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/area')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # 在樹莓派上執行時，將 host 設為 '0.0.0.0' 允許從外部訪問
    app.run(host='0.0.0.0', port=5002)