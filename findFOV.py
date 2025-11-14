from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# --- 1. 相機參數設定 (假設使用羅技 C920 參數) ---
# 羅技 C920 (焦距 f = 3.67mm, 感光元件 d ≈ 4.27mm)
FOCAL_TO_SENSOR_RATIO = 0.8595 

CAMERA_INDEX = 2  # 樹莓派上 0 或 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# 全局變數用於接收前端的測量數據
measurement_data = {
    'start_x': 100,  # 預設起始像素 X
    'end_x': 540,    # 預設結束像素 X
    'actual_H': 100, # 實物實際長度 H (mm), 預設 100mm
    'distance_D': None # 計算出的距離 D
}


# --- 2. 影像串流與處理函數 ---
def generate_frames():
    # 嘗試開啟相機
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(f"錯誤：無法開啟相機，請檢查 CAMERA_INDEX: {CAMERA_INDEX}")
        return

    # 設置解析度
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    CENTER_Y = CAMERA_HEIGHT // 2

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # --- 影像疊加處理 ---
            
            # 1. 繪製測量基準線 (中央紅線，確保直尺水平對齊此線)
            cv2.line(frame, 
                     (0, CENTER_Y), 
                     (CAMERA_WIDTH, CENTER_Y), 
                     (0, 0, 255), 2) # BGR 顏色 (紅色)
            
            # 2. 繪製實時測量標記 (綠色/藍色)
            start_x = measurement_data.get('start_x', 100)
            end_x = measurement_data.get('end_x', 540)
            
            # 繪製起始標記 (綠色，加粗並標示刻度線)
            cv2.line(frame, 
                     (start_x, CENTER_Y - 30), 
                     (start_x, CENTER_Y + 30), 
                     (0, 255, 0), 3) # 綠色
            
            # 繪製結束標記 (藍色，加粗並標示刻度線)
            cv2.line(frame, 
                     (end_x, CENTER_Y - 30), 
                     (end_x, CENTER_Y + 30), 
                     (255, 0, 0), 3) # 藍色
            
            # 3. 顯示計算結果 (如果已經計算)
            if measurement_data['distance_D'] is not None:
                text = f"D: {measurement_data['distance_D']:.2f} mm"
                # 放在右上方
                cv2.putText(frame, text, (CAMERA_WIDTH - 200, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # 將畫面轉換為 JPEG 格式以串流
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- 3. Flask 路由定義 ---

@app.route('/fov', methods=['GET', 'POST']) # <-- 【修正點】允許 POST 避免 405 錯誤
def fov_interface():
    """主介面路由，載入 HTML 模板"""
    # 儘管允許 POST，但該函數只處理 GET，POST 請求會被忽略，直接渲染模板
    return render_template('fov.html', ratio=FOCAL_TO_SENSOR_RATIO)

@app.route('/video_feed')
def video_feed():
    """視訊串流路由"""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/calculate', methods=['POST'])
def calculate_distance():
    """接收前端像素點和實際長度，並計算距離"""
    global measurement_data
    
    data = request.get_json()
    
    try:
        start_x = int(data['start_x'])
        end_x = int(data['end_x'])
        actual_H = float(data['actual_H']) # 實際長度 H (mm)
        
        # 1. 計算畫面中的像素長度 h
        pixel_h = abs(end_x - start_x)
        if pixel_h == 0:
             raise ValueError("像素長度為零，請拖曳標記")
        
        # 2. 根據 FOV 公式計算距離
        W = actual_H * (CAMERA_WIDTH / pixel_h)
        distance_D = W * FOCAL_TO_SENSOR_RATIO
        
        # 更新全局數據
        measurement_data['start_x'] = start_x
        measurement_data['end_x'] = end_x
        measurement_data['actual_H'] = actual_H
        measurement_data['distance_D'] = distance_D
        
        return jsonify({
            'success': True,
            'distance_D': f"{distance_D:.2f}",
            'pixel_h': pixel_h,
            'W': f"{W:.2f}"
        })

    except Exception as e:
        # print(f"計算錯誤: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # 在樹莓派上執行時，將 host 設為 '0.0.0.0' 才能從其他設備訪問
    app.run(host='0.0.0.0', port=5000, debug=True)
