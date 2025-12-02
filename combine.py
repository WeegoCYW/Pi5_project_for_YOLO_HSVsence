import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import threading
import time
import logging
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import json

# ====================================================================
# ⭐ 全域設定與資源初始化
# ====================================================================

## MQTT 範例
MQTT_BROKER = "192.168.99.168"   # MQTT broker IP
MQTT_PORT = 1883
MQTT_USERNAME = ""              # 若沒有帳密可留空
MQTT_PASSWORD = ""
# 訂閱範例指令: mosquitto_sub -h localhost -t pi5/vision/yolo

mqtt_client = mqtt.Client()

if MQTT_USERNAME:
    mqtt_client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()
## MQTT 範例

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# 載入 YOLO 模型 (僅載入一次)
try:
    model_path = "y11n_batch4_e50_size320_renew.pt"
    model = YOLO(model_path)
    logging.info(f"YOLOv11 model loaded from {model_path}")
except Exception as e:
    logging.error(f"Failed to load YOLO model: {e}")
    model = None

# --- Camera 0 (YOLO) 專用變數 ---
yolo_camera = None
yolo_frame_event = threading.Event()
yolo_latest_frame = None

# --- Camera 1 (HSV/面積) 專用變數 ---
area_camera = None
area_frame_event = threading.Event()
area_latest_frame = None

is_running = True 

# --- HSV/Area 參數 (共用 API 更新) ---
hsv_params = {
    'H_min': 10, 'S_min': 50, 'V_min': 100,
    'H_max': 30, 'S_max': 255, 'V_max': 255
}
hsv_lock = threading.Lock() 

area_params = {
    'Ref_Area': 2500, 
    'Max_Area': 25000 
}
area_lock = threading.Lock() 

# ====================================================================
# 雙攝影機擷取執行緒函數
# ====================================================================

def capture_frames_yolo(camera_index=0):
    # """獨立執行緒：持續從 Camera 0 讀取影像給 YOLO 用，並發布 MQTT"""
    global yolo_latest_frame, yolo_camera, is_running

    yolo_camera = cv2.VideoCapture(camera_index)
    if not yolo_camera.isOpened():
        logging.error(f"無法開啟 Camera {camera_index} (YOLO)。")
        return

    logging.info(f"Camera {camera_index} (YOLO) capture thread started.")

    while is_running:
        success, frame = yolo_camera.read()
        if not success:
            continue

        yolo_latest_frame = frame.copy()
        yolo_frame_event.set()
        yolo_frame_event.clear()

        # --- YOLO 偵測 ---
        if model is not None:
            results = model(frame, conf=0.7, verbose=False)[0]

            from collections import defaultdict
            yolo_aggregate = defaultdict(list)

            for box in results.boxes:
                cls = int(box.cls[0])
                label = results.names[cls]
                conf = float(box.conf[0])
                yolo_aggregate[label].append(conf)

            # 計算平均信心度
            yolo_data = []
            for label, conf_list in yolo_aggregate.items():
                avg_conf = sum(conf_list) / len(conf_list)
                # 範例：只發送 avg_conf >= 0.6 的類別，可依需求調整
                if avg_conf >= 0.5:
                    yolo_data.append({
                        "label": label,
                        "confidence": round(avg_conf, 3)
                    })

            if yolo_data:

                # 取得陣列中的第一個元素 (物件)
                single_object = yolo_data[0]   
                # 將單一物件轉換為 JSON 字串
                json_output = json.dumps(single_object)
                # 發布 MQTT 訊息
                mqtt_client.publish("pi5/vision/yolo", json_output)
                print(json_output) 

        time.sleep(0.01)

    if yolo_camera:
        yolo_camera.release()
    logging.info(f"Camera {camera_index} (YOLO) capture thread stopped.")


def capture_frames_area(camera_index=1):
    # """獨立執行緒：持續從 Camera 1 讀取影像給 HSV/面積用，並發布 MQTT"""
    global area_latest_frame, area_camera, is_running

    area_camera = cv2.VideoCapture(camera_index)
    if not area_camera.isOpened():
        logging.error(f"無法開啟 Camera {camera_index} (AREA)。")
        return

    logging.info(f"Camera {camera_index} (AREA) capture thread started.")

    while is_running:
        success, frame = area_camera.read()
        if not success:
            continue

        # --- HSV 偵測 ---
        with hsv_lock:
            LOWER_HSV = np.array([hsv_params['H_min'], hsv_params['S_min'], hsv_params['V_min']])
            UPPER_HSV = np.array([hsv_params['H_max'], hsv_params['S_max'], hsv_params['V_max']])

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        kernel = np.ones((7, 7), np.uint8)
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = 0
        percentage = 0

        min_area_threshold = 500
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area_threshold]
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            percentage = calculate_inflation_percentage(area)
            # 繪製輪廓
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        # 繪製文字，顯示面積和百分比
        cv2.putText(frame, f"Area: {area:.2f} px", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Inflation: {percentage:.2f}%", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)    

        # --- 發布 MQTT ---
        area_data = {"inflation_percentage": round(percentage, 2)}
        mqtt_client.publish("pi5/vision/area", json.dumps(area_data))
        print(json.dumps(area_data))

        area_latest_frame = frame.copy()
        area_frame_event.set()
        area_frame_event.clear()

        time.sleep(0.1)

    if area_camera:
        area_camera.release()
    logging.info(f"Camera {camera_index} (AREA) capture thread stopped.")

# ====================================================================
# 輔助計算函數 (不變)
# ====================================================================

def calculate_inflation_percentage(current_area):
    # """
    # 根據當前面積，與參考面積和最大面積計算膨脹百分比。
    # """
    with area_lock:
        ref_area = area_params['Ref_Area']
        max_area = area_params['Max_Area']
    
    if max_area - ref_area <= 0:
        return 0
    
    percentage = (current_area - ref_area) / (max_area - ref_area) * 100
    
    return max(0, min(100, percentage))

# ====================================================================
# 影像串流產生器 (YOLO 偵測) - 使用 Camera 0 數據
# ====================================================================

@app.route('/yolo')
def video_feed_yolo():
    logging.debug("YOLO video feed route accessed")

    def generate_frames_yolo():
        while True:
            yolo_frame_event.wait()
            if yolo_latest_frame is None:
                continue
            frame = yolo_latest_frame.copy()
            results = model(frame, conf=0.7, verbose=False)[0]
            annotated_frame = results.plot(conf=False)  # 或 results.plot() 如果你希望加上 YOLO 標註
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames_yolo(), mimetype='multipart/x-mixed-replace; boundary=frame')



# ====================================================================
# 影像串流產生器 (HSV/面積偵測) - 使用 Camera 1 數據
# ====================================================================

@app.route('/area')
def video_feed_area():
    logging.debug("Area video feed route accessed")

    def generate_frames_area():
        while True:
            area_frame_event.wait()
            if area_latest_frame is None:
                continue
            frame = area_latest_frame.copy()
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate_frames_area(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ====================================================================
# API 路由與 HTML 介面 (保持不變)
# ====================================================================

# ... (hsv_update, area_update, index 函數與上一個答案中的整合版本相同) ...
@app.route('/hsv_update', methods=['POST'])
def hsv_update():
    # """接收來自前端的 HSV 參數並更新全域變數。"""
    global hsv_params
    data = request.get_json()
    
    with hsv_lock:
        hsv_params = {
            'H_min': int(data.get('H_min', 0)),
            'S_min': int(data.get('S_min', 0)),
            'V_min': int(data.get('V_min', 0)),
            'H_max': int(data.get('H_max', 180)),
            'S_max': int(data.get('S_max', 255)),
            'V_max': int(data.get('V_max', 255))
        }
        
        hsv_params['H_max'] = max(hsv_params['H_min'], hsv_params['H_max'])
        hsv_params['S_max'] = max(hsv_params['S_min'], hsv_params['S_max'])
        hsv_params['V_max'] = max(hsv_params['V_min'], hsv_params['V_max'])
        
    return jsonify(success=True)

@app.route('/area_update', methods=['POST'])
def area_update():
    # """接收來自前端的面積參數 (Ref_Area 和 Max_Area) 並更新全域變數。"""
    global area_params
    data = request.get_json()
    
    with area_lock:
        try:
            new_ref = int(data.get('Ref_Area', area_params['Ref_Area']))
            new_max = int(data.get('Max_Area', area_params['Max_Area']))

            if new_max < new_ref:
                new_max = new_ref 
            
            area_params['Ref_Area'] = max(0, new_ref)
            area_params['Max_Area'] = max(0, new_max)
            
        except (TypeError, ValueError) as e:
            logging.error(f"面積參數更新錯誤: {e}")
            return jsonify(success=False, message="Invalid area parameters"), 400
        
    return jsonify(success=True)

@app.route('/')
def index():
    # """HTML 介面 (同時顯示兩個影像串流和參數控制)"""
    global hsv_params
    global area_params
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>雙攝影機偵測與控制介面</title>
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                text-align: center;
                margin: 0;
                padding: 20px;
                background-color: #f0f4f8;
            }}
            .container {{
                max-width: 1400px; /* 增加最大寬度以容納兩個影片 */
                margin: 0 auto;
            }}
            h1 {{
                color: #2c3e50;
                margin-bottom: 10px;
            }}
            p {{
                color: #7f8c8d;
                margin-bottom: 20px;
            }}
            .main-content {{
                display: flex;
                gap: 20px;
                margin-top: 20px;
                flex-wrap: wrap;
                justify-content: center;
            }}
            .video-stream-container {{
                flex: 1 1 640px; /* 每個影片容器佔用 640px 寬度 */
                max-width: 640px;
                background-color: #ffffff;
                border-radius: 8px;
                padding: 10px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            }}
            .video-stream {{
                width: 100%;
                border: 4px solid #34495e;
                border-radius: 6px;
                display: block;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            .controls {{
                flex: 0 0 350px;
                padding: 20px;
                background-color: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                text-align: left;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            @media (max-width: 1400px) {{
                .main-content {{
                    flex-direction: column;
                    align-items: center;
                }}
                .video-stream-container {{
                    max-width: 90%;
                }}
                .controls {{
                    width: 90%;
                    max-width: 640px;
                }}
            }}
            .control-group {{
                padding: 15px;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
            }}
            .hsv-label, .area-label {{
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 10px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
            }}
            .hsv-label {{ color: #2980b9; }}
            .area-label {{ color: #e67e22; }}
            
            .slider-group {{
                margin-bottom: 15px;
            }}
            .slider-group label {{
                display: block;
                font-weight: 500;
                color: #34495e;
                margin-bottom: 5px;
                display: flex;
                justify-content: space-between;
            }}
            .slider-group input[type="range"] {{
                width: 100%;
                margin: 0;
            }}
            .video-title {{
                font-size: 1.5em;
                color: #34495e;
                margin-bottom: 10px;
                font-weight: 600;
            }}
        </style>
        </head>
    <body>
        <div class="container">
            <h1>雙攝影機偵測與控制介面</h1>
            <p>左側為 YOLO 偵測畫面 (Camera 0)，右側為 HSV/面積偵測畫面 (Camera 2)，右下方為參數控制。</p>
            
            <div class="main-content">
                
                <div class="video-stream-container">
                    <div class="video-title"> Camera 0：YOLO 辛香料偵測</div>
                    <img class="video-stream" src="/yolo" alt="YOLO 影像串流" width="640" height="480">
                </div>
                
                <div class="video-stream-container">
                    <div class="video-title"> Camera 2：HSV/面積膨脹偵測</div>
                    <img class="video-stream" src="/area" alt="HSV/面積影像串流" width="640" height="480">
                </div>
                
                <div class="controls">
                    
                    <div class="control-group">
                        <div class="area-label">膨脹計算參數 (像素面積)</div>
                        
                        <div class="slider-group">
                            <label for="Ref_Area">參考面積 (Ref_Area): <span id="Ref_Area_val">{area_params['Ref_Area']}</span> px²</label>
                            <input type="range" id="Ref_Area" min="100" max="50000" step="100" value="{area_params['Ref_Area']}" oninput="updateAreaParams()">
                        </div>
                        
                        <div class="slider-group">
                            <label for="Max_Area">最大面積 (Max_Area): <span id="Max_Area_val">{area_params['Max_Area']}</span> px²</label>
                            <input type="range" id="Max_Area" min="100" max="100000" step="100" value="{area_params['Max_Area']}" oninput="updateAreaParams()">
                        </div>
                    </div>
                    
                    <div class="control-group">
                        <div class="hsv-label">HSV 顏色閾值調整</div>

                        <div class="slider-group">
                            <label for="H_min">H 最小值 (0-180): <span id="H_min_val">{hsv_params['H_min']}</span></label>
                            <input type="range" id="H_min" min="0" max="180" value="{hsv_params['H_min']}" oninput="updateHSV()">
                        </div>
                        
                        <div class="slider-group">
                            <label for="H_max">H 最大值 (0-180): <span id="H_max_val">{hsv_params['H_max']}</span></label>
                            <input type="range" id="H_max" min="0" max="180" value="{hsv_params['H_max']}" oninput="updateHSV()">
                        </div>

                        <div class="slider-group">
                            <label for="S_min">S 最小值 (0-255): <span id="S_min_val">{hsv_params['S_min']}</span></label>
                            <input type="range" id="S_min" min="0" max="255" value="{hsv_params['S_min']}" oninput="updateHSV()">
                        </div>

                        <div class="slider-group">
                            <label for="S_max">S 最大值 (0-255): <span id="S_max_val">{hsv_params['S_max']}</span></label>
                            <input type="range" id="S_max" min="0" max="255" value="{hsv_params['S_max']}" oninput="updateHSV()">
                        </div>

                        <div class="slider-group">
                            <label for="V_min">V 最小值 (0-255): <span id="V_min_val">{hsv_params['V_min']}</span></label>
                            <input type="range" id="V_min" min="0" max="255" value="{hsv_params['V_min']}" oninput="updateHSV()">
                        </div>
                        
                        <div class="slider-group">
                            <label for="V_max">V 最大值 (0-255): <span id="V_max_val">{hsv_params['V_max']}</span></label>
                            <input type="range" id="V_max" min="0" max="255" value="{hsv_params['V_max']}" oninput="updateHSV()">
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // ... (JavaScript 保持不變，因為它們只處理參數更新，與影像顯示無關)
            
            let hsvDebounceTimer;
            let areaDebounceTimer;

            function sendUpdate(endpoint, data) {{
                fetch(endpoint, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify(data),
                }})
                .then(response => response.json())
                .then(result => {{
                    if (!result.success) {{
                        console.error(`更新 ${{endpoint}} 參數失敗:`, result.message);
                    }}
                }})
                .catch(error => {{
                    console.error('Error:', error);
                }});
            }}

            function updateHSV() {{
                clearTimeout(hsvDebounceTimer);
                
                document.getElementById('H_min_val').textContent = document.getElementById('H_min').value;
                document.getElementById('S_min_val').textContent = document.getElementById('S_min').value;
                document.getElementById('V_min_val').textContent = document.getElementById('V_min').value;
                document.getElementById('H_max_val').textContent = document.getElementById('H_max').value;
                document.getElementById('S_max_val').textContent = document.getElementById('S_max').value;
                document.getElementById('V_max_val').textContent = document.getElementById('V_max').value;

                hsvDebounceTimer = setTimeout(() => {{
                    const data = {{
                        H_min: document.getElementById('H_min').value,
                        S_min: document.getElementById('S_min').value,
                        V_min: document.getElementById('V_min').value,
                        H_max: document.getElementById('H_max').value,
                        S_max: document.getElementById('S_max').value,
                        V_max: document.getElementById('V_max').value
                    }};
                    sendUpdate('/hsv_update', data);
                }}, 50); 
            }}
            
            function updateAreaParams() {{
                clearTimeout(areaDebounceTimer);
                
                const refSlider = document.getElementById('Ref_Area');
                const maxSlider = document.getElementById('Max_Area');
                
                let refVal = parseInt(refSlider.value);
                let maxVal = parseInt(maxSlider.value);

                if (maxVal < refVal) {{
                    maxSlider.value = refVal;
                    maxVal = refVal;
                }}

                document.getElementById('Ref_Area_val').textContent = refVal;
                document.getElementById('Max_Area_val').textContent = maxVal;

                areaDebounceTimer = setTimeout(() => {{
                    const data = {{
                        Ref_Area: refVal,
                        Max_Area: maxVal
                    }};
                    sendUpdate('/area_update', data);
                }}, 50); 
            }}


            document.addEventListener('DOMContentLoaded', () => {{
                
                // H 聯動邏輯
                document.getElementById('H_max').addEventListener('input', (e) => {{
                    const minVal = parseInt(document.getElementById('H_min').value);
                    if (parseInt(e.target.value) < minVal) {{
                        e.target.value = minVal;
                    }}
                    updateHSV();
                }});

                document.getElementById('H_min').addEventListener('input', (e) => {{
                    const maxVal = parseInt(document.getElementById('H_max').value);
                    if (parseInt(e.target.value) > maxVal) {{
                        document.getElementById('H_max').value = e.target.value;
                        document.getElementById('H_max_val').textContent = e.target.value;
                    }}
                    updateHSV();
                }});
                
                // 面積聯動邏輯
                document.getElementById('Max_Area').addEventListener('input', (e) => {{
                    const refVal = parseInt(document.getElementById('Ref_Area').value);
                    if (parseInt(e.target.value) < refVal) {{
                        e.target.value = refVal;
                    }}
                    updateAreaParams();
                }});

                document.getElementById('Ref_Area').addEventListener('input', (e) => {{
                    const maxVal = parseInt(document.getElementById('Max_Area').value);
                    if (parseInt(e.target.value) > maxVal) {{
                        document.getElementById('Max_Area').value = e.target.value;
                        document.getElementById('Max_Area_val').textContent = e.target.value;
                    }}
                    updateAreaParams();
                }});

                
                updateHSV(); 
                updateAreaParams();
            }});
        </script>
    </body>
    </html>
    """
    
# ====================================================================
# 主運行區
# ====================================================================

if __name__ == '__main__':
    # 啟動兩個獨立的執行緒來處理兩個攝影機擷取
    yolo_thread = threading.Thread(target=capture_frames_yolo, args=(2,))
    area_thread = threading.Thread(target=capture_frames_area, args=(0,)) # 注意這裡使用 Camera 1
    
    yolo_thread.daemon = True 
    area_thread.daemon = True 
    
    yolo_thread.start()
    area_thread.start()
    
    # 確保攝影機執行緒在程式結束時停止
    def shutdown_server():
        global is_running
        is_running = False
        # 等待兩個執行緒結束
        yolo_thread.join(timeout=1)
        area_thread.join(timeout=1)
        logging.info("Flask application shutting down and camera threads stopped.")

    import atexit
    atexit.register(shutdown_server)
    
    # Flask 必須在主執行緒中運行，使用指定的單一端口 (5002)
    try:
        app.run(host='0.0.0.0', port=5002, threaded=True)
    except Exception as e:
        logging.error(f"Flask run error: {e}")
        shutdown_server()
