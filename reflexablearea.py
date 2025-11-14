import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
import threading
import time

# --- 參數設定 ---

# --- HSV 全域變數（由前端滑塊控制） ---
# 預設值為咖啡色，但會被前端滑塊的初始值覆蓋
hsv_params = {
    'H_min': 10, 'S_min': 50, 'V_min': 100,
    'H_max': 30, 'S_max': 255, 'V_max': 255
}
hsv_lock = threading.Lock() # 用於多執行緒安全更新 HSV 參數

# --- 面積參數全域變數（由前端滑塊控制） ---
area_params = {
    'Ref_Area': 2500,  # 設定未膨脹的參考面積 (預設值)
    'Max_Area': 25000  # 設定完全膨脹的最大面積 (預設值)
}
area_lock = threading.Lock() # 用於多執行緒安全更新面積參數

# 全域變數和鎖 (用於多執行緒攝影機)
camera = None
frame_event = threading.Event()
latest_frame = None

def capture_frames():
    """獨立的執行緒，負責持續從攝影機讀取影像，解決多執行緒攝影機衝突問題。"""
    global latest_frame
    global camera

    # 必須在此執行緒中開啟攝影機
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("無法開啟攝影機，請檢查攝影機是否連接正確。") # 檢查攝影機狀態
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        
        latest_frame = frame.copy()
        frame_event.set() # 通知等待中的 generate_frames 有新畫面
        frame_event.clear()
        time.sleep(0.01) # 稍微延遲以避免 CPU 過載

def calculate_inflation_percentage(current_area):
    """
    根據當前面積，與參考面積和最大面積計算膨脹百分比。
    使用 area_params 中的全域參數。
    """
    with area_lock:
        ref_area = area_params['Ref_Area']
        max_area = area_params['Max_Area']
    
    # 確保 max_area > ref_area 才能進行有效計算
    if max_area - ref_area <= 0:
        return 0
    
    # 計算百分比
    percentage = (current_area - ref_area) / (max_area - ref_area) * 100
    
    # 將百分比限制在 0 到 100 之間
    return max(0, min(100, percentage))

def generate_frames():
    while True:
        frame_event.wait() # 等待新畫面
        
        if latest_frame is None:
            continue
        
        frame = latest_frame.copy()
        
        # --- 讀取最新的 HSV 參數 (使用 hsv_lock) ---
        with hsv_lock:
            LOWER_HSV = np.array([hsv_params['H_min'], hsv_params['S_min'], hsv_params['V_min']])
            UPPER_HSV = np.array([hsv_params['H_max'], hsv_params['S_max'], hsv_params['V_max']])
        
        # --- 影像處理 ---
        
        # 1. 轉換到 HSV 顏色空間
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 2. 建立顏色範圍的遮罩
        mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)
        
        # 3. 形態學操作：填補顏色區域中的孔洞並平滑輪廓
        # 使用較大的核尺寸來連接受損的輪廓
        kernel_morph = np.ones((7, 7), np.uint8) 
        # 使用 MORPH_CLOSE (關閉) 連接間隙
        processed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)
        # 使用 MORPH_OPEN (開啟) 移除小的顏色雜訊點
        processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_OPEN, kernel_morph)
        
        # 4. 尋找輪廓
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 處理與顯示 ---

        area = 0
        percentage = 0
        
        # 篩選輪廓：只考慮面積大於一定閾值的輪廓
        min_area_threshold = 500 # 假設袋子的最小面積
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area_threshold]

        if valid_contours:
            # 找到最大的有效輪廓
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

        # 傳輸 MJPEG 串流
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

app = Flask(__name__)

@app.route('/area')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/hsv_update', methods=['POST'])
def hsv_update():
    """接收來自前端的 HSV 參數並更新全域變數。"""
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
        
        # 確保 min 不大於 max
        hsv_params['H_max'] = max(hsv_params['H_min'], hsv_params['H_max'])
        hsv_params['S_max'] = max(hsv_params['S_min'], hsv_params['S_max'])
        hsv_params['V_max'] = max(hsv_params['V_min'], hsv_params['V_max'])
        
    return jsonify(success=True)

@app.route('/area_update', methods=['POST'])
def area_update():
    """接收來自前端的面積參數 (Ref_Area 和 Max_Area) 並更新全域變數。"""
    global area_params
    data = request.get_json()
    
    with area_lock:
        try:
            # 使用 .get() 並確保轉換為 int
            new_ref = int(data.get('Ref_Area', area_params['Ref_Area']))
            new_max = int(data.get('Max_Area', area_params['Max_Area']))

            # 確保 Max_Area >= Ref_Area
            if new_max < new_ref:
                # 如果使用者設定 Max < Ref，讓 Max = Ref
                new_max = new_ref 
            
            # 確保值是正數 (面積不能是負數)
            area_params['Ref_Area'] = max(0, new_ref)
            area_params['Max_Area'] = max(0, new_max)
            
        except (TypeError, ValueError) as e:
            print(f"面積參數更新錯誤: {e}")
            return jsonify(success=False, message="Invalid area parameters"), 400
        
    return jsonify(success=True)


@app.route('/')
def index():
    # 根據目前的預設值渲染 HTML 介面
    global hsv_params
    global area_params
    
    # 注意：這裡使用 f-string 格式化，因此 HTML 內所有非 Python 變數的 `{}` 都必須使用 `{{}}` 轉義
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>夾鏈袋膨脹偵測 (HSV/面積 可調式)</title>
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                text-align: center;
                margin: 0;
                padding: 20px;
                background-color: #f0f4f8;
            }}
            .container {{
                max-width: 1200px;
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
            .video-stream {{
                flex: 1 1 640px;
                border: 4px solid #34495e;
                border-radius: 8px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
                max-width: 100%;
                height: auto;
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
            @media (max-width: 1000px) {{
                .main-content {{
                    flex-direction: column;
                    align-items: center;
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
            .hsv-label {{
                font-size: 1.2em;
                font-weight: bold;
                color: #2980b9;
                margin-bottom: 10px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
            }}
            .area-label {{
                font-size: 1.2em;
                font-weight: bold;
                color: #e67e22;
                margin-bottom: 10px;
                border-bottom: 2px solid #ecf0f1;
                padding-bottom: 5px;
            }}
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
        </style>
    </head>
    <body>
        <div class="container">
            <h1>夾鏈袋膨脹偵測 (HSV/面積 可調式)</h1>
            <p>請使用右側滑塊調整顏色和面積參數，以準確鎖定袋子的輪廓並計算膨脹百分比。</p>
            
            <div class="main-content">
                <!-- 影像串流 -->
                <img class="video-stream" src="/area" alt="影像串流" width="640" height="480">
                
                <div class="controls">
                    
                    <!-- 面積參數控制區 -->
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
                    
                    <!-- HSV 顏色控制區 -->
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
            // 定義緩衝計時器
            let hsvDebounceTimer;
            let areaDebounceTimer;

            // 通用的發送函數
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

            // --- 處理 HSV 參數更新 ---
            function updateHSV() {{
                clearTimeout(hsvDebounceTimer);
                
                // 立即更新顯示值
                document.getElementById('H_min_val').textContent = document.getElementById('H_min').value;
                document.getElementById('S_min_val').textContent = document.getElementById('S_min').value;
                document.getElementById('V_min_val').textContent = document.getElementById('V_min').value;
                document.getElementById('H_max_val').textContent = document.getElementById('H_max').value;
                document.getElementById('S_max_val').textContent = document.getElementById('S_max').value;
                document.getElementById('V_max_val').textContent = document.getElementById('V_max').value;

                // 緩衝發送
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
            
            // --- 處理面積參數更新 ---
            function updateAreaParams() {{
                clearTimeout(areaDebounceTimer);
                
                const refSlider = document.getElementById('Ref_Area');
                const maxSlider = document.getElementById('Max_Area');
                
                let refVal = parseInt(refSlider.value);
                let maxVal = parseInt(maxSlider.value);

                // 確保 Max_Area >= Ref_Area 的前端邏輯
                if (maxVal < refVal) {{
                    maxSlider.value = refVal;
                    maxVal = refVal;
                }}

                // 立即更新顯示值
                document.getElementById('Ref_Area_val').textContent = refVal;
                document.getElementById('Max_Area_val').textContent = maxVal;

                // 緩衝發送
                areaDebounceTimer = setTimeout(() => {{
                    const data = {{
                        Ref_Area: refVal,
                        Max_Area: maxVal
                    }};
                    sendUpdate('/area_update', data);
                }}, 50); 
            }}


            // 確保頁面加載時先發送一次初始參數，並處理 min/max 聯動
            document.addEventListener('DOMContentLoaded', () => {{
                // H 聯動邏輯 (確保 H_min <= H_max)
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
                
                // 面積聯動邏輯 (確保 Ref_Area <= Max_Area)
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

                
                // 第一次加載時發送初始值
                updateHSV(); 
                updateAreaParams();
            }});
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    # 啟動獨立的執行緒來處理攝影機擷取
    thread = threading.Thread(target=capture_frames)
    thread.daemon = True 
    thread.start()
    
    # 在樹莓派上執行時，將 host 設為 '0.0.0.0' 允許從外部訪問
    # 注意：Flask 必須在主執行緒中運行
    app.run(host='0.0.0.0', port=5002)
