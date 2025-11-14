import cv2
import numpy as np
import os

def get_target_contour_area(image_path, output_dir="output_images", image_prefix=""):
    """
    載入圖片，使用顏色閾值和 Canny 偵測，
    回傳目標物體（氣球）的最大輪廓面積，並將處理過程的影像存檔。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"無法讀取圖片: {image_path}")
        return 0

    # 確保輸出資料夾存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 產生檔名
    base_name = os.path.basename(image_path)
    output_base_name = f"{image_prefix}_{os.path.splitext(base_name)[0]}"

    # 儲存原始圖片
    cv2.imwrite(os.path.join(output_dir, f"{output_base_name}_original.png"), image)

    # 1. 顏色空間轉換到 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. 定義氣球的黃色範圍
    # 根據你的圖片，黃色在HSV中的範圍通常為H:20-40，但為了能抓到較亮的黃色，
    # 我將S和V的下限設定得稍微高一點。
    lower_hsv = np.array([20, 100, 100])
    upper_hsv = np.array([40, 255, 255])

    # 3. 建立遮罩
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    cv2.imwrite(os.path.join(output_dir, f"{output_base_name}_hsv_mask.png"), mask)

    # 4. 形態學操作來平滑遮罩（可選，但有助於去除雜訊）
    # 在背景乾淨的情況下，不進行形態學操作也能有不錯的結果。
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(os.path.join(output_dir, f"{output_base_name}_morph_mask.png"), mask)

    # 5. 將遮罩應用在原圖上
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)

    # 6. Canny 邊緣偵測
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(os.path.join(output_dir, f"{output_base_name}_canny_edges.png"), edges)

    # 7. 找輪廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"在 {image_path} 中沒有找到任何輪廓。")
        cv2.imwrite(os.path.join(output_dir, f"{output_base_name}_no_contour_result.png"), image)
        return 0

    # 8. 尋找最大輪廓
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # **儲存：繪製並顯示輪廓**
    contour_display_image = image.copy()
    cv2.drawContours(contour_display_image, [largest_contour], -1, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, f"{output_base_name}_detected_contour.png"), contour_display_image)

    return area

def calculate_inflation_percentage(ref_area, target_area):
    """
    根據參考面積與目標面積計算膨脹百分比
    """
    if ref_area == 0:
        return 0
    return ((target_area - ref_area) / ref_area) * 100

def main():
    # 設定圖片檔案名稱
    ref_image_path = '螢幕擷取畫面 2025-09-12 134729.png'  # 未膨脹
    target_image_path = '螢幕擷取畫面 2025-09-12 134709.png'  # 膨脹後
    
    output_directory = "output_images"

    # 偵測面積，並傳入前綴以區分
    ref_area = get_target_contour_area(ref_image_path, output_directory, image_prefix="ref")
    target_area = get_target_contour_area(target_image_path, output_directory, image_prefix="target")

    # 計算膨脹百分比
    inflation = calculate_inflation_percentage(ref_area, target_area)

    print(f"參考面積 (未膨脹): {ref_area:.2f}")
    print(f"目前面積 (膨脹後): {target_area:.2f}")
    print(f"膨脹百分比: {inflation:.2f}%")
    print(f"所有處理後的圖片已儲存在 ./{output_directory}/ 資料夾中。")

if __name__ == "__main__":
    main()