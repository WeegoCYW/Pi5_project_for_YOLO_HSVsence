import os
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# 載入 .pt 模型
model = YOLO("y11n_batch4_e50_size320.pt")

# 類別名稱（應與模型訓練時一致）
class_names = ['chilli', 'ginger', 'pepper']

# 驗證資料路徑
image_dir = 'val_images'
label_dir = 'val_labels'

# 設定評估用的閾值
# 針對偵測的 IoU 閾值 (預測框與真實框重疊程度)
EVAL_IOU_THRESHOLD = 0.7 
# 針對偵測的置信度閾值 (只有高於此閾值的預測才被納入評估)
EVAL_CONF_THRESHOLD = 0.8 # 將這個值設定為你的目標效能閾值，YOLOv8 通常從 0.25 開始

def iou(box1, box2):
    # box format: [x1, y1, x2, y2]
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area else 0

def yolo_to_xyxy(label_parts, img_w, img_h):
    # label_parts: [class_id, center_x_norm, center_y_norm, width_norm, height_norm]
    cls, cx_norm, cy_norm, w_norm, h_norm = map(float, label_parts)
    x1 = (cx_norm - w_norm / 2) * img_w
    y1 = (cy_norm - h_norm / 2) * img_h
    x2 = (cx_norm + w_norm / 2) * img_w
    y2 = (cy_norm + h_norm / 2) * img_h
    return int(cls), [x1, y1, x2, y2]

# 初始化 TP, FP, FN 計數
total_tp = 0
total_fp = 0
total_fn = 0

# For displaying comparable raw output
first_image_detections = None
first_image_path = None # Store path of the first image

print(f"Loading .pt model: {model.model_name}")
try:
    input_size_pt = model.overrides['imgsz']
    if isinstance(input_size_pt, int):
        input_size_pt = (input_size_pt, input_size_pt)
    print(f"Model input size (from .pt model): {input_size_pt}")
except Exception:
    print("Could not determine input size from .pt model overrides.")

print("\nProcessing images for .pt model evaluation...")

# Get list of image files to process
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

for fname in tqdm(image_files):
    img_path = os.path.join(image_dir, fname)
    label_path = os.path.join(label_dir, fname.replace(os.path.splitext(fname)[1], '.txt'))

    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue
    ih, iw = image.shape[:2]

    # --- Step 1: Load Ground Truths ---
    ground_truths = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, bbox_xyxy = yolo_to_xyxy(parts, iw, ih)
                    ground_truths.append({'class_id': cls_id, 'bbox': bbox_xyxy, 'matched': False})
    
    # --- Step 2: Get Model Predictions ---
    # We set conf=EVAL_CONF_THRESHOLD directly here for the model's prediction output
    # This ensures YOLOv8's internal NMS and filtering are done at the desired evaluation threshold.
    results = model.predict(source=img_path, conf=EVAL_CONF_THRESHOLD, iou=0.7, verbose=False) # iou=0.7 for NMS

    # Store first image detections for display later
    if first_image_detections is None:
        first_image_detections = results[0].boxes.data.cpu().numpy()
        first_image_path = img_path # Store path for reference

    predictions = []
    if results and results[0].boxes:
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = det
            predictions.append({'class_id': int(class_id), 'bbox': [x1, y1, x2, y2], 'conf': conf})

    # --- Step 3: Match Predictions to Ground Truths and Count TP/FP/FN ---
    
    # Sort predictions by confidence in descending order for robust matching (optional but good practice)
    predictions = sorted(predictions, key=lambda x: x['conf'], reverse=True)

    current_image_tps = 0
    current_image_fps = 0
    
    for pred in predictions:
        pred_bbox = pred['bbox']
        pred_cls_id = pred['class_id']
        
        is_matched = False
        # Iterate through ground truths to find a match
        # Prioritize matching predictions to unmatched ground truths
        for gt_idx, gt in enumerate(ground_truths):
            if not gt['matched'] and pred_cls_id == gt['class_id']:
                current_iou = iou(pred_bbox, gt['bbox'])
                if current_iou >= EVAL_IOU_THRESHOLD:
                    current_image_tps += 1
                    ground_truths[gt_idx]['matched'] = True # Mark this ground truth as matched
                    is_matched = True
                    break # Move to the next prediction
        
        if not is_matched:
            current_image_fps += 1 # This prediction is a False Positive

    # Count False Negatives for the current image
    current_image_fns = 0
    for gt in ground_truths:
        if not gt['matched']:
            current_image_fns += 1
            
    total_tp += current_image_tps
    total_fp += current_image_fps
    total_fn += current_image_fns

# Calculate final metrics
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

print(f"\n📊 Evaluation Results for .pt Model (Evaluated at Conf={EVAL_CONF_THRESHOLD}, IoU={EVAL_IOU_THRESHOLD}):")
print(f"Total True Positives (TP): {total_tp}")
print(f"Total False Positives (FP): {total_fp}")
print(f"Total False Negatives (FN): {total_fn}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# === Add display for comparable raw output ===
# print("\n--- Comparable Raw Output (from .pt model with internal NMS/filter at evaluation threshold) ---")

# print("Model output details: (N/A for .pt models, shown for TPU comparison)")
# print("Index: N/A, Name: N/A, Shape: N/A, Dtype: N/A")
# print("Input scale: N/A, zero_point: N/A")
# print("Expected input shape: (1, 320, 320, 3) (Conceptual for comparison, actual input handled by YOLO)")

# print("\nFirst 5 predictions (from first image processed, after NMS and Conf Threshold):")
print("\nFirst 5 predictions (from first image processed):")
if first_image_detections is not None:
    # We want to display predictions that meet the EVAL_CONF_THRESHOLD
    displayed_count = 0
    for det in first_image_detections:
        x1, y1, x2, y2, conf, class_id = det
        
        if conf < EVAL_CONF_THRESHOLD: # Only display predictions that meet the evaluation's confidence criteria
            continue

        # Get original image dimensions for normalization
        original_img = cv2.imread(first_image_path)
        original_iw, original_ih = original_img.shape[1], original_img.shape[0]

        # Calculate normalized center_x, center_y, width, height for comparison
        center_x_norm = ((x1 + x2) / 2) / original_iw
        center_y_norm = ((y1 + y2) / 2) / original_ih
        width_norm = (x2 - x1) / original_iw
        height_norm = (y2 - y1) / original_ih

        box_display_norm = [center_x_norm, center_y_norm, width_norm, height_norm]
        
        # Simulated class scores: YOLOv8 already gives you the best confidence for the predicted class
        simulated_class_scores = np.zeros(len(class_names))
        simulated_class_scores[int(class_id)] = conf
        
        print(f"Box: [{box_display_norm[0]:.4f} {box_display_norm[1]:.4f} {box_display_norm[2]:.4f} {box_display_norm[3]:.4f}], "
              f"Confidence (Max Class Score): {conf:.4f}, "
              f"Class scores: {simulated_class_scores} (Simulated)")
        
        displayed_count += 1
        if displayed_count >= 5:
            break
    
    if displayed_count == 0:
        print(f"No predictions found meeting Conf={EVAL_CONF_THRESHOLD} for the first image.")
else:
    print("No images processed or no detections found for the first image.")