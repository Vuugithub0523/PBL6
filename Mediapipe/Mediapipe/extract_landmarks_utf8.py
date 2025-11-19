# -*- coding: utf-8 -*-
import sys
import os
import csv
import cv2
import mediapipe as mp
from tqdm import tqdm

# Bắt buộc để Windows console và Python xử lý UTF-8
sys.stdout.reconfigure(encoding='utf-8')
os.environ["PYTHONUTF8"] = "1"

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks_from_dir(root_dir, csv_out):
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                           min_detection_confidence=0.5)
    rows = []

    # Duyệt tất cả lớp (A, B, C, đ, ...)
    for label in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir):
            continue
        print(f"\n🧩 Đang xử lý lớp: {label}")

        for fname in tqdm(os.listdir(class_dir), desc=f"Processing {label}", ncols=80):
            fpath = os.path.join(class_dir, fname)
            image = cv2.imread(fpath)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(image_rgb)
            if not result.multi_hand_landmarks:
                continue

            hand_landmarks = result.multi_hand_landmarks[0]
            data = []
            for lm in hand_landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            data.append(label)
            rows.append(data)

    # Lưu CSV với encoding UTF-8
    if len(rows) == 0:
        print("⚠️ Không có dữ liệu hợp lệ, kiểm tra lại thư mục nguồn!")
        return

    n_cols = len(rows[0]) - 1
    header = [f"x{i//3}_coord_{['x','y','z'][i%3]}" for i in range(n_cols)] + ["label"]

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n✅ Đã lưu file CSV: {csv_out}")
    print(f"   Tổng số mẫu: {len(rows)}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        extract_landmarks_from_dir(f"Split_Dataset/{split}", f"landmarks_{split}.csv")
