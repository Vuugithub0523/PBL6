#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script đơn giản để tạo file ZIP chứa source code
Chỉ nén các file được liệt kê cụ thể
"""

import os
import zipfile
from pathlib import Path

# Thư mục gốc của project
BASE_DIR = Path(__file__).parent

# Tên file ZIP output
ZIP_NAME = "Sourcecode.zip"

# Các file/thư mục CẦN NÉN (theo danh sách từ dòng 43-68)
FILES_TO_ZIP = [
    # 1. SOURCE CODE (.py)
    "unified_server.py",
    "Mediapipe.py",
    "outputTiengViet.py",
    "Train.py",
    "templates.py",
    
    # 2. CONFIG FILES
    "users.json",
    "client_api_keys.json",
    "sign_mapping.json",
    "selected_tags_names.txt",
    
    # 3. TEMPLATES & STATIC
    "templates/",  # Thư mục templates
    "static/",     # Thư mục static
    
    # 4. MODEL FILES
    "best_vsl_landmarks_model.h5",
    "vsl_landmarks_model.tflite",
    "label_encoder.pkl",
    "scaler.pkl",
]

def add_file_to_zip(zipf: zipfile.ZipFile, file_path: Path, base_dir: Path):
    """Thêm một file vào zip"""
    try:
        relative_path = file_path.relative_to(base_dir)
        zipf.write(file_path, relative_path)
        print(f"  ✅ {relative_path}")
        return True
    except Exception as e:
        print(f"  ⚠️  Lỗi: {file_path.name} - {e}")
        return False

def add_directory_to_zip(zipf: zipfile.ZipFile, dir_path: Path, base_dir: Path):
    """Thêm toàn bộ thư mục vào zip, loại bỏ cache files"""
    added_count = 0
    
    for root, dirs, files in os.walk(dir_path):
        root_path = Path(root)
        
        # Loại bỏ các thư mục không cần thiết
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.vscode']]
        
        for file in files:
            file_path = root_path / file
            
            # Loại bỏ cache files
            if file.endswith(('.pyc', '.pyo', '.pyd')) or file == '.DS_Store':
                continue
            
            if add_file_to_zip(zipf, file_path, base_dir):
                added_count += 1
    
    return added_count

def create_sourcecode_zip():
    """Tạo file ZIP chứa source code"""
    zip_path = BASE_DIR / ZIP_NAME
    
    print("=" * 70)
    print("📦 TẠO FILE ZIP SOURCE CODE")
    print("=" * 70)
    print(f"\n📁 Thư mục gốc: {BASE_DIR}")
    print(f"📦 File ZIP: {zip_path}")
    print(f"\n🔍 Bắt đầu nén các file...\n")
    
    total_files = 0
    missing_files = []
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in FILES_TO_ZIP:
            item_path = BASE_DIR / item
            
            if not item_path.exists():
                missing_files.append(item)
                print(f"  ⚠️  File không tồn tại: {item}")
                continue
            
            if item_path.is_file():
                # Thêm file đơn lẻ
                if add_file_to_zip(zipf, item_path, BASE_DIR):
                    total_files += 1
            
            elif item_path.is_dir():
                # Thêm toàn bộ thư mục
                print(f"📁 Đang nén thư mục: {item}")
                count = add_directory_to_zip(zipf, item_path, BASE_DIR)
                total_files += count
                print(f"  ✅ Đã thêm {count} file từ {item}\n")
    
    # Tính kích thước file
    if zip_path.exists():
        zip_size = zip_path.stat().st_size
        zip_size_mb = zip_size / (1024 * 1024)
        
        print("=" * 70)
        print("✅ HOÀN THÀNH!")
        print("=" * 70)
        print(f"📦 File ZIP: {zip_path}")
        print(f"📊 Tổng số file đã nén: {total_files}")
        print(f"📏 Kích thước: {zip_size_mb:.2f} MB ({zip_size:,} bytes)")
        
        if missing_files:
            print(f"\n⚠️  Các file không tìm thấy ({len(missing_files)}):")
            for f in missing_files:
                print(f"   - {f}")
        
        print("\n" + "=" * 70)
        print("📝 LƯU Ý:")
        print("=" * 70)
        print("✅ Đã loại bỏ:")
        print("   - hand_env/ (virtual environment)")
        print("   - __pycache__/ (Python cache)")
        print("   - .vscode/ (IDE config)")
        print("   - received_data/ (runtime data)")
        print("\n✅ File ZIP đã sẵn sàng để nộp cho giảng viên!")
    else:
        print("\n❌ Không thể tạo file ZIP!")

if __name__ == '__main__':
    try:
        create_sourcecode_zip()
    except KeyboardInterrupt:
        print("\n\n❌ Đã hủy bởi người dùng.")
    except Exception as e:
        print(f"\n\n❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()

