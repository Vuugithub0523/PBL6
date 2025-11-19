import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

# ==================== GPU CONFIGURATION ====================
print("="*70)
print("🔍 KIỂM TRA GPU CUDA")
print("="*70)

# Kiểm tra TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Kiểm tra GPU có sẵn không
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ Tìm thấy {len(gpus)} GPU:")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        # Cấu hình memory growth để không chiếm hết VRAM
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"   ✅ Đã bật memory growth cho GPU {i}")
        except RuntimeError as e:
            print(f"   ⚠️ Không thể set memory growth: {e}")
    
    # Hiển thị GPU hiện tại đang dùng
    print(f"\n🚀 Training sẽ sử dụng: {gpus[0].name}")
else:
    print("⚠️ KHÔNG tìm thấy GPU!")
    print("⚠️ Training sẽ chạy trên CPU (chậm hơn)")
    print("\n💡 Để sử dụng GPU, cần:")
    print("   1. Cài đặt CUDA Toolkit")
    print("   2. Cài đặt cuDNN")
    print("   3. Cài đặt tensorflow-gpu hoặc tensorflow>=2.0")

# Kiểm tra CUDA có được build không
print(f"\nCUDA Available: {tf.test.is_built_with_cuda()}")
print(f"GPU Available: {tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)}")

print("="*70 + "\n")
# ============================================================

# Đọc dữ liệu từ extract_landmarks.py (63 features + label)
print("📂 Đọc dữ liệu landmarks...")
train = pd.read_csv("landmarks_train.csv")
val = pd.read_csv("landmarks_val.csv")
test = pd.read_csv("landmarks_test.csv")

X_train, y_train = train.drop("label", axis=1).values, train["label"]
X_val, y_val = val.drop("label", axis=1).values, val["label"]
X_test, y_test = test.drop("label", axis=1).values, test["label"]

print(f"✅ Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Chuẩn hóa dữ liệu (quan trọng vì z coordinate chưa được chuẩn hóa)
print("\n🔧 Chuẩn hóa dữ liệu...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

le_classes_fixed = []
for c in le.classes_:
    if c == "dd":
        le_classes_fixed.append("đ")
    else:
        le_classes_fixed.append(c)
le.classes_ = np.array(le_classes_fixed)

num_classes = len(le.classes_)
print(f"✅ Số lớp: {num_classes}")
print(f"✅ Classes: {list(le.classes_)}")

# Xây dựng mô hình (phù hợp với 63 landmarks features)
print("\n🏗️ Xây dựng mô hình...")
model = models.Sequential([
    layers.Input((63,)),  # 21 landmarks × 3 tọa độ (x,y,z)
    
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    
    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.2),
    
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Callbacks để tối ưu training
print("\n⚙️ Thiết lập callbacks...")
cb = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        'best_vsl_landmarks_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Training
print("\n🚀 Bắt đầu training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=cb,
    verbose=1
)

# Đánh giá trên test set
print("\n📊 Đánh giá trên test set...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

# Classification Report
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, 
                          target_names=le.classes_,
                          digits=4))

# Confusion Matrix
print("\n📊 Tạo Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(max(10, num_classes), max(8, num_classes-2)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - VSL Hand Landmarks Classification', fontsize=14, pad=20)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix.png")
plt.close()

# Plot training history
print("\n📈 Tạo Training History Plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
print("✅ Saved: training_history.png")
plt.close()

# Lưu model và preprocessors
print("\n💾 Lưu model và preprocessors...")
model.save("vsl_landmarks_model.h5")
print("✅ Saved: vsl_landmarks_model.h5")

# Lưu LabelEncoder và Scaler (cần cho inference)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: label_encoder.pkl")
print("✅ Saved: scaler.pkl")

# Tổng kết
print("\n" + "="*60)
print("🎉 TRAINING HOÀN TẤT!")
print("="*60)
print(f"📊 Final Test Accuracy: {test_acc*100:.2f}%")
print(f"📊 Total Epochs Trained: {len(history.history['loss'])}")
print(f"📊 Number of Classes: {num_classes}")
print(f"📊 Classes: {list(le.classes_)}")
print("="*60)
