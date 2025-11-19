"""
Script đánh giá model sau training
Test độ chính xác, tạo confusion matrix, phân tích per-class accuracy
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import time
import json
import os
import sys
import codecs
from tqdm import tqdm

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from model import create_model, count_parameters
from data_preprocessing import create_dataloaders


class ModelEvaluator:
    """Class đánh giá model"""
    
    def __init__(self, model_path, data_root, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.data_root = data_root
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.config = checkpoint['config']
        self.model = create_model(
            model_type=self.config['model_type'],
            num_classes=self.config['num_classes'],
            hidden_dim=self.config.get('hidden_dim', 128),
            num_layers=self.config.get('num_layers', 2),
            dropout=self.config.get('dropout', 0.5)
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Thông tin model
        self.model_info = {
            'epoch': checkpoint['epoch'] + 1,
            'val_acc': checkpoint['val_acc'],
            'parameters': count_parameters(self.model)
        }
        
        # Label mapping
        self.label_names = [
            "An", "AnhHung", "HomQua", "DauVaiPhai", "Cam",
            "Chay", "TuDo", "CaiDat", "BenPhai", "TheDuc",
            "BuoiSang", "CauVong", "Bao", "CapCuu", "Lut",
            "MatTroi", "May", "Uong", "Cha", "Me", "Ong",
            "Chao", "TamBiet", "CamOn", "XinLoi", "BongDa", "BongChuyen"
        ]
        
        print(f"✅ Loaded model from epoch {self.model_info['epoch']}")
        print(f"✅ Val accuracy during training: {self.model_info['val_acc']:.2f}%")
        print(f"✅ Model parameters: {self.model_info['parameters']:,}")
    
    def evaluate(self, test_loader):
        """Đánh giá model trên test set"""
        
        print("\n" + "=" * 80)
        print("🧪 ĐÁNH GIÁ MODEL TRÊN TEST SET")
        print("=" * 80)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        inference_times = []
        
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Testing')
            
            for batch in pbar:
                skeleton = batch['skeleton'].to(self.device)
                labels = batch['label']
                
                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                outputs = self.model(skeleton)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                inference_times.append(end_time - start_time)
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs, inference_times
    
    def print_metrics(self, all_labels, all_preds):
        """In các metrics"""
        
        print("\n" + "=" * 80)
        print("📊 METRICS TỔNG QUAN")
        print("=" * 80)
        
        # Overall accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n🎯 Overall Accuracy: {accuracy * 100:.2f}%")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Macro average
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        print(f"\n📈 Macro Average:")
        print(f"   - Precision: {macro_precision * 100:.2f}%")
        print(f"   - Recall:    {macro_recall * 100:.2f}%")
        print(f"   - F1-Score:  {macro_f1 * 100:.2f}%")
        
        # Per-class details
        print("\n" + "=" * 80)
        print("📋 PER-CLASS METRICS")
        print("=" * 80)
        print()
        print(f"{'Class':>4} | {'Name':<15} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Support':>8}")
        print("-" * 80)
        
        for idx in range(len(self.label_names)):
            if idx < len(precision):
                print(f"{idx:4d} | {self.label_names[idx]:<15} | "
                      f"{precision[idx]*100:9.2f}% | {recall[idx]*100:9.2f}% | "
                      f"{f1[idx]*100:9.2f}% | {support[idx]:8d}")
        
        # Tìm classes tốt nhất và tệ nhất
        print("\n" + "-" * 80)
        
        best_5_idx = np.argsort(f1)[-5:][::-1]
        worst_5_idx = np.argsort(f1)[:5]
        
        print(f"\n🏆 Top 5 classes (highest F1-score):")
        for idx in best_5_idx:
            if idx < len(self.label_names):
                print(f"   {idx:2d}. {self.label_names[idx]:<15}: {f1[idx]*100:.2f}%")
        
        print(f"\n⚠️  Bottom 5 classes (lowest F1-score):")
        for idx in worst_5_idx:
            if idx < len(self.label_names):
                print(f"   {idx:2d}. {self.label_names[idx]:<15}: {f1[idx]*100:.2f}%")
    
    def plot_confusion_matrix(self, all_labels, all_preds, save_path='confusion_matrix.png'):
        """Vẽ confusion matrix"""
        
        print("\n" + "=" * 80)
        print("📊 CONFUSION MATRIX")
        print("=" * 80)
        
        cm = confusion_matrix(all_labels, all_preds)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
        
        # Plot
        plt.figure(figsize=(16, 14))
        
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to: {save_path}")
        
        # Tìm confused pairs
        print("\n🔍 Most confused pairs:")
        
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))
        
        # Sort by count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, j, count in confused_pairs[:10]:
            if i < len(self.label_names) and j < len(self.label_names):
                print(f"   {self.label_names[i]:<15} → {self.label_names[j]:<15}: {count:3d} lần")
    
    def analyze_inference_speed(self, inference_times):
        """Phân tích tốc độ inference"""
        
        print("\n" + "=" * 80)
        print("⚡ INFERENCE SPEED")
        print("=" * 80)
        
        avg_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        
        # FPS calculation (batch-based)
        avg_fps = 1.0 / avg_time
        
        print(f"\nPer-batch statistics:")
        print(f"   - Average time: {avg_time*1000:.2f} ms")
        print(f"   - Std dev:      {std_time*1000:.2f} ms")
        print(f"   - Min time:     {min_time*1000:.2f} ms")
        print(f"   - Max time:     {max_time*1000:.2f} ms")
        print(f"   - Batches/sec:  {avg_fps:.2f}")
        
        # Per-sample estimation (assuming batch_size=16)
        batch_size = 16
        per_sample_time = avg_time / batch_size
        sample_fps = 1.0 / per_sample_time
        
        print(f"\nPer-sample estimation:")
        print(f"   - Time/sample:  {per_sample_time*1000:.2f} ms")
        print(f"   - FPS:          {sample_fps:.2f}")
        
        # Realtime readiness
        if sample_fps >= 30:
            print(f"\n✅ Model đủ nhanh cho REALTIME (≥30 FPS)")
        elif sample_fps >= 15:
            print(f"\n⚠️  Model hơi chậm nhưng vẫn có thể dùng realtime (≥15 FPS)")
        else:
            print(f"\n❌ Model quá chậm cho realtime (< 15 FPS)")
    
    def save_results(self, results, save_path='test_results.json'):
        """Lưu kết quả đánh giá"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved results to: {save_path}")
    
    def run_full_evaluation(self):
        """Chạy đánh giá đầy đủ"""
        
        print("\n" + "=" * 80)
        print("🚀 BẮT ĐẦU ĐÁNH GIÁ MODEL")
        print("=" * 80)
        
        # Create test loader
        print("\nCreating test dataloader...")
        dataloaders = create_dataloaders(
            data_root=self.data_root,
            batch_size=16,
            num_workers=0,
            modality='skeleton'
        )
        
        test_loader = dataloaders['test']
        print(f"Test set: {len(test_loader.dataset)} samples")
        
        # Evaluate
        all_preds, all_labels, all_probs, inference_times = self.evaluate(test_loader)
        
        # Print metrics
        self.print_metrics(all_labels, all_preds)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(all_labels, all_preds)
        
        # Analyze speed
        self.analyze_inference_speed(inference_times)
        
        # Calculate additional metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        # Top-k accuracy
        top5_correct = 0
        for i, label in enumerate(all_labels):
            top5_preds = np.argsort(all_probs[i])[-5:]
            if label in top5_preds:
                top5_correct += 1
        
        top5_accuracy = top5_correct / len(all_labels)
        
        print(f"\n📊 Top-5 Accuracy: {top5_accuracy * 100:.2f}%")
        
        # Prepare results
        results = {
            "model_info": {
                "model_type": self.config['model_type'],
                "parameters": self.model_info['parameters'],
                "epoch": self.model_info['epoch'],
                "val_acc_during_training": self.model_info['val_acc']
            },
            "test_metrics": {
                "accuracy": float(accuracy * 100),
                "top5_accuracy": float(top5_accuracy * 100),
                "macro_precision": float(precision * 100),
                "macro_recall": float(recall * 100),
                "macro_f1": float(f1 * 100)
            },
            "inference_speed": {
                "avg_time_ms": float(np.mean(inference_times) * 1000),
                "fps": float(1.0 / (np.mean(inference_times) / 16))
            }
        }
        
        # Save results
        self.save_results(results, 'test_results.json')
        
        print("\n" + "=" * 80)
        print("✅ ĐÁNH GIÁ HOÀN TẤT!")
        print("=" * 80)
        
        return results


def main():
    """Main evaluation function"""
    
    # Paths
    model_path = 'checkpoints/best_model.pth'
    data_root = r"D:\Hoctap\ki1nam4\PBL6\Dataset\D_VSL_Share\D_VSL_Share\dataset_root"
    
    # Check model exists
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model: {model_path}")
        print("Vui lòng train model trước!")
        return
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model_path=model_path,
        data_root=data_root,
        device=device
    )
    
    # Run evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    print("\n" + "=" * 80)
    print("📝 TÓM TẮT KẾT QUẢ")
    print("=" * 80)
    print(f"\n🎯 Test Accuracy:  {results['test_metrics']['accuracy']:.2f}%")
    print(f"🎯 Top-5 Accuracy: {results['test_metrics']['top5_accuracy']:.2f}%")
    print(f"⚡ Inference FPS:  {results['inference_speed']['fps']:.2f}")
    
    # Compare with training
    val_acc = results['model_info']['val_acc_during_training']
    test_acc = results['test_metrics']['accuracy']
    
    print(f"\n📊 So sánh Val vs Test:")
    print(f"   - Val Acc:  {val_acc:.2f}%")
    print(f"   - Test Acc: {test_acc:.2f}%")
    
    if abs(val_acc - test_acc) < 3:
        print(f"   ✅ Model khá ổn định (chênh lệch < 3%)")
    elif test_acc > val_acc:
        print(f"   ✅ Test tốt hơn Val (may mắn!)")
    else:
        diff = val_acc - test_acc
        if diff < 5:
            print(f"   ⚠️  Có dấu hiệu overfitting nhẹ (chênh {diff:.2f}%)")
        else:
            print(f"   ❌ Overfitting nghiêm trọng (chênh {diff:.2f}%)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

