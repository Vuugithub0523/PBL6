"""
Dự đoán trên một sample cụ thể
"""

import torch
import sys
import codecs
import os
import argparse

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from model import create_model
from data_preprocessing import SkeletonPreprocessor


def predict_single_sample(model_path, skeleton_file, device='cuda'):
    """Dự đoán cho một skeleton file"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = create_model(
        model_type=checkpoint['config']['model_type'],
        num_classes=checkpoint['config']['num_classes'],
        hidden_dim=checkpoint['config'].get('hidden_dim', 128),
        num_layers=checkpoint['config'].get('num_layers', 2),
        dropout=checkpoint['config'].get('dropout', 0.5)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Label names
    label_names = [
        "An", "AnhHung", "HomQua", "DauVaiPhai", "Cam",
        "Chay", "TuDo", "CaiDat", "BenPhai", "TheDuc",
        "BuoiSang", "CauVong", "Bao", "CapCuu", "Lut",
        "MatTroi", "May", "Uong", "Cha", "Me", "Ong",
        "Chao", "TamBiet", "CamOn", "XinLoi", "BongDa", "BongChuyen"
    ]
    
    # Load and preprocess skeleton
    print(f"Loading skeleton: {skeleton_file}")
    preprocessor = SkeletonPreprocessor(fixed_length=64)
    features = preprocessor.preprocess(skeleton_file)
    
    skeleton_tensor = torch.from_numpy(features['skeleton']).unsqueeze(0).to(device)
    
    print(f"Skeleton shape: {skeleton_tensor.shape}")
    
    # Predict
    print("\nPredicting...")
    with torch.no_grad():
        output = model(skeleton_tensor)
        probs = torch.softmax(output, dim=1)
    
    # Get top-5 predictions
    top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
    
    top5_probs = top5_probs[0].cpu().numpy()
    top5_indices = top5_indices[0].cpu().numpy()
    
    # Print results
    print("\n" + "=" * 80)
    print("🎯 KẾT QUẢ DỰ ĐOÁN")
    print("=" * 80)
    
    print(f"\n🥇 Top-1: {label_names[top5_indices[0]]:15s} (Confidence: {top5_probs[0]*100:.2f}%)")
    
    print(f"\n📊 Top-5 predictions:")
    for i, (idx, prob) in enumerate(zip(top5_indices, top5_probs), 1):
        bar = '█' * int(prob * 50)
        print(f"   {i}. {label_names[idx]:15s} {prob*100:6.2f}% {bar}")
    
    print("\n" + "=" * 80)
    
    return label_names[top5_indices[0]], top5_probs[0]


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Dự đoán trên một skeleton file')
    parser.add_argument('--model', default='checkpoints/best_model.pth', help='Path to model')
    parser.add_argument('--file', required=True, help='Path to skeleton .txt file')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Không tìm thấy model: {args.model}")
        return
    
    if not os.path.exists(args.file):
        print(f"❌ Không tìm thấy file: {args.file}")
        return
    
    predict_single_sample(args.model, args.file, args.device)


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) == 1:
        print("=" * 80)
        print("📝 HƯỚNG DẪN SỬ DỤNG")
        print("=" * 80)
        print("\nCách sử dụng:")
        print("  python src/predict_single.py --file <path_to_skeleton_file>")
        print("\nVí dụ:")
        print('  python src/predict_single.py --file "dataset_root/1.An/1.txt"')
        print('  python src/predict_single.py --file "dataset_root/25.Chao/50.txt"')
        print("\nOptions:")
        print("  --model <path>   : Path to model (default: checkpoints/best_model.pth)")
        print("  --file <path>    : Path to skeleton .txt file (required)")
        print("  --device <dev>   : Device cuda/cpu (default: cuda)")
        print()
        
        # Test với một file mẫu
        test_file = r"D:\Hoctap\ki1nam4\PBL6\Dataset\D_VSL_Share\D_VSL_Share\dataset_root\1.An\1.txt"
        model_file = 'checkpoints/best_model.pth'
        
        if os.path.exists(test_file) and os.path.exists(model_file):
            print("🧪 Chạy test với file mẫu...")
            predict_single_sample(model_file, test_file)
        else:
            print("ℹ️  Để test, hãy chạy với --file argument")
    else:
        main()

