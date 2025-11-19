"""
Vẽ biểu đồ training history
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import codecs
import os

# Fix encoding
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def plot_training_history(history_file='checkpoints/history.json', save_path='training_curves.png'):
    """Vẽ biểu đồ training curves"""
    
    if not os.path.exists(history_file):
        print(f"❌ Không tìm thấy: {history_file}")
        return
    
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Loss curves
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Accuracy curves
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Target (90%)')
    
    # 3. Overfitting analysis
    ax3 = plt.subplot(2, 2, 3)
    gap = np.array(history['train_acc']) - np.array(history['val_acc'])
    ax3.plot(epochs, gap, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='Warning (5%)')
    ax3.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Overfitting (10%)')
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Train Acc - Val Acc (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.fill_between(epochs, 0, gap, where=(gap > 0), alpha=0.3, color='red')
    
    # 4. Statistics summary
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate stats
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    final_val_acc = history['val_acc'][-1]
    final_train_acc = history['train_acc'][-1]
    
    avg_train_acc = np.mean(history['train_acc'][-10:])  # Last 10 epochs
    avg_val_acc = np.mean(history['val_acc'][-10:])
    
    stats_text = f"""
    TRAINING STATISTICS
    {'='*50}
    
    Total Epochs:           {len(epochs)}
    
    Final Results:
      - Train Acc:          {final_train_acc:.2f}%
      - Val Acc:            {final_val_acc:.2f}%
      - Gap:                {final_train_acc - final_val_acc:.2f}%
    
    Best Val Accuracy:      {best_val_acc:.2f}%
      - At epoch:           {best_epoch}
    
    Average (Last 10 epochs):
      - Train Acc:          {avg_train_acc:.2f}%
      - Val Acc:            {avg_val_acc:.2f}%
    
    {'='*50}
    
    Status:
    """
    
    if final_val_acc >= 90:
        stats_text += "\n    ✅ EXCELLENT! Model đạt target!"
    elif final_val_acc >= 80:
        stats_text += "\n    ⚠️  GOOD, có thể cải thiện thêm"
    else:
        stats_text += "\n    ❌ Cần train lại với config khác"
    
    if abs(final_train_acc - final_val_acc) < 5:
        stats_text += "\n    ✅ No overfitting"
    elif abs(final_train_acc - final_val_acc) < 10:
        stats_text += "\n    ⚠️  Slight overfitting"
    else:
        stats_text += "\n    ❌ Overfitting detected!"
    
    ax4.text(0.1, 0.9, stats_text, fontsize=10, family='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved training curves to: {save_path}")
    
    # Show
    plt.show()
    
    return history


def main():
    """Main function"""
    
    print("=" * 80)
    print("📊 VISUALIZE TRAINING HISTORY")
    print("=" * 80)
    
    history_file = 'checkpoints/history.json'
    
    if not os.path.exists(history_file):
        print(f"\n❌ Không tìm thấy {history_file}")
        print("Vui lòng train model trước!")
        return
    
    history = plot_training_history(history_file)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📈 TRAINING SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal epochs: {len(history['train_loss'])}")
    print(f"Best val accuracy: {max(history['val_acc']):.2f}% (epoch {history['val_acc'].index(max(history['val_acc'])) + 1})")
    print(f"Final val accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final train accuracy: {history['train_acc'][-1]:.2f}%")
    
    # Learning progress
    print(f"\n📈 Learning progress:")
    milestones = [0, len(epochs)//4, len(epochs)//2, 3*len(epochs)//4, len(epochs)-1]
    
    for idx in milestones:
        if idx < len(history['val_acc']):
            print(f"   Epoch {idx+1:2d}: Val Acc = {history['val_acc'][idx]:.2f}%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

