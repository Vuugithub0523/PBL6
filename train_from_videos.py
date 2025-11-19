"""
Training trực tiếp từ VIDEO (không qua file .txt)
→ Đảm bảo format skeleton giống khi test!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
import mediapipe as mp
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import codecs
import time
from tqdm import tqdm
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Fix encoding
if sys.platform == 'win32':
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
    except AttributeError:
        pass

sys.path.insert(0, 'src')
from model import create_model, count_parameters


# Label names (27 classes)
LABEL_NAMES = [
    "An", "AnhHung", "HomQua", "DauVaiPhai", "Cam",
    "Chay", "TuDo", "CaiDat", "BenPhai", "TheDuc",
    "BuoiSang", "CauVong", "Bao", "CapCuu", "Lut",
    "MatTroi", "May", "Uong", "Cha", "Me",
    "Ong", "Chao", "TamBiet", "CamOn", "XinLoi",
    "BongDa", "BongChuyen"
]


class VideoDataset(Dataset):
    """
    Dataset train trực tiếp từ VIDEO
    Extract skeleton bằng MediaPipe on-the-fly
    """
    
    def __init__(self, data_root, split_file, augment=False, fixed_length=60, cache_skeletons=True):
        """
        Args:
            data_root: Đường dẫn đến dataset_root (chứa video)
            split_file: File JSON split (train/val/test)
            augment: Có augment không
            fixed_length: Độ dài cố định của skeleton sequence
            cache_skeletons: Cache skeleton để tăng tốc (khuyến nghị True)
        """
        self.data_root = Path(data_root)
        self.augment = augment
        self.fixed_length = fixed_length
        self.cache_skeletons = cache_skeletons
        
        # MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        
        # 8 joints UPPER BODY (quay từ HÔNG LÊN - không bao gồm chân)
        # Mapping với file .txt gốc:
        # 0:Head, 1:Neck, 2:LShoulder, 3:RShoulder, 4:LElbow, 5:RElbow, 6:LWrist, 7:RWrist
        self.joint_indices = [
            0,   # NOSE (thay cho Head)
            0,   # NOSE lần 2 (thay cho Neck - vì MediaPipe không có Neck riêng)
            11,  # LEFT_SHOULDER
            12,  # RIGHT_SHOULDER
            13,  # LEFT_ELBOW
            14,  # RIGHT_ELBOW
            15,  # LEFT_WRIST
            16,  # RIGHT_WRIST
        ]
        self.num_output_joints = 8  # 8 joints upper body
        
        # Load split file
        with open(split_file, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        
        # Cache
        self.skeleton_cache = {}
        
        split_name = Path(split_file).stem.replace('_split_mediapipe', '')
        print(f"✅ Loaded {len(self.samples)} samples for {split_name}")
        
        # Pre-extract skeletons nếu cache enabled
        if self.cache_skeletons:
            print(f"⏳ Pre-extracting skeletons (cache enabled)...")
            self._preload_skeletons()
    
    def _preload_skeletons(self):
        """Pre-extract tất cả skeletons để cache"""
        for idx in tqdm(range(len(self.samples)), desc="Extracting skeletons"):
            class_name, sample_id, label = self.samples[idx]
            video_path = self.data_root / class_name / f"{sample_id}_color.avi"
            
            if video_path.exists() and str(video_path) not in self.skeleton_cache:
                skeleton = self.extract_skeleton_from_video(video_path)
                self.skeleton_cache[str(video_path)] = skeleton
    
    def extract_skeleton_from_video(self, video_path):
        """
        Extract skeleton từ video bằng MediaPipe
        GIỐNG HỆT với test_video_simple.py!
        """
        pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(str(video_path))
        skeletons = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                skeleton_frame = []
                landmarks = results.pose_landmarks.landmark
                
                # Extract 8 joints UPPER BODY
                for idx in self.joint_indices:
                    if idx < len(landmarks):
                        lm = landmarks[idx]
                        skeleton_frame.append([lm.x, lm.y, lm.z])
                
                if len(skeleton_frame) == 8:
                    skeletons.append(skeleton_frame)
        
        cap.release()
        pose.close()
        
        if len(skeletons) == 0:
            return np.zeros((self.fixed_length, 8, 3), dtype=np.float32)
        
        skeletons = np.array(skeletons, dtype=np.float32)
        
        # Normalize
        skeletons = self.normalize_skeleton(skeletons)
        
        # Interpolate to fixed length
        skeletons = self.interpolate_to_fixed_length(skeletons)
        
        return skeletons
    
    def normalize_skeleton(self, frames):
        """
        Normalize skeleton UPPER BODY:
        - Center by neck (joint 1)
        - Scale by shoulder width
        """
        if len(frames) == 0:
            return frames
        
        # Center by neck (joint 1)
        neck = frames[:, 1:2, :]  # (T, 1, 3)
        frames = frames - neck
        
        # Scale by shoulder width (distance between left and right shoulder)
        left_shoulder = frames[:, 2, :]   # (T, 3)
        right_shoulder = frames[:, 3, :]  # (T, 3)
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder, axis=-1, keepdims=True)  # (T, 1)
        shoulder_width = np.maximum(shoulder_width, 1e-6)
        
        frames = frames / shoulder_width[:, None, :]  # (T, 8, 3) / (T, 1, 1)
        
        return frames
    
    def interpolate_to_fixed_length(self, frames):
        """Interpolate skeleton sequence to fixed length"""
        T = len(frames)
        
        if T == 0:
            return np.zeros((self.fixed_length, 8, 3), dtype=np.float32)
        
        if T == self.fixed_length:
            return frames
        
        # Interpolate
        old_indices = np.linspace(0, T - 1, T)
        new_indices = np.linspace(0, T - 1, self.fixed_length)
        
        interpolated = []
        for joint_idx in range(8):
            for coord_idx in range(3):
                values = frames[:, joint_idx, coord_idx]
                interpolator = interp1d(old_indices, values, kind='linear', fill_value='extrapolate')
                interpolated.append(interpolator(new_indices))
        
        result = np.array(interpolated).reshape(3, 8, self.fixed_length).transpose(2, 1, 0)
        return result.astype(np.float32)
    
    def augment_skeleton(self, skeleton):
        """Augment skeleton UPPER BODY"""
        # 1. Horizontal flip (70%)
        if np.random.rand() < 0.7:
            skeleton = skeleton.copy()
            skeleton[:, :, 0] *= -1  # Flip X
            
            # Swap left-right (2↔3, 4↔5, 6↔7)
            skeleton[:, [2, 3], :] = skeleton[:, [3, 2], :]  # Shoulders
            skeleton[:, [4, 5], :] = skeleton[:, [5, 4], :]  # Elbows
            skeleton[:, [6, 7], :] = skeleton[:, [7, 6], :]  # Wrists
        
        # 2. Random scale (0.8-1.2)
        scale = np.random.uniform(0.8, 1.2)
        skeleton = skeleton * scale
        
        # 3. Random rotation around Y-axis (±20°)
        angle = np.random.uniform(-20, 20) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        skeleton = skeleton @ rotation_matrix.T
        
        # 4. Add noise
        noise = np.random.normal(0, 0.02, skeleton.shape).astype(np.float32)
        skeleton = skeleton + noise
        
        return skeleton
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        class_name, sample_id, label = self.samples[idx]
        
        # Video path
        video_path = self.data_root / class_name / f"{sample_id}_color.avi"
        
        # Load skeleton (from cache or extract)
        if str(video_path) in self.skeleton_cache:
            skeleton = self.skeleton_cache[str(video_path)].copy()
        else:
            skeleton = self.extract_skeleton_from_video(video_path)
        
        # Augment
        if self.augment:
            skeleton = self.augment_skeleton(skeleton)
        
        # To tensor
        skeleton_tensor = torch.from_numpy(skeleton).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return skeleton_tensor, label_tensor


def create_video_dataloaders(data_root, batch_size=16, num_workers=0):
    """Tạo dataloaders từ video"""
    
    print(f"\n⏳ Creating dataloaders from videos...")
    
    train_dataset = VideoDataset(
        data_root=data_root,
        split_file='train_split_mediapipe.json',
        augment=True,
        fixed_length=60,
        cache_skeletons=True  # ✅ BẬT cache: pre-extract 35 phút, sau đó nhanh
    )
    
    val_dataset = VideoDataset(
        data_root=data_root,
        split_file='val_split_mediapipe.json',
        augment=False,
        fixed_length=60,
        cache_skeletons=True  # ✅ BẬT cache
    )
    
    test_dataset = VideoDataset(
        data_root=data_root,
        split_file='test_split_mediapipe.json',
        augment=False,
        fixed_length=60,
        cache_skeletons=True  # ✅ BẬT cache
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


class Trainer:
    """Trainer cho model"""
    
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        # Loss & Optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.15))
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.001)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['num_epochs'],
            steps_per_epoch=config['steps_per_epoch'],
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Logging
        self.writer = SummaryWriter(config['log_dir'])
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Best model
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
        # History
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train một epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, (skeleton, labels) in enumerate(pbar):
            skeleton = skeleton.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            self.optimizer.zero_grad()
            outputs = self.model(skeleton)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Stats
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss/(batch_idx+1):.4f}",
                'acc': f"{100.*correct/total:.2f}%",
                'lr': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, dataloader, epoch):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for skeleton, labels in tqdm(dataloader, desc="Validating"):
                skeleton = skeleton.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(skeleton)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        avg_acc = 100. * correct / total
        
        return avg_loss, avg_acc, all_preds, all_labels
    
    def train(self, dataloaders):
        """Main training loop"""
        print(f"\n{'='*80}")
        print(f"🚀 BẮT ĐẦU TRAINING TỪ VIDEO")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_loss, train_acc = self.train_epoch(dataloaders['train'], epoch)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(dataloaders['val'], epoch)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/val', val_acc, epoch)
            self.writer.add_scalar('LR', self.scheduler.get_last_lr()[0], epoch)
            
            # Print summary
            print(f"\n📊 Epoch {epoch+1}/{self.config['num_epochs']} Summary:")
            print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
                best_path = self.checkpoint_dir / 'best_model.pth'
                torch.save(self.model.state_dict(), best_path)
                print(f"   ✅ Best model saved: {best_path} (Val Acc: {val_acc:.2f}%)")
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                ckpt_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(self.model.state_dict(), ckpt_path)
                print(f"   💾 Checkpoint saved: {ckpt_path}")
            
            print()
        
        end_time = time.time()
        
        print(f"{'='*80}")
        print(f"✅ HOÀN TẤT TRAINING")
        print(f"{'='*80}")
        print(f"⏱️  Total time: {(end_time - start_time)/60:.1f} minutes")
        print(f"🏆 Best Val Acc: {self.best_val_acc:.2f}% (Epoch {self.best_epoch+1})")
        print(f"{'='*80}\n")
        
        # Save history
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        self.writer.close()


def main():
    """Main function"""
    
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING TỪ VIDEO (KHÔNG QUA FILE .TXT)")
    print(f"   → Skeleton extract on-the-fly bằng MediaPipe")
    print(f"   → Đảm bảo giống format khi test!")
    print(f"{'='*80}\n")
    
    # Config
    config = {
        # Data
        'data_root': r'D:\Hoctap\ki1nam4\PBL6\Dataset\D_VSL_Share\D_VSL_Share\dataset_root',
        'batch_size': 16,
        'num_workers': 0,
        
        # Model
        'model_type': 'skeleton_lstm',
        'num_classes': 27,  # 27 classes (BỎ TaoHoa - chỉ có 16 videos)
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.5,
        
        # Training
        'num_epochs': 60,
        'learning_rate': 0.0005,
        'weight_decay': 0.001,
        'grad_clip': 1.0,
        'label_smoothing': 0.15,
        
        # Logging
        'log_dir': 'logs/vsl_from_video',
        'checkpoint_dir': 'checkpoints_from_video',
        'save_interval': 10,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {device}")
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    
    # Check split files
    split_files = ['train_split_mediapipe.json', 'val_split_mediapipe.json', 'test_split_mediapipe.json']
    if not all(os.path.exists(f) for f in split_files):
        print(f"\n❌ Chưa có split files!")
        print(f"Chạy: python split_mediapipe.py")
        return
    
    # Create dataloaders (extract skeleton from video)
    dataloaders = create_video_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    # Add steps_per_epoch to config
    config['steps_per_epoch'] = len(dataloaders['train'])
    
    # Create model
    print(f"\n⏳ Creating model: {config['model_type']}")
    model = create_model(
        model_type=config['model_type'],
        num_classes=config['num_classes'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    print(f"✅ Model parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(model, device, config)
    
    # Train
    trainer.train(dataloaders)
    
    print(f"\n📝 Bước tiếp theo:")
    print(f"   1. Test với video: python test_video_simple.py --model checkpoints_from_video/best_model.pth --video <path>")
    print(f"   2. Real-time: python realtime_webcam_improved.py --model checkpoints_from_video/best_model.pth")


if __name__ == "__main__":
    main()

