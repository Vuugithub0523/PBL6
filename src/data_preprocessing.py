"""
Data Preprocessing Pipeline cho VSL Dataset
Xử lý Skeleton, RGB và Depth data
"""

import numpy as np
import cv2
import os
from scipy import interpolate
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader


class SkeletonPreprocessor:
    """Xử lý dữ liệu skeleton"""
    
    def __init__(self, fixed_length=64):
        self.fixed_length = fixed_length
        
    def load_skeleton(self, txt_file: str) -> np.ndarray:
        """
        Load skeleton từ file .txt
        Format: 0:x:y:z|1:x:y:z|2:x:y:z|3:x:y:z|4:x:y:z|5:x:y:z|6:x:y:z|7:x:y:z|leftHand|rightHand
        Returns: (num_frames, 8, 3) - 8 joints, xyz coordinates
        """
        skeleton_data = []
        
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    parts = line.strip().split('|')
                    
                    # Format: parts[0-7] = 8 joints, parts[8-9] = hand states
                    # Cần ít nhất 8 parts cho 8 joints
                    if len(parts) < 8:
                        continue
                    
                    frame_joints = []
                    
                    # Parse 8 joints (index 0-7)
                    for i in range(8):
                        coords = parts[i].split(':')
                        # Format: joint_index:x:y:z (4 phần)
                        if len(coords) == 4:
                            try:
                                x, y, z = float(coords[1]), float(coords[2]), float(coords[3])
                                frame_joints.append([x, y, z])
                            except (ValueError, IndexError):
                                # Nếu parse lỗi, bỏ qua frame này
                                break
                        else:
                            # Format không đúng
                            break
                    
                    # Chỉ thêm nếu parse đủ 8 joints
                    if len(frame_joints) == 8:
                        skeleton_data.append(frame_joints)
        
        except Exception as e:
            print(f"Warning: Error loading {txt_file}: {e}")
            return np.array([[0.0]*3]*8).reshape(1, 8, 3).astype(np.float32)
        
        # Kiểm tra có dữ liệu không
        if len(skeleton_data) == 0:
            print(f"Warning: No valid frames in {txt_file}")
            return np.array([[0.0]*3]*8).reshape(1, 8, 3).astype(np.float32)
        
        return np.array(skeleton_data, dtype=np.float32)
    
    def normalize_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa skeleton:
        - Center về origin (neck joint)
        - Scale theo khoảng cách vai
        """
        if skeleton.ndim != 3 or skeleton.shape[1] != 8 or skeleton.shape[2] != 3:
            print(f"Warning: Invalid skeleton shape {skeleton.shape}")
            return skeleton
        
        T, V, C = skeleton.shape  # (frames, joints, coords)
        
        # 1. Center về neck (joint 1)
        neck_position = skeleton[:, 1:2, :]  # (T, 1, 3)
        skeleton_centered = skeleton - neck_position
        
        # 2. Scale theo khoảng cách giữa 2 vai
        left_shoulder = skeleton[:, 2, :]   # (T, 3)
        right_shoulder = skeleton[:, 3, :]  # (T, 3)
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder, axis=1)  # (T,)
        
        # Tránh chia cho 0
        shoulder_dist = np.maximum(shoulder_dist, 1e-6)
        scale_factor = shoulder_dist.reshape(-1, 1, 1)  # (T, 1, 1)
        
        skeleton_normalized = skeleton_centered / scale_factor
        
        return skeleton_normalized
    
    def interpolate_missing(self, skeleton: np.ndarray) -> np.ndarray:
        """Nội suy các frame bị missing (NotTracked)"""
        if skeleton.ndim != 3 or skeleton.shape[1] != 8 or skeleton.shape[2] != 3:
            print(f"Warning: Invalid skeleton shape for interpolation {skeleton.shape}")
            return skeleton
        
        T, V, C = skeleton.shape
        
        for v in range(V):
            for c in range(C):
                coords = skeleton[:, v, c]
                
                # Tìm các frame valid (không phải 0 hoặc NaN)
                valid_mask = ~np.isnan(coords) & (np.abs(coords) > 1e-6)
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) > 1:
                    # Interpolate
                    f = interpolate.interp1d(
                        valid_indices, 
                        coords[valid_indices],
                        kind='linear',
                        fill_value='extrapolate'
                    )
                    skeleton[:, v, c] = f(np.arange(T))
        
        return skeleton
    
    def temporal_alignment(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Chuẩn hóa về fixed_length frames
        - Nếu ngắn hơn: interpolate
        - Nếu dài hơn: downsample
        """
        T, V, C = skeleton.shape
        
        if T == self.fixed_length:
            return skeleton
        
        # Tạo indices mới
        old_indices = np.linspace(0, T-1, T)
        new_indices = np.linspace(0, T-1, self.fixed_length)
        
        # Interpolate cho mỗi joint và coordinate
        skeleton_aligned = np.zeros((self.fixed_length, V, C), dtype=np.float32)
        
        for v in range(V):
            for c in range(C):
                f = interpolate.interp1d(old_indices, skeleton[:, v, c], kind='linear')
                skeleton_aligned[:, v, c] = f(new_indices)
        
        return skeleton_aligned
    
    def add_features(self, skeleton: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Thêm các features:
        - Velocity: vận tốc chuyển động
        - Acceleration: gia tốc
        - Bone vectors: vector giữa các joints
        """
        T, V, C = skeleton.shape
        
        # Velocity (first derivative)
        velocity = np.zeros_like(skeleton)
        velocity[1:] = skeleton[1:] - skeleton[:-1]
        
        # Acceleration (second derivative)
        acceleration = np.zeros_like(skeleton)
        acceleration[1:] = velocity[1:] - velocity[:-1]
        
        # Bone vectors (cặp joints kết nối)
        bone_pairs = [
            (0, 1),  # Head - Neck
            (1, 2),  # Neck - Left Shoulder
            (1, 3),  # Neck - Right Shoulder
            (2, 4),  # Left Shoulder - Left Hand
            (3, 5),  # Right Shoulder - Right Hand
            (1, 6),  # Neck - Left Hip
            (1, 7),  # Neck - Right Hip
        ]
        
        bones = np.zeros((T, len(bone_pairs), C), dtype=np.float32)
        for i, (j1, j2) in enumerate(bone_pairs):
            bones[:, i, :] = skeleton[:, j2, :] - skeleton[:, j1, :]
        
        return {
            'skeleton': skeleton,
            'velocity': velocity,
            'acceleration': acceleration,
            'bones': bones
        }
    
    def preprocess(self, txt_file: str) -> Dict[str, np.ndarray]:
        """Pipeline đầy đủ"""
        # 1. Load
        skeleton = self.load_skeleton(txt_file)
        
        # 2. Interpolate missing
        skeleton = self.interpolate_missing(skeleton)
        
        # 3. Normalize
        skeleton = self.normalize_skeleton(skeleton)
        
        # 4. Temporal alignment
        skeleton = self.temporal_alignment(skeleton)
        
        # 5. Add features
        features = self.add_features(skeleton)
        
        return features


class VideoPreprocessor:
    """Xử lý RGB và Depth video"""
    
    def __init__(self, size=(224, 224), frames=64):
        self.size = size
        self.frames = frames
    
    def load_video(self, video_path: str) -> np.ndarray:
        """Load video và extract frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        return np.array(frames)
    
    def temporal_sample(self, video: np.ndarray) -> np.ndarray:
        """Sample video về fixed number of frames"""
        T = len(video)
        
        if T == self.frames:
            return video
        
        # Uniform sampling
        indices = np.linspace(0, T-1, self.frames, dtype=int)
        return video[indices]
    
    def preprocess_rgb(self, video_path: str) -> torch.Tensor:
        """
        Preprocess RGB video
        Returns: (C, T, H, W) tensor
        """
        # Load
        video = self.load_video(video_path)
        
        # Temporal sample
        video = self.temporal_sample(video)
        
        # Resize
        video_resized = []
        for frame in video:
            frame_resized = cv2.resize(frame, self.size)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            video_resized.append(frame_rgb)
        
        video_resized = np.array(video_resized, dtype=np.float32)  # (T, H, W, C)
        
        # Normalize ImageNet
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)
        video_normalized = (video_resized / 255.0 - mean) / std
        
        # Convert to tensor (C, T, H, W)
        video_tensor = torch.from_numpy(video_normalized).permute(3, 0, 1, 2)
        
        return video_tensor
    
    def preprocess_depth(self, video_path: str) -> torch.Tensor:
        """
        Preprocess Depth video
        Returns: (1, T, H, W) tensor
        """
        # Load
        video = self.load_video(video_path)
        
        # Convert to grayscale
        video_gray = []
        for frame in video:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video_gray.append(gray)
        
        video_gray = np.array(video_gray, dtype=np.float32)
        
        # Temporal sample
        video_sampled = self.temporal_sample(video_gray)
        
        # Resize
        video_resized = []
        for frame in video_sampled:
            frame_resized = cv2.resize(frame, self.size)
            video_resized.append(frame_resized)
        
        video_resized = np.array(video_resized, dtype=np.float32)  # (T, H, W)
        
        # Normalize [0, 1]
        video_normalized = video_resized / 255.0
        
        # Convert to tensor (1, T, H, W)
        video_tensor = torch.from_numpy(video_normalized).unsqueeze(0)
        
        return video_tensor


class VSLDataset(Dataset):
    """PyTorch Dataset cho VSL"""
    
    def __init__(
        self, 
        data_root: str,
        split_file: str,  # Path to train_split.json / val_split.json / test_split.json
        modality: str = 'skeleton',  # 'skeleton', 'rgb', 'depth', 'all'
        augment: bool = False
    ):
        self.data_root = data_root
        self.split_file = split_file
        self.modality = modality
        self.augment = augment
        
        # Preprocessors
        self.skeleton_preprocessor = SkeletonPreprocessor(fixed_length=64)
        self.video_preprocessor = VideoPreprocessor(size=(224, 224), frames=64)
        
        # Load sample list từ split file
        self.samples = self._load_samples_from_split()
        
        split_name = os.path.basename(split_file).replace('_split.json', '')
        print(f"Loaded {len(self.samples)} samples for {split_name}")
    
    def _load_samples_from_split(self) -> List[Tuple[str, str, int]]:
        """Load samples từ JSON split file"""
        import json
        
        with open(self.split_file, 'r', encoding='utf-8') as f:
            split_data = json.load(f)
        
        samples = []
        for label_name, sample_id, label_idx in split_data:
            label_dir = os.path.join(self.data_root, label_name)
            samples.append((label_dir, sample_id, label_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        label_dir, sample_id, label = self.samples[idx]
        
        data = {}
        
        # Load skeleton
        if self.modality in ['skeleton', 'all']:
            skeleton_file = os.path.join(label_dir, f"{sample_id}.txt")
            skeleton_features = self.skeleton_preprocessor.preprocess(skeleton_file)
            data['skeleton'] = torch.from_numpy(skeleton_features['skeleton'])
            data['velocity'] = torch.from_numpy(skeleton_features['velocity'])
        
        # Load RGB
        if self.modality in ['rgb', 'all']:
            rgb_file = os.path.join(label_dir, f"{sample_id}_color.avi")
            if os.path.exists(rgb_file):
                data['rgb'] = self.video_preprocessor.preprocess_rgb(rgb_file)
        
        # Load Depth
        if self.modality in ['depth', 'all']:
            depth_file = os.path.join(label_dir, f"{sample_id}_depth.avi")
            if os.path.exists(depth_file):
                data['depth'] = self.video_preprocessor.preprocess_depth(depth_file)
        
        data['label'] = torch.tensor(label, dtype=torch.long)
        
        return data


def create_dataloaders(
    data_root: str,
    batch_size: int = 16,
    num_workers: int = 4,
    modality: str = 'skeleton'
) -> Dict[str, DataLoader]:
    """
    Tạo train/val/test dataloaders từ split files
    
    Args:
        data_root: Đường dẫn tới dataset_root
        batch_size: Batch size
        num_workers: Số worker threads (0 cho Windows)
        modality: 'skeleton', 'rgb', 'depth', hoặc 'all'
    """
    
    # Tạo datasets từ split files
    train_dataset = VSLDataset(
        data_root=data_root,
        split_file='train_split.json',
        modality=modality,
        augment=True
    )
    
    val_dataset = VSLDataset(
        data_root=data_root,
        split_file='val_split.json',
        modality=modality,
        augment=False
    )
    
    test_dataset = VSLDataset(
        data_root=data_root,
        split_file='test_split.json',
        modality=modality,
        augment=False
    )
    
    # Tạo dataloaders
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


if __name__ == "__main__":
    # Test preprocessing
    data_root = r"D:\Hoctap\ki1nam4\PBL6\Dataset\D_VSL_Share\D_VSL_Share\dataset_root"
    
    # Test skeleton preprocessing
    skeleton_file = os.path.join(data_root, "1.An", "1.txt")
    preprocessor = SkeletonPreprocessor()
    features = preprocessor.preprocess(skeleton_file)
    
    print("Skeleton shape:", features['skeleton'].shape)
    print("Velocity shape:", features['velocity'].shape)
    print("Bones shape:", features['bones'].shape)

