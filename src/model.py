"""
Model Architecture cho VSL Recognition
Bắt đầu với Skeleton-Only Model (Lightweight & Fast)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SkeletonEncoder(nn.Module):
    """
    Encode skeleton sequence
    Input: (batch, frames, joints, coords)
    Output: (batch, features)
    """
    
    def __init__(
        self,
        num_joints=8,
        in_channels=3,  # xyz
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()
        
        self.num_joints = num_joints
        self.in_channels = in_channels
        
        # Spatial feature extraction - Conv1D across joints
        self.spatial_conv = nn.Sequential(
            nn.Conv1d(num_joints * in_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            
            nn.Conv1d(64, hidden_dim, kernel_size=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Temporal feature extraction - BiLSTM
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, frames, joints, coords)
        Returns:
            features: (batch, hidden_dim * 2)
        """
        batch, frames, joints, coords = x.shape
        
        # Reshape for conv: (batch, joints*coords, frames)
        x = x.reshape(batch, frames, -1).permute(0, 2, 1).contiguous()
        
        # Spatial features
        x = self.spatial_conv(x)  # (batch, hidden_dim, frames)
        
        # Reshape for LSTM: (batch, frames, hidden_dim)
        x = x.permute(0, 2, 1)
        
        # Temporal features
        x, _ = self.temporal_lstm(x)  # (batch, frames, hidden_dim * 2)
        
        # Attention pooling
        x = self.attention(x)  # (batch, hidden_dim * 2)
        
        x = self.dropout(x)
        
        return x


class TemporalAttention(nn.Module):
    """Self-attention mechanism for temporal pooling"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, frames, hidden_dim)
        Returns:
            output: (batch, hidden_dim)
        """
        # Compute attention weights
        attention_weights = self.attention(x)  # (batch, frames, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        output = torch.sum(x * attention_weights, dim=1)  # (batch, hidden_dim)
        
        return output


class VSLSkeletonModel(nn.Module):
    """
    Skeleton-only model cho VSL Recognition
    Lightweight và nhanh, phù hợp cho realtime
    """
    
    def __init__(
        self,
        num_classes=27,
        num_joints=8,
        in_channels=3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.5
    ):
        super().__init__()
        
        self.encoder = SkeletonEncoder(
            num_joints=num_joints,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Classifier head
        feature_dim = hidden_dim * 2  # BiLSTM
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(feature_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, frames, joints, coords)
        Returns:
            logits: (batch, num_classes)
        """
        features = self.encoder(x)
        logits = self.classifier(features)
        return logits


class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, adjacency):
        super().__init__()
        
        self.adjacency = adjacency
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, frames, joints)
        Returns:
            output: (batch, out_channels, frames, joints)
        """
        # Graph convolution
        x = torch.einsum('nctv,vw->nctw', (x, self.adjacency))
        
        # 1x1 convolution
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block"""
    
    def __init__(self, in_channels, out_channels, adjacency, temporal_kernel=9, stride=1):
        super().__init__()
        
        # Spatial GCN
        self.gcn = GraphConvolution(in_channels, out_channels, adjacency)
        
        # Temporal Conv
        self.tcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(temporal_kernel, 1),
                stride=(stride, 1),
                padding=((temporal_kernel - 1) // 2, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual connection
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        x = x + res
        x = self.relu(x)
        return x


class STGCN(nn.Module):
    """
    Spatial-Temporal Graph Convolutional Network
    Mô hình mạnh hơn cho skeleton-based action recognition
    """
    
    def __init__(
        self,
        num_classes=27,
        num_joints=8,
        in_channels=3,
        graph_cfg=None,
        dropout=0.5
    ):
        super().__init__()
        
        # Define skeleton graph
        if graph_cfg is None:
            # Default: VSL 8-joint skeleton
            self.adjacency = self._build_adjacency_matrix(num_joints)
        else:
            self.adjacency = graph_cfg['adjacency']
        
        self.adjacency = torch.tensor(self.adjacency, dtype=torch.float32)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # ST-GCN blocks
        self.layers = nn.ModuleList([
            STGCNBlock(64, 64, self.adjacency, temporal_kernel=9),
            STGCNBlock(64, 128, self.adjacency, temporal_kernel=9),
            STGCNBlock(128, 128, self.adjacency, temporal_kernel=9),
            STGCNBlock(128, 256, self.adjacency, temporal_kernel=9),
        ])
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    
    def _build_adjacency_matrix(self, num_joints):
        """Build adjacency matrix for skeleton graph"""
        # Define edges (connections between joints)
        edges = [
            (0, 1),  # Head - Neck
            (1, 2),  # Neck - Left Shoulder
            (1, 3),  # Neck - Right Shoulder
            (2, 4),  # Left Shoulder - Left Hand
            (3, 5),  # Right Shoulder - Right Hand
            (1, 6),  # Neck - Left Hip
            (1, 7),  # Neck - Right Hip
        ]
        
        # Create adjacency matrix
        adjacency = torch.zeros(num_joints, num_joints)
        
        # Self-connections
        adjacency += torch.eye(num_joints)
        
        # Edge connections (bidirectional)
        for i, j in edges:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
        
        # Normalize
        degree = adjacency.sum(dim=1, keepdim=True)
        adjacency = adjacency / degree
        
        return adjacency.numpy()
    
    def forward(self, x):
        """
        Args:
            x: (batch, frames, joints, coords)
        Returns:
            logits: (batch, num_classes)
        """
        batch, frames, joints, coords = x.shape
        
        # Reshape: (batch, coords, frames, joints)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Move adjacency to same device
        if self.adjacency.device != x.device:
            self.adjacency = self.adjacency.to(x.device)
        
        # Input projection
        x = self.input_proj(x)
        
        # ST-GCN layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling
        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(batch, -1)    # (batch, 256)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


def create_model(model_type='skeleton_lstm', num_classes=27, **kwargs):
    """Factory function để tạo model"""
    
    if model_type == 'skeleton_lstm':
        return VSLSkeletonModel(
            num_classes=num_classes,
            num_joints=8,
            in_channels=3,
            hidden_dim=kwargs.get('hidden_dim', 128),
            num_layers=kwargs.get('num_layers', 2),
            dropout=kwargs.get('dropout', 0.5)
        )
    
    elif model_type == 'stgcn':
        return STGCN(
            num_classes=num_classes,
            num_joints=8,
            in_channels=3,
            dropout=kwargs.get('dropout', 0.5)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Đếm số parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    batch_size = 4
    frames = 64
    joints = 8
    coords = 3
    
    x = torch.randn(batch_size, frames, joints, coords)
    
    # Test Skeleton LSTM model
    print("=" * 50)
    print("Testing Skeleton LSTM Model")
    print("=" * 50)
    model_lstm = create_model('skeleton_lstm', num_classes=27)
    print(f"Parameters: {count_parameters(model_lstm):,}")
    
    output = model_lstm(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model size: ~{count_parameters(model_lstm) * 4 / 1024 / 1024:.2f} MB")
    
    # Test ST-GCN model
    print("\n" + "=" * 50)
    print("Testing ST-GCN Model")
    print("=" * 50)
    model_gcn = create_model('stgcn', num_classes=27)
    print(f"Parameters: {count_parameters(model_gcn):,}")
    
    output = model_gcn(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model size: ~{count_parameters(model_gcn) * 4 / 1024 / 1024:.2f} MB")

