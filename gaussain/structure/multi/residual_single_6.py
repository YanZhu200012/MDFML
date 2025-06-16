import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import copy
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F
import pickle
import os
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image, make_grid
from piqa import SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

def safe_item(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
        return tensor.item()
    return tensor

class MultiModalContaminantDataset(Dataset):
    def __init__(self, time_series, head_images, k_field_images, params):
        self.time_series = time_series
        self.head_images = head_images
        self.k_field_images = k_field_images
        self.params = params

    def __len__(self):
        return len(self.time_series)

    def __getitem__(self, idx):
        return self.time_series[idx], self.head_images[idx], self.k_field_images[idx], self.params[idx]

def create_multimodal_data_loaders(X_time_series, head_images, k_field_images, Y, batch_size=64, augment=False):
    min_samples = min(len(X_time_series), len(head_images), len(Y))
    X_time_series = X_time_series[:min_samples]
    head_images = head_images[:min_samples]
    k_field_images = k_field_images[:min_samples]
    Y = Y[:min_samples]
    full_dataset = MultiModalContaminantDataset(X_time_series, head_images, k_field_images, Y)

    def collate_fn(batch):
        btc_tensor = torch.stack([x[0] for x in batch]).unsqueeze(1).to(device)
        head_tensor = torch.stack([x[1] for x in batch]).to(device)
        k_field_tensor = torch.stack([x[2] for x in batch]).to(device)
        params_tensor = torch.stack([x[3] for x in batch]).to(device)
        return btc_tensor, head_tensor, k_field_tensor, params_tensor

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    loaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        'val': DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn),
        'test': DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    }
    return loaders['train'], loaders['val'], loaders['test']

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        self.feature_dim = hidden_dim * 2

    def forward(self, x):
        x = x.permute(0, 2, 1)
        outputs, _ = self.lstm(x)
        attn_weights = self.attention(outputs)
        context = torch.sum(attn_weights * outputs, dim=1)
        return context

class ResBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        self.feature_dim = feature_dim

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        return x6, [x1, x2, x3, x4, x5]

class KFieldDecoder(nn.Module):
    """K场解码器 - 保持与预训练模型相同的结构"""

    def __init__(self, out_channels=3, feature_dim=256):
        super().__init__()
        # === 原始解码器结构 - 确保与预训练模型完全兼容 ===
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )
        self.res1 = ResBlock(feature_dim + feature_dim, dropout_rate=0.2)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim + feature_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )
        self.res2 = ResBlock(256 + 256, dropout_rate=0.2)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )
        self.res3 = ResBlock(128 + 128, dropout_rate=0.2)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.res4 = ResBlock(64 + 64, dropout_rate=0.15)

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )
        self.res5 = ResBlock(32 + 32, dropout_rate=0.1)

        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # === 额外的后处理增强模块 - 不影响预训练权重加载 ===

        # 1. 细节增强模块 - 改进局部细节和纹理
        self.detail_enhancer = nn.Sequential(
            nn.Conv2d(out_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # 2. 边缘增强模块 - 改进结构边界
        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(out_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围为 -1 到 1
        )

        # 3. 通道注意力机制 - 增强重要通道
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 4. 后处理融合控制参数 - 可学习参数
        self.detail_weight = nn.Parameter(torch.tensor(0.2))
        self.edge_weight = nn.Parameter(torch.tensor(0.1))

        # 是否启用后处理增强
        self.use_enhancement = True

    def forward(self, x, skip_features, params=None):
        # === 保持原始解码器的前向传播逻辑不变 ===
        x1, x2, x3, x4, x5 = skip_features

        x = self.up1(x)
        x = torch.cat([x, x5], dim=1)
        x = self.res1(x)

        x = self.up2(x)
        x = torch.cat([x, x4], dim=1)
        x = self.res2(x)

        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.res3(x)

        x = self.up4(x)
        x = torch.cat([x, x2], dim=1)
        x = self.res4(x)

        x = self.up5(x)
        x = torch.cat([x, x1], dim=1)
        x = self.res5(x)

        x = self.up6(x)

        # 原始输出
        original_output = self.final(x)

        # 如果不需要增强，直接返回原始输出
        if not self.use_enhancement:
            return original_output

        # === 后处理增强步骤 ===

        # 1. 细节增强
        detail_enhanced = self.detail_enhancer(original_output)

        # 2. 边缘增强
        edge_map = self.edge_enhancer(original_output)

        # 3. 通道注意力
        channel_weights = self.channel_attention(original_output)
        attention_applied = original_output * channel_weights

        # 4. 融合所有增强结果
        # 将可学习权重限制在安全范围内
        detail_w = torch.sigmoid(self.detail_weight) * 0.4  # 最大值为0.4
        edge_w = torch.sigmoid(self.edge_weight) * 0.2  # 最大值为0.2

        # 融合公式: 原始 + 细节增强贡献 + 边缘增强贡献 + 注意力调整
        # 使用加权融合确保结果仍在0-1范围内
        base_weight = 1.0 - detail_w - edge_w
        enhanced_output = (base_weight * original_output +
                           detail_w * detail_enhanced +
                           edge_w * (original_output + 0.2 * edge_map).clamp(0, 1) +
                           0.1 * attention_applied)

        # 确保输出在有效范围内
        enhanced_output = enhanced_output.clamp(0, 1)

        return enhanced_output

    # 启用/禁用后处理增强的方法
    def set_enhancement(self, enable=True):
        self.use_enhancement = enable

class DualResidualFusionModule(nn.Module):
    def __init__(self, time_series_dim, image_dim, hidden_dim=64, task="image"):
        super().__init__()
        self.task = task
        self.time_series_proj = nn.Sequential(
            nn.Linear(time_series_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim)
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_series_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(0.5 if task == "image" else 0.1))

    def forward(self, time_series_features, image_features):
        if self.task == "param":
            # 投影图像特征
            image_proj = self.image_proj(image_features)
            alpha = torch.sigmoid(self.alpha) * 0.3
            # 规范化特征
            time_series_norm = F.normalize(time_series_features, p=2, dim=-1)
            image_proj_norm = F.normalize(image_proj, p=2, dim=-1)
            # 融合
            fused_features = time_series_norm + alpha * image_proj_norm
            # 计算占比
            time_series_contrib = torch.norm(time_series_norm, dim=-1) / torch.norm(fused_features, dim=-1)
            image_contrib = torch.norm(alpha * image_proj_norm, dim=-1) / torch.norm(fused_features, dim=-1)
        else:
            # 投影时间序列特征
            time_series_proj = self.time_series_proj(time_series_features)
            alpha = torch.sigmoid(self.alpha)
            # 规范化特征
            image_norm = F.normalize(image_features, p=2, dim=-1)
            time_series_proj_norm = F.normalize(time_series_proj, p=2, dim=-1)
            # 融合
            fused_features = image_norm + alpha * time_series_proj_norm
            # 计算占比
            image_contrib = torch.norm(image_norm, dim=-1) / torch.norm(fused_features, dim=-1)
            time_series_contrib = torch.norm(alpha * time_series_proj_norm, dim=-1) / torch.norm(fused_features, dim=-1)

        return {
            'fused_features': fused_features,
            'alpha': alpha,
            'time_series_contrib': time_series_contrib.mean(),  # 取批次均值
            'image_contrib': image_contrib.mean()              # 取批次均值
        }

class ParameterPredictionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        # 单一网络结构，直接输出全部6个参数
        self.param_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # 直接输出全部6个参数
        )

    def forward(self, features):
        # 直接预测全部参数
        all_params = self.param_net(features)
        return all_params

class PretrainedResidualFusionModel(nn.Module):
    def __init__(self, lstm_hidden_dim=128, cnn_feature_dim=256, lstm_path='best_specialized_model.pth',
                 cnn_path='checkpoints/best_model.pth', k_field_channels=3, img_size=256):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(input_dim=1, hidden_dim=lstm_hidden_dim)
        self.lstm_feature_dim = self.lstm_encoder.feature_dim
        self.cnn_encoder = CNNEncoder(in_channels=3, feature_dim=cnn_feature_dim)
        self.cnn_feature_dim = self.cnn_encoder.feature_dim
        self.param_fusion = DualResidualFusionModule(
            time_series_dim=self.lstm_feature_dim,
            image_dim=self.cnn_feature_dim * 4 * 4,
            task="param"
        )
        self.image_fusion = DualResidualFusionModule(
            time_series_dim=self.lstm_feature_dim,
            image_dim=self.cnn_feature_dim * 4 * 4,
            task="image"
        )
        self.param_network = ParameterPredictionNetwork(
            input_dim=self.lstm_feature_dim,
            hidden_dim=lstm_hidden_dim
        )
        self.k_field_decoder = KFieldDecoder(
            out_channels=k_field_channels,
            feature_dim=cnn_feature_dim
        )
        self._load_pretrained_weights(lstm_path, cnn_path)

    def _load_pretrained_weights(self, lstm_path, cnn_path):
        try:
            lstm_checkpoint = torch.load(lstm_path, map_location=device)
            lstm_state_dict = lstm_checkpoint.get('model_state_dict', lstm_checkpoint)
            lstm_encoder_dict = {k.split('encoder.')[1]: v for k, v in lstm_state_dict.items() if 'encoder' in k}
            missing, unexpected = self.lstm_encoder.load_state_dict(lstm_encoder_dict, strict=False)
            print(f"LSTM encoder loaded: {len(lstm_encoder_dict)} weights")
            if missing:
                print(f"Missing keys in LSTM: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected keys in LSTM: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            param_network_dict = {}
            for k, v in lstm_state_dict.items():
                if 'location_net' in k or 'release_net' in k:
                    param_network_dict[k] = v
            missing, unexpected = self.param_network.load_state_dict(param_network_dict, strict=False)
            print(f"Parameter network loaded: {len(param_network_dict)} weights")
            cnn_checkpoint = torch.load(cnn_path, map_location=device)
            cnn_state_dict = cnn_checkpoint.get('model_state_dict', cnn_checkpoint)
            cnn_encoder_dict = {k.split('encoder.')[1]: v for k, v in cnn_state_dict.items() if 'encoder' in k}
            missing, unexpected = self.cnn_encoder.load_state_dict(cnn_encoder_dict, strict=False)
            print(f"CNN encoder loaded: {len(cnn_encoder_dict)} weights")
            if missing:
                print(f"Missing keys in CNN: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected keys in CNN: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            decoder_dict = {k.split('decoder.')[1]: v for k, v in cnn_state_dict.items() if 'decoder' in k}
            missing, unexpected = self.k_field_decoder.load_state_dict(decoder_dict, strict=False)
            print(f"K-field decoder loaded: {len(decoder_dict)} weights")
            if missing:
                print(f"Missing keys in Decoder: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"Unexpected keys in Decoder: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Starting with randomly initialized weights")


    def forward(self, btc, head_img):
        lstm_features = self.lstm_encoder(btc)
        cnn_features, skip_features = self.cnn_encoder(head_img)
        cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)

        # 参数融合
        param_fusion_output = self.param_fusion(lstm_features, cnn_features_flat)
        param_features = param_fusion_output['fused_features']
        all_params = self.param_network(param_features)

        # 图像融合
        image_fusion_output = self.image_fusion(lstm_features, cnn_features_flat)
        image_features_flat = image_fusion_output['fused_features']
        image_features = image_features_flat.view(cnn_features.size(0), self.cnn_feature_dim, 4, 4)
        k_field = self.k_field_decoder(image_features, skip_features)

        return {
            'params': all_params,
            'k_field': k_field,
            'param_fusion_weight': param_fusion_output['alpha'],
            'image_fusion_weight': image_fusion_output['alpha'],
            'param_time_series_contrib': param_fusion_output['time_series_contrib'],
            'param_image_contrib': param_fusion_output['image_contrib'],
            'image_time_series_contrib': image_fusion_output['time_series_contrib'],
            'image_image_contrib': image_fusion_output['image_contrib']
        }

    def predict_params(self, btc, head_img=None):
        lstm_features = self.lstm_encoder(btc)
        if head_img is not None:
            cnn_features, _ = self.cnn_encoder(head_img)
            cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)
            param_features = self.param_fusion(lstm_features, cnn_features_flat)
        else:
            param_features = lstm_features
        return self.param_network(param_features)

    def generate_k_field(self, head_img, params=None, btc=None):
        cnn_features, skip_features = self.cnn_encoder(head_img)
        if btc is not None:
            lstm_features = self.lstm_encoder(btc)
            cnn_features_flat = cnn_features.view(cnn_features.size(0), -1)
            image_features_flat = self.image_fusion(lstm_features, cnn_features_flat)
            image_features = image_features_flat.view(cnn_features.size(0), self.cnn_feature_dim, 4, 4)
            if params is None:
                param_features = self.param_fusion(lstm_features, cnn_features_flat)
                params = self.param_network(param_features)
        else:
            image_features = cnn_features
        return self.k_field_decoder(image_features, skip_features)

def calculate_ssim(pred, target):
    pred_clamp = torch.clamp(pred, 0, 1)
    target_clamp = torch.clamp(target, 0, 1)
    ssim_metric = SSIM().to(device)
    return ssim_metric(pred_clamp, target_clamp)

def combined_loss_function(outputs, targets, k_field_target, loss_weights=None):
    if loss_weights is None:
        loss_weights = {'param': 0.7, 'image': 0.3, 'loc_weight': 3.0, 'rel_weight': 1.0}
    param_preds = outputs['params']
    param_weights = torch.tensor([loss_weights['loc_weight'], loss_weights['loc_weight'],
                                  loss_weights['rel_weight'], loss_weights['rel_weight'],
                                  loss_weights['rel_weight'], loss_weights['rel_weight']]).to(param_preds.device)
    squared_error = (param_preds - targets) ** 2
    weighted_squared_error = squared_error * param_weights.unsqueeze(0)
    param_loss = torch.mean(weighted_squared_error)
    k_field_pred = outputs['k_field']
    mse_loss = F.mse_loss(k_field_pred, k_field_target)
    l1_loss = F.l1_loss(k_field_pred, k_field_target)
    ssim_value = calculate_ssim(k_field_pred, k_field_target)
    ssim_loss = 1 - ssim_value
    image_loss = 0.5 * mse_loss + 0.2 * l1_loss + 0.3 * ssim_loss
    # image_loss = 0.5 * mse_loss + 0.3 * l1_loss + 0.2 * ssim_loss

    total_loss = loss_weights['param'] * param_loss + loss_weights['image'] * image_loss
    return {
        'total_loss': total_loss,
        'param_loss': param_loss,
        'image_loss': image_loss,
        'mse_loss': mse_loss,
        'l1_loss': l1_loss,
        'ssim': ssim_value
    }

def train_onestage_fusion_model(X_time_series, head_images, k_field_images, Y,
                                lstm_path='best_specialized_model.pth',
                                cnn_path='checkpoints/best_model.pth',
                                epochs=90, patience=30, batch_size=32):
    # 数据预处理
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_np = X_time_series.numpy()
    Y_np = Y.numpy()
    X_scaled = scaler_X.fit_transform(X_np)
    Y_scaled = scaler_Y.fit_transform(Y_np)
    X_time_series_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    Y_scaled = torch.tensor(Y_scaled, dtype=torch.float32)

    train_loader, val_loader, test_loader = create_multimodal_data_loaders(
        X_time_series_scaled, head_images, k_field_images, Y_scaled, batch_size=batch_size)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # 保存标准化器
    with open("checkpoints/scaler_X_6.pkl", "wb") as f:
        pickle.dump(scaler_X, f)
    with open("checkpoints/scaler_Y_6.pkl", "wb") as f:
        pickle.dump(scaler_Y, f)

    print("Standard scalers saved")

    # 获取数据维度
    _, sample_head, sample_k, _ = next(iter(train_loader))
    img_size = sample_head.shape[2]
    k_channels = sample_k.shape[1]
    h_channels = sample_head.shape[1]

    # 初始化模型
    model = PretrainedResidualFusionModel(
        lstm_hidden_dim=128,
        cnn_feature_dim=256,
        lstm_path=lstm_path,
        cnn_path=cnn_path,
        k_field_channels=k_channels,
        img_size=img_size
    ).to(device)

    # 训练历史记录
    history = {
        'train_total_loss': [], 'train_param_loss': [], 'train_image_loss': [],
        'val_total_loss': [], 'val_param_loss': [], 'val_image_loss': [],
        'lr': [], 'param_fusion_weight': [], 'image_fusion_weight': [],
        'param_time_series_contrib': [], 'param_image_contrib': [],
        'image_time_series_contrib': [], 'image_image_contrib': []
    }

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, verbose=True, min_lr=1e-6)

    # 损失权重
    loss_weights = {'param': 0.6, 'image': 0.4, 'loc_weight': 3.0, 'rel_weight': 1.0}

    best_val_loss = float('inf')
    best_model_state = None
    waiting = 0

    print("Starting one-stage training with all parameters unfrozen")

    def safe_item(tensor):
        if isinstance(tensor, torch.Tensor) and tensor.numel() == 1:
            return tensor.item()
        return tensor

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = {'total': 0.0, 'param': 0.0, 'image': 0.0}
        fusion_weights = {'param': 0.0, 'image': 0.0}
        contribs = {
            'param_ts': 0.0, 'param_img': 0.0,
            'img_ts': 0.0, 'img_img': 0.0
        }

        for btc, head_img, k_field, params in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            outputs = model(btc, head_img)
            losses = combined_loss_function(outputs, params, k_field, loss_weights)
            losses['total_loss'].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses['total'] += losses['total_loss'].item()
            train_losses['param'] += losses['param_loss'].item()
            train_losses['image'] += losses['image_loss'].item()

            fusion_weights['param'] += safe_item(outputs['param_fusion_weight'])
            fusion_weights['image'] += safe_item(outputs['image_fusion_weight'])

            contribs['param_ts'] += safe_item(outputs['param_time_series_contrib'])
            contribs['param_img'] += safe_item(outputs['param_image_contrib'])
            contribs['img_ts'] += safe_item(outputs['image_time_series_contrib'])
            contribs['img_img'] += safe_item(outputs['image_image_contrib'])

        avg_train_total = train_losses['total'] / len(train_loader)
        avg_train_param = train_losses['param'] / len(train_loader)
        avg_train_image = train_losses['image'] / len(train_loader)
        avg_param_weight = fusion_weights['param'] / len(train_loader)
        avg_image_weight = fusion_weights['image'] / len(train_loader)
        avg_param_ts_contrib = contribs['param_ts'] / len(train_loader)
        avg_param_img_contrib = contribs['param_img'] / len(train_loader)
        avg_img_ts_contrib = contribs['img_ts'] / len(train_loader)
        avg_img_img_contrib = contribs['img_img'] / len(train_loader)

        # 记录训练历史
        history['train_total_loss'].append(avg_train_total)
        history['train_param_loss'].append(avg_train_param)
        history['train_image_loss'].append(avg_train_image)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['param_fusion_weight'].append(avg_param_weight)
        history['image_fusion_weight'].append(avg_image_weight)
        history['param_time_series_contrib'].append(avg_param_ts_contrib)
        history['param_image_contrib'].append(avg_param_img_contrib)
        history['image_time_series_contrib'].append(avg_img_ts_contrib)
        history['image_image_contrib'].append(avg_img_img_contrib)

        # 验证阶段
        model.eval()
        val_losses = {'total': 0.0, 'param': 0.0, 'image': 0.0, 'ssim': 0.0}
        val_fusion_weights = {'param': 0.0, 'image': 0.0}
        val_contribs = {
            'param_ts': 0.0, 'param_img': 0.0,
            'img_ts': 0.0, 'img_img': 0.0
        }

        with torch.no_grad():
            for btc, head_img, k_field, params in val_loader:
                outputs = model(btc, head_img)
                losses = combined_loss_function(outputs, params, k_field, loss_weights)

                val_losses['total'] += losses['total_loss'].item()
                val_losses['param'] += losses['param_loss'].item()
                val_losses['image'] += losses['image_loss'].item()
                val_losses['ssim'] += safe_item(losses['ssim'])

                val_fusion_weights['param'] += safe_item(outputs['param_fusion_weight'])
                val_fusion_weights['image'] += safe_item(outputs['image_fusion_weight'])

                val_contribs['param_ts'] += safe_item(outputs['param_time_series_contrib'])
                val_contribs['param_img'] += safe_item(outputs['param_image_contrib'])
                val_contribs['img_ts'] += safe_item(outputs['image_time_series_contrib'])
                val_contribs['img_img'] += safe_item(outputs['image_image_contrib'])

        avg_val_total = val_losses['total'] / len(val_loader)
        avg_val_param = val_losses['param'] / len(val_loader)
        avg_val_image = val_losses['image'] / len(val_loader)
        avg_val_ssim = val_losses['ssim'] / len(val_loader)

        # 记录验证指标
        history['val_total_loss'].append(avg_val_total)
        history['val_param_loss'].append(avg_val_param)
        history['val_image_loss'].append(avg_val_image)

        # 更新学习率
        scheduler.step(avg_val_total)

        # 保存样本预测结果
        if epoch % 5 == 0:
            from torchvision.utils import make_grid, save_image
            val_batch = next(iter(val_loader))
            val_btc, val_head, val_k_true, val_params = val_batch
            with torch.no_grad():
                val_outputs = model(val_btc[:4], val_head[:4])
                val_k_pred = val_outputs['k_field']

            comparison = torch.cat([val_k_true[:4], val_k_pred], dim=0)
            comparison_grid = make_grid(comparison, nrow=4, normalize=True)
            save_image(comparison_grid, f'samples/6_epoch{epoch + 1}_comparison.png')

        # 保存最佳模型
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            best_model_state = copy.deepcopy(model.state_dict())

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'param_fusion_weight': safe_item(val_fusion_weights['param'] / len(val_loader)),
                'image_fusion_weight': safe_item(val_fusion_weights['image'] / len(val_loader))
            }, 'checkpoints/best_fusion_model_6.pth')

            print(f"New best model saved with val_loss: {avg_val_total:.4f}")
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 输出训练信息
        avg_val_param_weight = val_fusion_weights['param'] / len(val_loader)
        avg_val_image_weight = val_fusion_weights['image'] / len(val_loader)
        avg_val_param_ts_contrib = val_contribs['param_ts'] / len(val_loader)
        avg_val_param_img_contrib = val_contribs['param_img'] / len(val_loader)
        avg_val_img_ts_contrib = val_contribs['img_ts'] / len(val_loader)
        avg_val_img_img_contrib = val_contribs['img_img'] / len(val_loader)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {avg_train_total:.4f} (P:{avg_train_param:.4f}, I:{avg_train_image:.4f}) | "
            f"Val Loss: {avg_val_total:.4f} (P:{avg_val_param:.4f}, I:{avg_val_image:.4f}) | "
            f"Fusion Weights - Param: {avg_val_param_weight:.4f}, Image: {avg_val_image_weight:.4f} | "
            f"Param Contribs - TS: {avg_val_param_ts_contrib:.4f}, Img: {avg_val_param_img_contrib:.4f} | "
            f"Image Contribs - TS: {avg_val_img_ts_contrib:.4f}, Img: {avg_val_img_img_contrib:.4f} | "
            f"Val SSIM: {avg_val_ssim:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 评估模型
    evaluate_residual_fusion_model(model, test_loader, scaler_Y)
    plot_training_history(history)

    return model, history

def evaluate_residual_fusion_model(model, test_loader, scaler_Y):
    print("\nEvaluating on test set...")
    model.eval()
    test_preds, test_true = [], []
    test_loc_preds, test_loc_true = [], []
    test_rel_preds, test_rel_true = [], []
    image_metrics = {'mse': [], 'ssim': [], 'psnr': []}
    test_data = {'btc': [], 'head_img': [], 'k_field_true': [], 'params': []}  # 新增：保存测试数据

    with torch.no_grad():
        for btc, head_img, k_field_true, params in tqdm(test_loader, desc="Testing"):
            outputs = model(btc, head_img)
            pred = outputs['params']
            test_preds.append(pred.cpu().numpy())
            test_true.append(params.cpu().numpy())
            loc_pred = pred[:, :2]
            loc_true = params[:, :2]
            test_loc_preds.append(loc_pred.cpu().numpy())
            test_loc_true.append(loc_true.cpu().numpy())
            rel_pred = pred[:, 2:]
            rel_true = params[:, 2:]
            test_rel_preds.append(rel_pred.cpu().numpy())
            test_rel_true.append(rel_true.cpu().numpy())
            k_field_pred = outputs['k_field']
            mse = F.mse_loss(k_field_pred, k_field_true).item()
            ssim = safe_item(calculate_ssim(k_field_pred, k_field_true))
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()
            image_metrics['mse'].append(mse)
            image_metrics['ssim'].append(ssim)
            image_metrics['psnr'].append(psnr)
            # 保存测试数据
            test_data['btc'].append(btc.cpu().numpy())
            test_data['head_img'].append(head_img.cpu().numpy())
            test_data['k_field_true'].append(k_field_true.cpu().numpy())
            test_data['params'].append(params.cpu().numpy())

    # 保存测试数据集到文件
    test_data_arrays = {
        'btc': np.concatenate(test_data['btc']),
        'head_img': np.concatenate(test_data['head_img']),
        'k_field_true': np.concatenate(test_data['k_field_true']),
        'params': np.concatenate(test_data['params'])
    }
    os.makedirs("test_data", exist_ok=True)
    np.savez("test_data/test_dataset_6.npz", **test_data_arrays)
    print("Test dataset saved to 'test_data/test_dataset.npz'")

    # 反标准化预测和真实值
    test_preds_scaled = np.concatenate(test_preds)
    test_true_scaled = np.concatenate(test_true)
    test_preds = scaler_Y.inverse_transform(test_preds_scaled)
    test_true = scaler_Y.inverse_transform(test_true_scaled)
    test_loc_preds_scaled = np.concatenate(test_loc_preds)
    test_loc_true_scaled = np.concatenate(test_loc_true)
    test_rel_preds_scaled = np.concatenate(test_rel_preds)
    test_rel_true_scaled = np.concatenate(test_rel_true)
    loc_columns = [0, 1]
    rel_columns = [2, 3, 4, 5]
    test_loc_preds = np.zeros_like(test_loc_preds_scaled)
    test_loc_true = np.zeros_like(test_loc_true_scaled)
    test_rel_preds = np.zeros_like(test_rel_preds_scaled)
    test_rel_true = np.zeros_like(test_rel_true_scaled)
    for i, col in enumerate(loc_columns):
        mean = scaler_Y.mean_[col]
        std = scaler_Y.scale_[col]
        test_loc_preds[:, i] = test_loc_preds_scaled[:, i] * std + mean
        test_loc_true[:, i] = test_loc_true_scaled[:, i] * std + mean
    for i, col in enumerate(rel_columns):
        mean = scaler_Y.mean_[col]
        std = scaler_Y.scale_[col]
        test_rel_preds[:, i] = test_rel_preds_scaled[:, i] * std + mean
        test_rel_true[:, i] = test_rel_true_scaled[:, i] * std + mean

    # 计算评估指标
    r2_scores = [r2_score(test_true[:, i], test_preds[:, i]) for i in range(test_preds.shape[1])]
    mse_scores = [mean_squared_error(test_true[:, i], test_preds[:, i]) for i in range(test_preds.shape[1])]
    for i, (r2, mse) in enumerate(zip(r2_scores, mse_scores)):
        print(f"Parameter {i + 1}: R² = {r2:.4f}, MSE = {mse:.4f}")
    loc_r2 = r2_score(test_loc_true, test_loc_preds)
    loc_mse = mean_squared_error(test_loc_true, test_loc_preds)
    rel_r2 = r2_score(test_rel_true, test_rel_preds)
    rel_mse = mean_squared_error(test_rel_true, test_rel_preds)
    print(f"\nLocation Parameters: R² = {loc_r2:.4f}, MSE = {loc_mse:.4f}")
    print(f"Release Parameters: R² = {rel_r2:.4f}, MSE = {rel_mse:.4f}")
    avg_mse = np.mean(image_metrics['mse'])
    avg_ssim = np.mean(image_metrics['ssim'])
    avg_psnr = np.mean(image_metrics['psnr'])
    print(f"\nK-Field Generation Metrics:")
    print(f"MSE: {avg_mse:.4f}")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    overall_r2 = r2_score(test_true, test_preds)
    overall_mse = mean_squared_error(test_true, test_preds)
    overall_rmse = np.sqrt(overall_mse)
    print(f"\nOverall Parameter Metrics:")
    print(f"R²: {overall_r2:.4f}")
    print(f"MSE: {overall_mse:.4f}")
    print(f"RMSE: {overall_rmse:.4f}")
    test_metrics = {
        'r2': overall_r2, 'mse': overall_mse, 'rmse': overall_rmse,
        'loc_r2': loc_r2, 'loc_mse': loc_mse, 'rel_r2': rel_r2, 'rel_mse': rel_mse,
        'per_param_r2': r2_scores, 'per_param_mse': mse_scores,
        'image_mse': avg_mse, 'image_ssim': avg_ssim, 'image_psnr': avg_psnr
    }
    plot_parameter_predictions(test_true, test_preds, r2_scores)
    plot_sample_images(model, test_loader)
    return test_metrics

def plot_training_history(history):
    plt.figure(figsize=(18, 15))
    plt.subplot(3, 2, 1)
    plt.plot(history['train_total_loss'], label='Train Total')
    plt.plot(history['val_total_loss'], label='Val Total')
    plt.title('Total Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 2, 2)
    plt.plot(history['train_param_loss'], label='Train Param')
    plt.plot(history['val_param_loss'], label='Val Param')
    plt.plot(history['train_image_loss'], label='Train Image')
    plt.plot(history['val_image_loss'], label='Val Image')
    plt.title('Component Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(history['param_fusion_weight'], label='Parameter Fusion Weight')
    plt.plot(history['image_fusion_weight'], label='Image Fusion Weight')
    plt.title('Residual Fusion Weights')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    plt.legend()
    plt.subplot(3, 2, 4)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig('residual_fusion_training_history_6.png')
    plt.close()

def plot_parameter_predictions(test_true, test_preds, r2_scores):
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.scatter(test_true[:, i], test_preds[:, i], alpha=0.5)
        plt.plot([test_true[:, i].min(), test_true[:, i].max()],
                 [test_true[:, i].min(), test_true[:, i].max()], 'r--')
        plt.title(f'Parameter {i + 1}: R² = {r2_scores[i]:.4f}')
        plt.xlabel('True')
        plt.ylabel('Predicted')
    plt.tight_layout()
    plt.savefig('residual_fusion_parameter_predictions_6.png')
    plt.close()

def plot_sample_images(model, data_loader):
    model.eval()
    btc, head_img, k_true, params = next(iter(data_loader))
    batch_size = min(4, btc.shape[0])
    btc, head_img, k_true, params = btc[:batch_size], head_img[:batch_size], k_true[:batch_size], params[:batch_size]
    with torch.no_grad():
        outputs = model(btc, head_img)
        k_pred = outputs['k_field']
        image_fusion_weight = outputs['image_fusion_weight']
        plt.figure(figsize=(16, 4 * batch_size))
        for i in range(batch_size):
            plt.subplot(batch_size, 3, i * 3 + 1)
            if head_img[i].shape[0] > 3:
                img_to_show = head_img[i][:3].cpu()
            else:
                img_to_show = head_img[i].cpu()
            plt.imshow(img_to_show.permute(1, 2, 0))
            plt.title(f'Sample {i + 1}: Water Head')
            plt.axis('off')
            plt.subplot(batch_size, 3, i * 3 + 2)
            if k_true[i].shape[0] == 1:
                plt.imshow(k_true[i][0].cpu(), cmap='viridis')
            else:
                plt.imshow(k_true[i].cpu().permute(1, 2, 0))
            plt.title('Ground Truth K-Field')
            plt.axis('off')
            plt.subplot(batch_size, 3, i * 3 + 3)
            if k_pred[i].shape[0] == 1:
                plt.imshow(k_pred[i][0].cpu(), cmap='viridis')
            else:
                plt.imshow(k_pred[i].cpu().permute(1, 2, 0))
            fw_value = safe_item(image_fusion_weight)
            plt.title(f'Predicted K-Field (FW: {fw_value:.2f})')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('residual_fusion_sample_predictions_6.png')
        plt.close()

def main():
    print("Loading data...")
    try:
        a = np.loadtxt('Data_treat/BTCs_InterpNew.txt')
        b = np.loadtxt('Data_treat/DataGCSI.txt')
        X_time_series = torch.tensor(a, dtype=torch.float32)
        Y = torch.tensor(b, dtype=torch.float32)
        head_images = torch.load('processed_data/h-1.pt')
        k_field_images = torch.load('processed_data/k-1.pt')
        print(f"Loaded time series shape: {X_time_series.shape}")
        print(f"Loaded parameters shape: {Y.shape}")
        print(f"Loaded head images shape: {head_images.shape}")
        print(f"Loaded K field images shape: {k_field_images.shape}")
        model, history = train_onestage_fusion_model(
            X_time_series=X_time_series,
            head_images=head_images,
            k_field_images=k_field_images,
            Y=Y,
            lstm_path='best_specialized_model.pth',
            cnn_path='checkpoints/best_model.pth',
            epochs=50,
            patience=15,
            batch_size=32
        )
        print("Training completed!")
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        import traceback
        traceback.print_exc()
        print("Please ensure data paths are correct")

if __name__ == "__main__":
    main()