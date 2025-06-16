import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import os
from torchvision.utils import save_image, make_grid
from piqa import SSIM

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.manual_seed(42)


# 创建简单的数据集类
class ImageDataset(Dataset):
    def __init__(self, input_images, target_images):
        self.input_images = input_images
        self.target_images = target_images

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        return self.input_images[idx], self.target_images[idx]


# 定义残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

    def forward(self, x):
        return x + self.block(x)


# 定义编码器
class Encoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2)
        )  # 128x128

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )  # 64x64

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )  # 32x32

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2)
        )  # 16x16

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.MaxPool2d(2)
        )  # 8x8

        self.conv6 = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.MaxPool2d(2)
        )  # 4x4

    def forward(self, x):
        x1 = self.conv1(x)  # 128x128
        x2 = self.conv2(x1)  # 64x64
        x3 = self.conv3(x2)  # 32x32
        x4 = self.conv4(x3)  # 16x16
        x5 = self.conv5(x4)  # 8x8
        x6 = self.conv6(x5)  # 4x4

        # 返回各个层的特征，用于跳跃连接
        return x6, [x1, x2, x3, x4, x5]


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, out_channels=3, feature_dim=256):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, feature_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.GELU()
        )  # 8x8
        self.res1 = ResBlock(feature_dim + feature_dim)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(feature_dim + feature_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU()
        )  # 16x16
        self.res2 = ResBlock(256 + 256)

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()
        )  # 32x32
        self.res3 = ResBlock(128 + 128)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )  # 64x64
        self.res4 = ResBlock(64 + 64)

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )  # 128x128
        self.res5 = ResBlock(32 + 32)

        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU()
        )  # 256x256

        self.final = nn.Sequential(
            nn.Conv2d(16, out_channels, 3, padding=1),
            nn.Sigmoid()  # 输出范围 [0, 1]
        )

    def forward(self, x, skip_features):
        x1, x2, x3, x4, x5 = skip_features

        # 上采样和跳跃连接
        x = self.up1(x)  # 4x4 -> 8x8
        x = torch.cat([x, x5], dim=1)
        x = self.res1(x)

        x = self.up2(x)  # 8x8 -> 16x16
        x = torch.cat([x, x4], dim=1)
        x = self.res2(x)

        x = self.up3(x)  # 16x16 -> 32x32
        x = torch.cat([x, x3], dim=1)
        x = self.res3(x)

        x = self.up4(x)  # 32x32 -> 64x64
        x = torch.cat([x, x2], dim=1)
        x = self.res4(x)

        x = self.up5(x)  # 64x64 -> 128x128
        x = torch.cat([x, x1], dim=1)
        x = self.res5(x)

        x = self.up6(x)  # 128x128 -> 256x256

        # 最终卷积层
        output = self.final(x)

        return output


# 完整的图像处理模型
class ImageGenerationModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, feature_dim=256):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, feature_dim=feature_dim)
        self.decoder = Decoder(out_channels=out_channels, feature_dim=feature_dim)

    def forward(self, x):
        features, skip_features = self.encoder(x)
        output = self.decoder(features, skip_features)
        return output


# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)


    # 创建保存目录
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)

    best_val_loss = float('inf')

    # 训练历史记录
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for input_imgs, target_imgs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            # 前向传播
            outputs = model(input_imgs)
            loss = criterion(outputs, target_imgs)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # 验证阶段
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for input_imgs, target_imgs in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                input_imgs = input_imgs.to(device)
                target_imgs = target_imgs.to(device)

                outputs = model(input_imgs)
                loss = criterion(outputs, target_imgs)

                val_loss += loss.item()

                # 保存一些示例图像
                if epoch % 5 == 0 and val_loss == 0.0:  # 只在第一个batch保存
                    batch_size = min(4, input_imgs.size(0))
                    comparison = torch.cat([
                        target_imgs[:batch_size],
                        outputs[:batch_size]
                    ], dim=0)
                    comparison_grid = make_grid(comparison, nrow=batch_size, normalize=True)
                    save_image(comparison_grid, f'samples/epoch_{epoch + 1}_comparison.png')

        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        # 打印训练和验证损失
        print(f"Epoch {epoch + 1}/{num_epochs} - "
              f"Train Loss: {avg_train_loss:.6f}, "
              f"Val Loss: {avg_val_loss:.6f}, "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 如果验证损失更好，保存模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'checkpoints/best_model.pth')
            print(f"Model saved with val_loss: {best_val_loss:.6f}")

        # 更新学习率
        # scheduler.step(avg_val_loss)

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()

    return history


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0

    all_psnr = []
    all_ssim = []

    ssim_metric = SSIM().to(device)  # Move SSIM to the appropriate device

    with torch.no_grad():
        for input_imgs, target_imgs in tqdm(test_loader, desc="Testing"):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            outputs = model(input_imgs)
            loss = criterion(outputs, target_imgs)
            test_loss += loss.item()

            # 计算PSNR (Peak Signal-to-Noise Ratio)
            mse = F.mse_loss(outputs, target_imgs, reduction='mean').item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
            all_psnr.append(psnr)

            # 计算SSIM (需要自定义函数，这里省略)
            ssim_value = ssim_metric(outputs, target_imgs).item()
            all_ssim.append(ssim_value)

    avg_test_loss = test_loss / len(test_loader)
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)

    print(f"Test Loss: {avg_test_loss:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

    # 保存一些测试图像比较
    with torch.no_grad():
        input_imgs, target_imgs = next(iter(test_loader))
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        outputs = model(input_imgs)

        # 选择几个样本进行可视化
        num_samples = min(4, input_imgs.size(0))

        # 创建一个大的图像网格
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

        for i in range(num_samples):
            # 显示输入图像 (取前3个通道用于可视化)
            if input_imgs.size(1) > 3:
                display_input = input_imgs[i, :3].cpu().permute(1, 2, 0)
            else:
                display_input = input_imgs[i].cpu().permute(1, 2, 0)

            axes[i, 0].imshow(display_input)
            axes[i, 0].set_title(f"Input")
            axes[i, 0].axis('off')

            # 显示目标图像
            axes[i, 1].imshow(target_imgs[i].cpu().permute(1, 2, 0))
            axes[i, 1].set_title(f"Target")
            axes[i, 1].axis('off')

            # 显示生成的图像
            axes[i, 2].imshow(outputs[i].cpu().permute(1, 2, 0))
            axes[i, 2].set_title(f"Generated")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('test_results.png')
        plt.close()

    return avg_test_loss, avg_psnr


# 主函数
def main():
    # 加载数据
    print("Loading data...")
    try:

        head_images = torch.load('processed_data/h-1.pt')
        k_field_images = torch.load('processed_data/k-1.pt')

        print(f"Loaded input images shape: {head_images.shape}")
        print(f"Loaded target images shape: {k_field_images.shape}")

        # 创建数据集
        full_dataset = ImageDataset(head_images, k_field_images)

        # 分割数据集
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)
        test_loader = DataLoader(test_dataset, batch_size=16)

        # 初始化模型
        model = ImageGenerationModel(
            in_channels=head_images.shape[1],  # 输入通道数
            out_channels=k_field_images.shape[1],  # 输出通道数
            feature_dim=256
        ).to(device)

        # 训练模型
        print("Starting training...")
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
            lr=0.00001
        )

        # 加载最佳模型
        checkpoint = torch.load('checkpoints/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        # 评估模型
        print("Evaluating model...")
        test_loss, test_psnr = evaluate_model(model, test_loader)

        print("Done!")

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        print("Please ensure you have preprocessed image data correctly")


if __name__ == "__main__":
    main()