import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 全局配置（路径，参数）
CONFIG = {
    # 输入路径配置
    "k_field_pattern": "k_picture1/noisy_{}.png",  # 渗透率场路径模板
    "head_contour_pattern": "figure1/figure_{}.png",  # 水头等值线图路径模板

    # 输出配置
    "output_dir": "processed",
    "k_output": "k-1.pt",  # 渗透率场输出文件名
    "h_output": "h-1.pt",  # 水头图输出文件名

    # 处理参数
    "img_size": (256, 256),  # 建议根据模型输入尺寸调整
    "num_samples": 3000,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def create_transformer():
    """创建仅包含Resize和ToTensor的转换管道"""
    return transforms.Compose([
        transforms.Resize(CONFIG["img_size"]),
        transforms.ToTensor()  # 自动将像素值缩放到[0, 1]
    ])

def process_dataset(pattern, transform, device):
    """通用数据集处理流程"""
    tensor_batch = []
    for idx in tqdm(range(1, CONFIG["num_samples"] + 1),
                    desc=f"Processing {os.path.basename(pattern)}"):
        try:
            img_path = pattern.format(idx)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            tensor_batch.append(tensor)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            tensor_batch.append(torch.zeros(1, 3, *CONFIG["img_size"]).to(device))
    return torch.cat(tensor_batch, dim=0)

def main():
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 创建统一的转换器（无需计算均值和标准差）
    k_transform = create_transformer()
    h_transform = create_transformer()

    # 并行处理两个数据集
    print("=" * 50)
    print("Processing K Fields...")
    k_tensors = process_dataset(CONFIG["k_field_pattern"], k_transform, CONFIG["device"])

    print("\n" + "=" * 50)
    print("Processing Head Contours...")
    h_tensors = process_dataset(CONFIG["head_contour_pattern"], h_transform, CONFIG["device"])

    # 保存结果
    torch.save(k_tensors.cpu(), os.path.join(CONFIG["output_dir"], CONFIG["k_output"]))
    torch.save(h_tensors.cpu(), os.path.join(CONFIG["output_dir"], CONFIG["h_output"]))

    # 验证输出
    print("\nProcessing Complete!")
    print(f"K Fields Shape: {k_tensors.shape} (Saved to {CONFIG['k_output']})")
    print(f"Head Contours Shape: {h_tensors.shape} (Saved to {CONFIG['h_output']})")

if __name__ == "__main__":
    main()