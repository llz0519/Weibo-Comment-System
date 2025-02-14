import torch

def check_cuda():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print("="*40 + " CUDA 信息 " + "="*40)
        print(f"▪ PyTorch版本: {torch.__version__}")
        print(f"▪ 可用设备数量: {torch.cuda.device_count()}")
        print(f"▪ 当前设备索引: {device}")
        print(f"▪ 设备名称: {torch.cuda.get_device_name(device)}")
        print(f"▪ CUDA版本: {torch.version.cuda}")
        print(f"▪ 计算能力: {torch.cuda.get_device_capability(device)}")
        
        # 显存信息
        free, total = torch.cuda.mem_get_info(device)
        print(f"\n显存使用情况:")
        print(f"▪ 总显存: {total/1024**3:.2f} GB")
        print(f"▪ 剩余显存: {free/1024**3:.2f} GB")
        print(f"▪ 已用显存: {(total - free)/1024**3:.2f} GB")
        
        print("\n" + "="*40 + " 建议 " + "="*40)
        print("提示1: 请确保PyTorch版本与CUDA版本兼容")
        print("提示2: 多卡用户可通过device参数选择不同设备")
    else:
        print("\n" + "!"*40 + " 警告 " + "!"*40)
        print("CUDA 不可用，将使用CPU运行")
        print("可能原因:")
        print("- 未安装NVIDIA显卡驱动")
        print("- PyTorch未配置CUDA版本")
        print("- 显卡计算能力不足")

if __name__ == "__main__":
    check_cuda()
