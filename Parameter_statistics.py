import torch
import parser
import network


args = parser.parse_arguments()
model = network.network.HAM_VPRNet(args)

# 统计总参数量（单位：个）
total_params = sum(param.numel() for param in model.parameters())
print(f"总参数量: {total_params:,}")

# 转换为百万（M）或十亿（B）单位
print(f"总参数量: {total_params / 1e6:.2f} M")
print(f"总参数量: {total_params / 1e9:.2f} B")