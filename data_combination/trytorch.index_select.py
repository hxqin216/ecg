import torch

# 创建一个示例张量
tensor = torch.tensor([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])

# 要选择的行的索引
indices = torch.tensor([0, 2])

# 使用torch.index_select选择特定的行
selected_rows = torch.index_select(tensor, 0, indices)

print(selected_rows)
