import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from model.CNN import mycnn
# 定义你的 CNN 模型类（mycnn）



# 设置随机种子
torch.manual_seed(77)

# 文件路径列表
X_list = ['C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_EU_ST-T_V5.csv',
          'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_MITST_ECG.csv',
          'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_SU_ECG_cleaned.csv']
Y_list = ['C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_EU_ST-T_V5.csv',
          'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_MITST_ECG.csv',
          'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_SU_ECG_cleaned.csv']

# 定义超参数
num_epochs = 60
batch_size = 32
learning_rate = 0.001

# 从文件加载数据和标签
X = torch.tensor(np.loadtxt('data_load/3X_MITARR_MLII.csv', delimiter=',', skiprows=1).astype('float32'))
Y = torch.tensor(np.loadtxt('data_load/3Y_MITARR_MLII.csv', dtype="str", delimiter=',', skiprows=1))

# 合并数据集
print("begin concatenating...")
for database in X_list:
    X = torch.cat((X, torch.tensor(np.loadtxt(database, dtype="str", delimiter=',', skiprows=1).astype(np.float32))))
for database in Y_list:
    Y = torch.cat((Y, torch.tensor(np.loadtxt(database, dtype="str", delimiter=',', skiprows=1))))

AAMI = ['N', 'L', 'R', 'V', 'A', '|', 'B']

# 删除不在 AAMI 中标签的数据
save_list = [i for i in range(len(Y)) if Y[i] in AAMI]
X = torch.index_select(X, 0, torch.tensor(save_list))
Y = torch.index_select(Y, 0, torch.tensor(save_list))

# 数据标准化
print("begin standard scaler...")
ss = StandardScaler()
std_data = torch.tensor(ss.fit_transform(X))
X = std_data.unsqueeze(2)

# 把标签编码
le = preprocessing.LabelEncoder()
le = le.fit(AAMI)
Y = torch.tensor(le.transform(Y))

# 分层抽样
print("begin StratifiedShuffleSplit...")
indices = torch.arange(len(Y))
train_index, test_index = next(iter(StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0).split(indices, Y)))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = Y[train_index], Y[test_index]

# one-hot 编码
y_train_one_hot = F.one_hot(y_train)
y_test_one_hot = F.one_hot(y_test)

# 将数据转为 PyTorch DataLoader
train_dataset = TensorDataset(X_train, y_train_one_hot)
test_dataset = TensorDataset(X_test, y_test_one_hot)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 创建模型
model = mycnn()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    for batch_index, (X_batch, y_batch) in enumerate(train_loader):
        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, torch.argmax(y_batch, dim=1))

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 输出损失
        print(f'Epoch {epoch + 1}/{num_epochs}, Batch {batch_index + 1}/{len(train_loader)}, Loss: {loss.item()}')

# 测试模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():
    correct = 0
    total = 0
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == torch.argmax(y_batch, dim=1)).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')