import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from model.CNN_2 import mycnn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import time
import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_data

start_time = time.time()
# 设置随机种子
torch.manual_seed(77)

# torch.set_default_dtype(torch.float32)


# def load_data():
#     X_list = ['C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_EU_ST-T_V5_2.csv',
#               'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_MIT_ST_ECG_2.csv',
#               'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\X_SU_ECG_cleaned_2.csv']
#     Y_list = ['C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_EU_ST-T_V5_2.csv',
#               'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_MIT_ST_ECG_2.csv',
#               'C:\\Users\\qinxi\\PycharmProjects\\ecg_zh\\data_load\\Y_SU_ECG_cleaned_2.csv']
#
#     # 从文件加载数据和标签
#     X = torch.tensor(np.loadtxt('data_load/X_MITARR_MLII_2.csv', delimiter=',', skiprows=1).astype('float64'))
#     Y = np.loadtxt('data_load/Y_MITARR_MLII_2.csv', dtype="str", delimiter=',', skiprows=1)
#
#     # 合并数据集
#     print("begin concatenating...")
#     for x_file, y_file in zip(X_list, Y_list):
#         X = torch.cat((X, torch.tensor(np.loadtxt(x_file, delimiter=',', skiprows=1).astype(np.float64))))
#         Y = np.concatenate((Y, np.loadtxt(y_file, dtype="str", delimiter=',', skiprows=1)))
#
#     return X, Y


# 调用 load_data 函数来加载数据
X, Y = load_data()

# AAMI = ['N', 'L', 'R', 'V', 'A', '|', 'B']
AAMI = ['A', 'B', 'L', 'N', 'R', 'V', '|']


def filter_indices(y, valid_labels):
    return [i for i in range(len(y)) if y[i] in valid_labels]


save_list = filter_indices(Y, AAMI)

X = X[torch.tensor(save_list)]
Y = Y[torch.tensor(save_list)]

X = X.float()
# le = preprocessing.LabelEncoder()
# le.fit(AAMI)
# AAMI_encoded = le.transform(AAMI)
# Y_encoded = le.fit_transform(Y)
# Y = torch.tensor(Y_encoded)

# 对 AAMI 进行编码
le_aami = preprocessing.LabelEncoder()
le_aami.fit(AAMI)
AAMI_encoded = le_aami.transform(AAMI)

# 使用相同的编码器对 Y 进行编码
Y_encoded = le_aami.transform(Y)
Y = torch.tensor(Y_encoded, dtype=torch.long)




# 然后在需要的地方使用这个函数


# save_list = filter_indices(Y_encoded, AAMI_encoded)

# 删除不在 AAMI 中标签的数据
# save_list = [i for i in range(len(Y)) if Y[i] in AAMI_encoded]
# X = torch.index_select(X, 0, torch.tensor(save_list, dtype=torch.long))
# Y = torch.index_select(Y, 0, torch.tensor(save_list, dtype=torch.long))

# X = X[torch.tensor(save_list)]
# Y = Y[torch.tensor(save_list)]
#
# X = X.float()

# 数据标准化
print("begin standard scaler...")
ss = StandardScaler()
std_data = torch.tensor(ss.fit_transform(X))
X = std_data.unsqueeze(2)

# # 标签编码
# le = preprocessing.LabelEncoder()
# Y = torch.tensor(le.fit_transform(Y))

# # 把标签编码
# le = preprocessing.LabelEncoder()
# le = le.fit(AAMI)
# Y = torch.tensor(le.transform(Y))

# 分层抽样
print("begin StratifiedShuffleSplit...")
indices = torch.arange(len(Y))
train_index, test_index = next(iter(StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0).split(indices, Y)))
X_train, X_test = X[train_index], X[test_index]
y_train, y_test = Y[train_index], Y[test_index]

# one-hot 编码
# y_train_one_hot = F.one_hot(y_train).float()
# y_test_one_hot = F.one_hot(y_test).float()

# 统计每一类的数量
class_counts = torch.bincount(y_train)

# 打印每一类的数量
for class_idx, count in enumerate(class_counts):
    print(f'Class {class_idx}: {count.item()}')

# 将数据转为 PyTorch DataLoader
train_dataset = TensorDataset(X_train.transpose(1, 2), y_train)
test_dataset = TensorDataset(X_test.transpose(1, 2), y_test)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# # 获取一个 batch 的函数
# def get_batch(train_loader):
#     data_iter = iter(train_loader)
#     return next(data_iter)


print(f"X 的数据类型：{X.dtype}")
print(f"Y 的数据类型：{Y.dtype}")
print(f"X_train 的数据类型：{X_train.dtype}")
print(f"y_train 的数据类型：{y_train.dtype}")
print(f"X_test 的数据类型：{X_test.dtype}")
print(f"y_test 的数据类型：{y_test.dtype}")
print(f"y_train_one_hot 的数据类型：{y_train.dtype}")
print(f"y_test_one_hot 的数据类型：{y_test.dtype}")


model = mycnn()

num_epochs = 60
learning_rate = 0.0001

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_plt = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_index, (X_batch, y_batch) in enumerate(train_loader):
        # print("Input shape:", X_batch.shape)
        # print("Labels:", y_batch)
        # clear gradients
        optimizer.zero_grad()
        # forward pass
        outputs = model(X_batch.float())
        # outputs = model(X_batch)
        # outputs = outputs.double()
        # y_batch = y_batch.double()
        # loss = criterion(outputs, torch.argmax(y_batch, dim=1))
        loss = criterion(outputs, y_batch)
        # backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    loss_plt.append(average_loss)
    torch.save(model.state_dict(), f'model_weights_epoch_{epoch + 1}.pth')
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')
    plt.plot(range(1, len(loss_plt) + 1), loss_plt, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        # correct = 0
        # total = 0
        for X_batch, y_batch in test_loader:
            # outputs = model(X_batch)
            outputs = model(X_batch.float())
            _, predicted = torch.max(outputs, 1)
            # total += y_batch.size(0)
            # correct += (predicted == torch.argmax(y_batch, dim=1)).sum().item()
            all_predictions.extend(predicted.numpy())
            # all_labels.extend(torch.argmax(y_batch).numpy())
            # all_labels.extend(torch.argmax(torch.unsqueeze(y_batch, 0)).numpy())
            # all_labels.extend(torch.argmax(y_batch).item())
            for label in y_batch:
                all_labels.append(label.item())
            # all_labels.append(y_batch.item())
            # print(len(all_labels))
            # print(len(all_predictions))

    # accuracy = correct / total
    cm = confusion_matrix(all_labels, all_predictions)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # 绘制混淆矩阵的热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=AAMI, yticklabels=AAMI)
    plt.title(f'Normalized Confusion Matrix - Epoch {epoch + 1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # 计算精确度、召回率和 F1 分数
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=1.0)
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # print(f'test accuracy: {accuracy}')

    print(f'Test Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

end_time = time.time()
execution_time = end_time - start_time
print(f"程序运行时间：{execution_time} 秒")

