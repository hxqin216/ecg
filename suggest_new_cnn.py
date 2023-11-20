import torch.nn as nn

class mycnn(nn.Module):
    def __init__(self):
        super(mycnn, self).__init__()

        self.layer1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=3)

        self.layer2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # ... [其它层的定义] ...

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(128, 512)  # 确保 128 是正确的输入特征数
        self.dropout = nn.Dropout(p=0.1)
        self.output_softmax = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = self.max_pool2(x)

        # ... [其它层的前向传播] ...

        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        output = self.output_softmax(x)

        return output




import torch.nn as nn
import torch.nn.functional as F

class mycnn(nn.Module):
    def __init__(self):
        super(mycnn, self).__init__()

        # ... [卷积层定义] ...

        # 全连接层之前的 Flatten
        self.flatten = nn.Flatten()

        # 全连接层
        self.fc1 = nn.Linear(128, 512)  # 确保这里的输入维度正确
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.5)

        # 输出层
        self.output = nn.Linear(128, 7)  # 7个类别的输出

    def forward(self, x):
        # ... [卷积层的前向传播] ...

        x = self.flatten(x)

        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        output = F.log_softmax(self.output(x), dim=1)  # 使用log_softmax作为最后一层

        return output



# data_comb
import os
import wfdb
import pandas as pd

def load_record(name, rootdir):
    # 加载记录的代码
    return record, annotation

def process_record(record, annotation, segment_len, f):
    # 处理记录的代码，返回分段数据和标签
    return segmented_data, segmented_label

def save_to_csv(data, filename):
    # 保存数据到 CSV 文件
    pd.DataFrame(data).to_csv(filename, index=False)

def main():
    start_time = time.time()

    rootdir = 'mit-bih-arrhythmia-database-1.0.0'
    files = os.listdir(rootdir)
    # ... [其余代码] ...

    # 处理每个记录
    for person in mlii_records:
        record, annotation = load_record(person, rootdir)
        segmented_data, segmented_label = process_record(record, annotation, segment_len, f)
        # ... [保存或其他处理] ...

    save_to_csv(segmented_data, 'X_MITARR_ECG_2.csv')
    save_to_csv(segmented_label, 'Y_MITARR_ECG_2.csv')

    end_time = time.time()
    print(f"程序运行时间：{end_time - start_time} 秒")

if __name__ == "__main__":
    main()


