import torch
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

        self.layer3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding=1)
        self.layer4 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=1)
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.layer5 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding=1)
        self.layer6 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(128, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.output_softmax = nn.Linear(512, 7)

    def forward(self, x):
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)

        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.max_pool2(x)

        x = self.layer3(x)
        x = self.layer4(x)
        x = self.max_pool3(x)

        x = self.layer5(x)
        x = self.layer6(x)

        x = self.flatten(x)
        x = self.dense(x)
        x = self.dropout(x)
        output = self.output_softmax(x)

        return output


# 创建模型
model = mycnn()


