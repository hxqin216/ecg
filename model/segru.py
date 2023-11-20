import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

class SERBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SERBlock, self).__init__()
        self.se_block = SEBlock(channel, reduction)

    def forward(self, x):
        return x + self.se_block(x)





class SE_GRU(nn.Module):
    def __init__(self):
        super(SE_GRU, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=50, stride=3, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.se_block1 = SEBlock(128)
        self.pool1 = nn.MaxPool1d(2, 3)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.se_block2 = SEBlock(32)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding='same')
        self.conv4 = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding='same')
        self.se_block3 = SEBlock(128)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=512, kernel_size=5, stride=1, padding='same')
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding='same')
        self.se_block4 = SEBlock(128)

        self.gru = nn.GRU(input_size=128, hidden_size=60, batch_first=True)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(60, 512)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.Linear(512, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se_block1(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se_block2(x)
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.se_block3(x)
        x = self.pool3(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.se_block4(x)

        # Reshape for GRU
        x = x.permute(0, 2, 1)  # Batch first
        x, _ = self.gru(x)
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)
