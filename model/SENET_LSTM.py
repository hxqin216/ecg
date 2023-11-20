import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y


class SENet_LSTM(nn.Module):
    def __init__(self):
        super(SENet_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=20, stride=3, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2, 3)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding='same')
        self.se_block1 = SEBlock(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2, 2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding='same')
        self.se_block2 = SEBlock(32)
        self.pool3 = nn.MaxPool1d(2, 2)

        self.lstm = nn.LSTM(input_size=32, hidden_size=10, batch_first=True)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.output = nn.Linear(10, 7)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.se_block1(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3(x))
        x = self.se_block2(x)
        x = self.pool3(x)

        x = x.permute(0, 2, 1)  # Batch first for LSTM
        x, (hn, cn) = self.lstm(x)

        x = self.flatten(x[:, -1, :])
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return F.log_softmax(x, dim=1)
