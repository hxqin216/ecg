import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=20, stride=3, padding='same')
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=3)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=7, stride=1, padding='same')
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=10, stride=1, padding='same')
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=32, hidden_size=10, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 7)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.conv3(x)))

        # Reshape for LSTM layer
        x = x.permute(0, 2, 1)  # PyTorch LSTM expects batch_first format
        x, (hn, cn) = self.lstm(x)

        # Select the last time step's output
        x = x[:, -1, :]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)




class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
        self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y.expand_as(x)
