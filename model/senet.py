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



class SENet(nn.Module):
    def __init__(self):
        super(SENet, self).__init__()
        self.initial_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2, stride=3),
            nn.Conv1d(64, 64, 1, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, stride=2, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 256, 1, stride=2, padding='same'),
            nn.ReLU(),
            SEBlock(256)
        )

        self.middle_layers = self._make_layers(64, [128, 256, 512], [4, 6, 3])

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 7),
            nn.Softmax(dim=1)
        )

    def _make_layers(self, in_channels, out_channels_list, repetitions_list):
        layers = []
        for out_channels, reps in zip(out_channels_list, repetitions_list):
            for _ in range(reps):
                layers.append(nn.Conv1d(in_channels, out_channels, 1, stride=2, padding='same'))
                layers.append(nn.ReLU())
                layers.append(nn.Conv1d(out_channels, out_channels, 3, stride=2, padding='same'))
                layers.append(nn.ReLU())
                layers.append(nn.Conv1d(out_channels, out_channels * 4, 1, stride=2, padding='same'))
                layers.append(nn.ReLU())
                layers.append(SEBlock(out_channels * 4))
                in_channels = out_channels * 4
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.middle_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x




