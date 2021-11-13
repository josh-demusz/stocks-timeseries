import torch.nn.functional as F
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, n_feature, n_hidden, n_classes):
        super(SimpleNetwork, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.batch_norm = nn.BatchNorm1d(1, momentum=0.001)
        self.fully_connected = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = self.batch_norm(x)
        x = F.relu(self.hidden_2(x))
        x = self.batch_norm(x)
        x = self.fully_connected(x)
        return x

class SimpleLSTMNetwork(nn.Module):
    def __init__(self, n_feature, n_hidden, n_classes):
        super(SimpleLSTMNetwork, self).__init__()
        self.hidden_1 = nn.LSTM(n_feature, n_hidden)
        self.hidden_2 = nn.LSTM(n_hidden, n_hidden)
        self.batch_norm = nn.BatchNorm1d(1, momentum=0.001)
        self.fully_connected = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = self.batch_norm(x)
        x = F.relu(self.hidden_2(x))
        x = self.batch_norm(x)
        x = self.fully_connected(x)
        return x

class SimpleRNNNetwork(nn.Module):
    def __init__(self, n_feature, n_hidden, n_classes):
        super(SimpleRNNNetwork, self).__init__()
        self.hidden_1 = nn.RNN(n_feature, n_hidden)
        self.hidden_2 = nn.RNN(n_hidden, n_hidden)
        self.batch_norm = nn.BatchNorm1d(1, momentum=0.001)
        self.fully_connected = nn.Linear(n_hidden, n_classes)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = self.batch_norm(x)
        x = F.relu(self.hidden_2(x))
        x = self.batch_norm(x)
        x = self.fully_connected(x)
        return x

class SimpleConvNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=1, widen_factor=2, kernel_size=3, stride=1):
        super(SimpleConvNetwork, self).__init__()
        self.channels = [4 * widen_factor * val for val in [1, 2, 3]]
        print(self.channels)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=self.channels[0], kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv1d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=kernel_size,
                               stride=stride)
        self.conv3 = nn.Conv1d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=kernel_size,
                               stride=stride)
        self.batch_norm = nn.BatchNorm1d(self.channels[2], momentum=0.001)
        self.leaky_relu = nn.LeakyReLU()
        self.linear = nn.Linear(self.channels[2], num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.batch_norm(out)
        out = self.leaky_relu(out)
        out = F.adaptive_avg_pool1d(out, 1)
        out = self.linear(out.squeeze())
        return out
