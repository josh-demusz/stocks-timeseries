import torch.nn.functional as F
import torch.nn as nn

class SimpleConvNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, widen_factor=2, kernel_size=3, stride=1):
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

    #         self.model = nn.Sequential(
    #             nn.Conv1d(in_channels=in_channels, out_channels=channels[0], kernel_size=kernel_size, stride=stride),
    #             nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=stride),
    #             nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, stride=stride),
    #             nn.BatchNorm1d(channels[2], momentum=0.001),
    #             nn.LeakyReLU(),
    #             nn.Linear(channels[2], num_classes)
    #         )

    #     def forward(self, x):
    #         out = self.model(x)
    #         return out
    def forward(self, x):
        out = self.conv1(x)
        #         print('conv1 output: {}'.format(out.size()))
        out = self.conv2(out)
        #         print('conv2 output: {}'.format(out.size()))
        out = self.conv3(out)
        #         print('conv3 output: {}'.format(out.size()))
        out = self.batch_norm(out)
        #         print('batch_norm output: {}'.format(out.size()))
        out = self.leaky_relu(out)
        #         print('leaky_relu output: {}'.format(out.size()))
        out = F.adaptive_avg_pool1d(out, 1)
        #         print('adaptive_avg_pool1d output: {}'.format(out.size()))
        # out = out.view(-1, self.channels)
        # print('out.view: {}'.format(out.size()))
        out = self.linear(out.squeeze())
        #         print('linear output: {}'.format(out.size()))
        return out



class SimpleConvNetwork2d(nn.Module):
    def __init__(self, in_channels=1, num_classes=2, widen_factor=2, kernel_size=3, stride=1):
        super(SimpleConvNetwork2d, self).__init__()
        self.channels = [4 * widen_factor * val for val in [1, 2, 3]]
        print(self.channels)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.channels[0], kernel_size=kernel_size,
                               stride=stride)
        self.conv2 = nn.Conv2d(in_channels=self.channels[0], out_channels=self.channels[1], kernel_size=kernel_size,
                               stride=stride)
        self.conv3 = nn.Conv2d(in_channels=self.channels[1], out_channels=self.channels[2], kernel_size=kernel_size,
                               stride=stride)
        self.batch_norm = nn.BatchNorm2d(self.channels[2], momentum=0.001)
        self.leaky_relu = nn.LeakyReLU()
        self.linear = nn.Linear(self.channels[2], num_classes)

    #         self.model = nn.Sequential(
    #             nn.Conv1d(in_channels=in_channels, out_channels=channels[0], kernel_size=kernel_size, stride=stride),
    #             nn.Conv1d(in_channels=channels[0], out_channels=channels[1], kernel_size=kernel_size, stride=stride),
    #             nn.Conv1d(in_channels=channels[1], out_channels=channels[2], kernel_size=kernel_size, stride=stride),
    #             nn.BatchNorm1d(channels[2], momentum=0.001),
    #             nn.LeakyReLU(),
    #             nn.Linear(channels[2], num_classes)
    #         )

    #     def forward(self, x):
    #         out = self.model(x)
    #         return out
    def forward(self, x):
        out = self.conv1(x)
        #         print('conv1 output: {}'.format(out.size()))
        out = self.conv2(out)
        #         print('conv2 output: {}'.format(out.size()))
        out = self.conv3(out)
        #         print('conv3 output: {}'.format(out.size()))
        out = self.batch_norm(out)
        #         print('batch_norm output: {}'.format(out.size()))
        out = self.leaky_relu(out)
        #         print('leaky_relu output: {}'.format(out.size()))
        out = F.adaptive_avg_pool2d(out, 1)
        #         print('adaptive_avg_pool1d output: {}'.format(out.size()))
        # out = out.view(-1, self.channels)
        # print('out.view: {}'.format(out.size()))
        out = self.linear(out.squeeze())
        #         print('linear output: {}'.format(out.size()))
        return out