from datasets import *
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

filepath = 'data/transformed_indexData.csv'
label_name = 'Close'
train_dataset = StocksDataset(filepath, 0.3, label_name=label_name)

batch_size = 10

data_size = len(train_dataset)

train_ratio = 0.7

train_size = int(np.floor(data_size * train_ratio))

indices = list(range(data_size))

train_indices, test_indices = indices[:train_size], indices[train_size:]

train_sampler = SubsetRandomSampler(train_indices)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler)



print(next(iter(train_loader)))