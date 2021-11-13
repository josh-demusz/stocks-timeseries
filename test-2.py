from datasets import *
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

filepath = 'data/transformed_indexData.csv'
label_name = 'Close'
train_dataset = StocksSequenceDataset(filepath, 50, label_name=label_name, include_time=True)

print('Dataset output: ')
print(train_dataset[0])