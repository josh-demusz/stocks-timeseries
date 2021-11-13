import torch
import pandas as pd
import random

class StocksDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, data_perc=None, transform=None, label_name='Close'):
        self.filepath = filepath
        self.data_size = None
        self.data = pd.read_csv(filepath)
        self.transform = transform
        self.data_perc = data_perc
        self.label_name = label_name

        self.convert_types()

        self.data = self.data.drop(columns="N")

        # Normalize data
        self.data.loc[:, self.data.columns != 'Index'] = (self.data.loc[:, self.data.columns != 'Index'] - self.data.loc[:, self.data.columns != 'Index'].mean()) / self.data.loc[:, self.data.columns != 'Index'].std()

        self.data = self.data.dropna()

        if self.data_perc:
            reduced_data_size = int(data_perc * self.data.shape[0])
            self.data = self.data.iloc[:reduced_data_size]
            # print(self.data.shape[0])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):

        try:
            row = self.data.loc[idx, :]
        except:
            row = self.data.loc[0, :]
            print('Error in dataloader')

        features = torch.from_numpy(row[self.data.columns != self.label_name].to_numpy())
        target = torch.from_numpy(row[self.data.columns == self.label_name].to_numpy()).float()[0]

        features = torch.unsqueeze(features, 0).float()

        if self.transform:
            features = self.transform(features)

        return features, target

    def mean(self):
        return self.data.mean()

    def convert_types(self):
        # self.data.Index = pd.Categorical(self.data.Index)
        self.data.Index = self.data.Index.astype('category').cat.codes

class StocksSequenceDataset(torch.utils.data.Dataset):

    def __init__(self, filepath, n_samples, variable_seq_len=True, seq_len=[3, 30], transform=None, label_name='Close', include_time=False):
        self.filepath = filepath
        self.data_size = None
        self.data = pd.read_csv(filepath)
        self.transform = transform
        self.label_name = label_name
        self.variable_seq_len = variable_seq_len
        self.seq_len = seq_len
        self.include_time = include_time

        self.convert_types()

        self.data = self.data.drop(columns="N")

        # Normalize data
        self.data.loc[:, self.data.columns != 'Index'] = (self.data.loc[:, self.data.columns != 'Index'] - self.data.loc[:, self.data.columns != 'Index'].mean()) / self.data.loc[:, self.data.columns != 'Index'].std()

        self.data = self.data.dropna()

        self.sequence_data = self.generate_sequences(n_samples)

    def generate_sequences(self, n_samples):
        sequence_data = []

        data = self.data.sort_values(by=['Date'])
        n_unique_indexes = len(pd.unique(data['Index']))

        for i in range(n_samples):
            index = random.randint(0, n_unique_indexes - 1)
            index_data = data[data['Index'] == index]

            if self.variable_seq_len:
                len_seq = random.randint(self.seq_len[0], self.seq_len[1])
            else:
                len_seq = self.seq_len

            start_i = random.randint(0, index_data.shape[0] - len_seq - 1)

            # sequence = (index_data['Close'].iloc[start_i:start_i + len_seq], index_data['Close'].iloc[start_i:start_i + len_seq].values)
            try:
                target = index_data['Close'].iloc[start_i + len_seq + 1]
            except IndexError:
                print('start_i: {}'.format(start_i))
                print('len_seq: {}'.format(len_seq))
                print('index_data.shape[0]: {}'.format(index_data.shape[0]))

            if self.include_time:
                (timestamps, sequence) = (index_data['Date'].iloc[start_i:start_i + len_seq].values,
                                          index_data['Close'].iloc[start_i:start_i + len_seq].values)
                #print(timestamps.dtypes)
                sequence_data.append((timestamps, sequence, target))
            else:
                sequence = index_data['Close'].iloc[start_i:start_i + len_seq].values
                sequence_data.append((sequence, target))

        return sequence_data

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        if self.include_time:
            try:
                (timestamps, sequence, target) = self.sequence_data[idx]
            except:
                (timestamps, sequence, target) = self.sequence_data[0]
                print('Error in dataloader')
        else:
            try:
                (sequence, target) = self.sequence_data[idx]
            except:
                (sequence, target) = self.sequence_data[0]
                print('Error in dataloader')

        if self.transform:
            sequence = self.transform(sequence)

        sequence = torch.from_numpy(sequence)

        if self.include_time:
            return timestamps, sequence, target

        return sequence, target

    def convert_types(self):
        # self.data.Index = pd.Categorical(self.data.Index)
        self.data.Index = self.data.Index.astype('category').cat.codes
