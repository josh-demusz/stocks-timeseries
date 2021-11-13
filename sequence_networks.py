import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1, batch_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden = (torch.zeros(1,batch_size,self.hidden_size),
                            torch.zeros(1,batch_size,self.hidden_size))

    def forward(self, input_seq):
        #len_input_seq = input_seq.size()[1]

        self.hidden = tuple([each.data for each in self.hidden])

        lstm_out, self.hidden = self.lstm(input_seq, self.hidden)
        predictions = self.linear(lstm_out)
        #predictions = predictions.squeeze()

        # Only take prediction following the LAST item in the sequence
        predictions_reduced = predictions[:, -1, :]
        #print('predictions_reduced: {}'.format(predictions_reduced.size()))

        # linear_2 = nn.Linear(len_input_seq, self.output_size)
        # predictions_reduced = linear_2(predictions)
        return predictions_reduced

class RNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1, batch_size=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden = torch.zeros(1,batch_size,self.hidden_size)

    def forward(self, input_seq):

        #self.hidden = [each.data for each in self.hidden]
        
        self.hidden = self.hidden.data

        rnn_out, self.hidden = self.rnn(input_seq, self.hidden)
        predictions = self.linear(rnn_out)

        # Only take prediction following the LAST item in the sequence
        predictions_reduced = predictions[:, -1, :]

        # linear_2 = nn.Linear(len_input_seq, self.output_size)
        return predictions_reduced