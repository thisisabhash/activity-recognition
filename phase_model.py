import torch
import torch.nn as nn

# Different architectures for phase recognition

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, softmax, device):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.softmax = softmax
        self.output_dim = output_dim
        self.device = device
        self.model = nn.LSTM(input_dim, hidden_dim, 1)

        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(1, 1, self.hidden_dim).type(torch.FloatTensor).to(self.device),
               torch.randn(1, 1, self.hidden_dim).type(torch.FloatTensor).to(self.device))

    def forward(self, input):
        hidden_output, self.hidden = self.model(input, self.hidden)
        output = self.hidden2out(hidden_output)

        output_resized = torch.randn(len(output), self.output_dim)
        for i in range(len(output)):
            output_resized[i] = output[i][0]
        output_resized = torch.FloatTensor(output_resized)
        return output_resized
    

class GRU(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, softmax, device):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.softmax = softmax
        self.output_dim = output_dim
        self.device = device
        self.model = nn.GRU(input_dim, hidden_dim, 1)

        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(1, 1, self.hidden_dim).type(torch.FloatTensor).to(self.device)

    def forward(self, input):
        hidden_output, self.hidden = self.model(input, self.hidden)
        output = self.hidden2out(hidden_output)

        output_resized = torch.randn(len(output), self.output_dim)
        for i in range(len(output)):
            output_resized[i] = output[i][0]
        output_resized = torch.FloatTensor(output_resized)
        return output_resized

class RNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, softmax, device):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.softmax = softmax
        self.output_dim = output_dim
        self.device = device
        self.model = nn.RNN(input_dim, hidden_dim, 1)

        self.hidden2out = nn.Linear(hidden_dim, output_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return torch.randn(1, 1, self.hidden_dim).type(torch.FloatTensor).to(self.device)

    def forward(self, input):
        hidden_output, self.hidden = self.model(input, self.hidden)
        output = self.hidden2out(hidden_output)
        
        output_resized = torch.randn(len(output), self.output_dim)
        for i in range(len(output)):
            output_resized[i] = output[i][0]
        output_resized = torch.FloatTensor(output_resized)
        return output_resized