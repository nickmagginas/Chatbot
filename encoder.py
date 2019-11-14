import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, x, s):
        embedded = self.embedding(x)
        return self.gru(embedded, s)

    @staticmethod
    def __init_hidden__(hidden_size): return torch.zeros(1, hidden_size)

