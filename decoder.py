import torch

class Decoder(torch.nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.LogSoftmax(dim = 2)

    def forward(self, x, s):
        embedded = torch.nn.functional.relu(self.embedding(x))
        reccurent_weights, hidden = self.gru(embedded, s)
        output = self.fc(reccurent_weights)
        return self.softmax(output), hidden

##################################################################################

