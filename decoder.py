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

'''
class Decoder(torch.nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.attention = torch.nn.Linear(3 * hidden_size, 1)
        self.softmax_attention = torch.nn.Softmax()
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, s, encoder_outputs):
        sequence_length = encoder_outputs.shape[1]
        repeated_state = s.repeat(1, sequence_length, 1)
        embedded = self.embedding(x)
        repeated_input = embedded.repeat(1, sequence_length, 1)
        print(repeated_state.shape)
        print(repeated_input.shape)

encoder_outputs = torch.randn(53, 3, 32)
encoder_state = torch.randn(1, 3, 32)
sample = torch.randint(0, 3, (53, 1))

decoder = Decoder(1501, 32)
decoder(sample, encoder_state, encoder_outputs)
'''
