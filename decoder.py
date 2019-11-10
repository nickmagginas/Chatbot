import encoder as enc
import read_dialogues as reader
import torch

FILENAME = 'data/dialogues/AGREEMENT_BOT.txt'
HIDDEN_SIZE = 512

class Decoder(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(output_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, s):
        embedded = torch.nn.functional.relu(self.embedding(x))
        o, ns = self.gru(embedded, s)
        return torch.nn.functional.softmax(o, dim = 1), ns

def test():
    dictionary, pairs = reader.prepare_data(FILENAME)
    length = dictionary.__len__()
    encoder = enc.Encoder(length, HIDDEN_SIZE)
    state = encoder.init_state(HIDDEN_SIZE)
    state = enc.feed_sentence(encoder, pairs[0][0], state, length)
    decoder = Decoder(HIDDEN_SIZE, length)
    x = pairs[0][1][3]
    output, _ = decoder(torch.LongTensor([x]).view(1, 1), state)
    print(output)

test()
