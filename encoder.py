import read_dialogues as reader
import torch

FILENAME = 'data/dialogues/AGREEMENT_BOT.txt'
HIDDEN_SIZE = 512

### read data
def test():
    dictionary, pairs = reader.prepare_data(FILENAME)
    encoder = Encoder(dictionary.__len__(), HIDDEN_SIZE)
    state = encoder.init_state(HIDDEN_SIZE)
    feed_sentence(encoder, pairs[0][0], state, dictionary.__len__())

### Helper for OneHoting Words
def onehot(word, length):
    return [1 if i == word else 0 for i in range(length)]

### Propagate Recursively until EOS flag
def propagate_state(encoder, sentence, state, length):
    word = next(sentence)
    if word == 1: return propagate_state(encoder, sentence, state, length)
    x = torch.LongTensor(onehot(word, length)).view(length, 1)
    _, new_state = encoder.forward(x, state)
    call = lambda: propagate_state(encoder, sentence, new_state, length)
    return new_state if word == 2 else call()

### Feed to get encoding
def feed_sentence(encoder, sentence, state, length):
    sentence_iter = iter(sentence)
    state = propagate_state(encoder, sentence_iter, state, length)
    return state

### Encoder -- Embedding and GRU --
class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, x, s):
        embedded = self.embedding(x)
        return self.gru(embedded, s)

    @staticmethod
    def init_state(dimensions):
        return torch.zeros(1, 1, dimensions)

if __name__ == '__main__':
    test()
