import torch

### Propagate Recursively until EOS flag
def propagate_state(encoder, sentence, state):
    word = next(sentence)
    if word == 1: return propagate_state(encoder, sentence, state)
    x = torch.LongTensor([word]).view(1,1)
    _, new_state = encoder.forward(x, state)
    call = lambda: propagate_state(encoder, sentence, new_state)
    return new_state if word == 2 else call()

### Feed to get encoding
def feed_sentence(encoder, sentence, state):
    sentence_iter = iter(sentence)
    state = propagate_state(encoder, sentence_iter, state)
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

