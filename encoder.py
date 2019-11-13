import torch
import read_files as r

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

def parallel_training(pairs):
    dimension_range = range(r.MIN_LENGTH, r.MAX_LENGTH + 1)
    for dimension in [(ql, al) for ql in dimension_range for al in dimension_range]:
        parallel_pairs = [*filter(lambda p: r.pair_dimensions(*p) == dimension , pairs)]
        yield parallel_pairs

pairs, dictionary = r.create_dataset('data/dialogues/AGREEMENT_BOT.txt')
encoder = Encoder(dictionary.__len__(), 64)

###########################################################################################################

'''
for parallel_pairs in parallel_training(pairs):
    questions = [*map(lambda p: p[0], parallel_pairs)]
    questions = torch.tensor(questions, dtype = torch.long)
    output, state = hidden_states = encoder(questions, encoder.__init_hidden__(64).repeat(1, 3, 1))
'''
