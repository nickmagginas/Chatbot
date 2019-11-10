import read_dialogues as reader
import torch

FILENAME = 'data/dialogues/AGREEMENT_BOT.txt'

### read data
def main():
    dictionary, pairs = reader.prepare_data(FILENAME)

### Encoder -- Embedding and GRU --
class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding(torch.nn.Embedding(input_size, hidden_size))
        self.gru = torch.nn.GRU(hidden_size, hidden_size)


if __name__ == '__main__':
    main()
