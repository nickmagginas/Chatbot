import json
import string
import torch
# import matplotlib.pyplot as plt

FILTER_LENGTH = 8

compose = lambda f, g: lambda args: f(g(args))
get_dialogue = lambda line: json.loads(line)['turns']
flatten = lambda l: [qa for qal in l for qa in qal]

def get_qa(filename):
    return [*map(compose(construct_pairs, get_dialogue), open(filename, 'r'))]

def construct_pairs(d):
    return [(d[2*n], d[2*n + 1]) for n in range(d.__len__() // 2)]

QA_PAIRS = flatten(get_qa('data/dialogues/AGREEMENT_BOT.txt'))

qa_chain = ''.join([f'{q} {a} ' for (q, a) in QA_PAIRS])

def remove_punctuation(s):
    return s.translate({ord(key): None for key in string.punctuation})

processed = map(lambda s: s.lower(), remove_punctuation(qa_chain).split())
unique_words = set(processed)

dictionary = {k: v for (k,v) in zip(unique_words, range(len(unique_words)))}

sentence_length = lambda s: len(s.split())


def filter_pairs(qa_pairs, cutoff):
    pair_length = lambda p: all(l < cutoff for l in map(sentence_length, p))
    return filter(pair_length, qa_pairs)

filtered_pairs = filter_pairs(QA_PAIRS, FILTER_LENGTH)

'''
lengths = [*map(sentence_length, [s for qa in filtered_pairs for s in qa])]

plt.hist(lengths, bins = 30)
plt.show()
'''

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.gru = torch.nn.GRU(hidden_size, hidden_size)

    def forward(self, x, s):
        embedding = self.embedding(x)
        output, hidden = self.gru(x, s)
        return output, hidden

    @staticmethod
    def init_state(hidden_size): return torch.zeros(1, 1, hidden_size)

def transform_pair(pair, dictionary):
    question, answer = pair


encoder = Encoder(8, 512)
initial_state = encoder.init_state(512)





















