import json
import string

FILTER_LENGTH = 8

### Function Composition
compose = lambda f, g: lambda args: f(g(args))

### JSON Parse and get Dialogue
get_dialogue = lambda line: json.loads(line)['turns']

### Length of sentence -- Helper --
sentence_length = lambda s: len(s.split())

### Flattens List -- Helper --
flatten = lambda l: [qa for qal in l for qa in qal]

### Get a Dialogue and Construct Pairs
def construct_pairs(d):
    return [(d[2*n], d[2*n + 1]) for n in range(d.__len__() // 2)]

### Read From file, compose and get QA pairs
def get_qa(filename):
    return [*map(compose(construct_pairs, get_dialogue), open(filename, 'r'))]

### Filter pairs than are smaller than CUTOFF
def filter_pairs(qa_pairs, cutoff):
    pair_length = lambda p: all(l < cutoff for l in map(sentence_length, p))
    return filter(pair_length, qa_pairs)

### Remove any puntuation
def remove_punctuation(s):
    return s.translate({ord(key): None for key in string.punctuation})

### Construct Vocabulary of Corpus
def construct_vocabulary(pairs):
    qa_chain = ''.join([f'{q} {a} ' for (q, a) in pairs])
    processed = map(lambda s: s.lower(), remove_punctuation(qa_chain).split()) ### Lower-Case, Remove Punctuation
    unique_words = set(processed)
    dictionary = {k: v + 3 for (k,v) in zip(unique_words, range(len(unique_words)))} ### Map to Indexes
    f_dictionary = {**{'PAD': 0, 'SOS': 1, 'EOS': 2}, **dictionary} ### Add PAD, SOS, EOS
    return f_dictionary

### Encode
def encode_sentence(s, dictionary):
    encoded = [dictionary[word] for word in s.split()]
    return [1] + encoded + [2]

### Encode Corpus
def encode_corpus(pairs, dictionary):
    return [tuple(map(lambda s: encode_sentence(s, dictionary), p)) for p in pairs]


### Lowercase remove punctuation
def process_pairs(pairs):
    process_sentence = lambda s: s.lower().translate({ord(i): None for i in string.punctuation})
    return [tuple(map(process_sentence, p)) for p in pairs]

### Compose module
def prepare_data(filename):
    qa_pairs = flatten(get_qa(filename))
    filtered_pairs = [*filter_pairs(qa_pairs, FILTER_LENGTH)]
    dictionary = construct_vocabulary(filtered_pairs)
    processed_pairs = process_pairs(filtered_pairs)
    return encode_corpus(processed_pairs, dictionary)















