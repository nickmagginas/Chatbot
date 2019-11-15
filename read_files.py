import json
import string
import os

MIN_LENGTH = 3
MAX_LENGTH = 10

flatten = lambda l: [i for e in l for i in e]

def read_file(filename):
    return [['Hi Mr Robot'] + json.loads(line)['turns'] for line in open(filename, 'r')]

def create_qa_pairs(dialogue):
    return [(dialogue[2 * i], dialogue[2 * i + 1]) for i in range(len(dialogue) // 2)]

def filter_pairs(pairs):
    length_constraints = lambda s: MIN_LENGTH <= len(s.split()) <= MAX_LENGTH
    filter_pair = lambda p: all(length_constraints(s) for s in p)
    return [*filter(filter_pair, pairs)]

def get_pairs(dialogues):
    return flatten([*map(create_qa_pairs, dialogues)])

def normalize_sentence(s):
    return s.translate({ord(i): None for i in string.punctuation}).lower()

def normalize_pairs(pairs):
    return [*map(lambda p: tuple(map(normalize_sentence, p)), pairs)]

### Set would work but does not preserve order between module calls
def get_unique_words(pairs):
    all_words = flatten([q.split() + a.split() for (q, a) in pairs])
    return [word for i, word in enumerate(all_words) if word not in all_words[i + 1:]]

def get_unique_words_fast(pairs):
    all_words = flatten([q.split() + a.split() for (q, a) in pairs])
    unique_words = []
    for word in all_words:
        if word not in unique_words:
            unique_words.append(word)
    return unique_words

def encode_pairs(pairs, dictionary):
    return [*map(lambda p: tuple(map(lambda s: encode_sentence(s, dictionary), p)), pairs)]

def create_dataset(filename, create_dictionary = True):
    dialogues = read_file(filename)
    pairs = get_pairs(dialogues)
    filtered_pairs = filter_pairs(pairs)
    normalized_pairs = normalize_pairs(filtered_pairs)
    if create_dictionary:
        unique_words = get_unique_words(normalized_pairs)
        dictionary = {key: value + 2 for value, key in enumerate(reversed(unique_words))}
        encoded_pairs = encode_pairs(normalized_pairs, dictionary)
        final_pairs = [*map(lambda p: (p[0], p[1] + [1]), encoded_pairs)]
        return final_pairs, {**{'SOS': 0, 'EOS': 1}, **dictionary}
    return normalized_pairs

def create_whole_corpus():
    filenames = map(lambda f: f'data/dialogues/{f}', os.listdir('data/dialogues'))
    pairs = flatten(map(lambda f: create_dataset(f, create_dictionary = False), filenames))
    print('Created Pairs')
    unique_words = get_unique_words_fast(pairs)
    dictionary = {key: value for value, key in enumerate(unique_words)}
    print('Creted Dictionary')
    encoded_pairs = encode_pairs(pairs, dictionary)
    print('Encoded Pairs')
    final_pairs = [*map(lambda p: (p[0], p[1] + [1]), encoded_pairs)]
    return final_pairs, {**{'SOS': 0, 'EOS': 1}, **dictionary}

#################################################################################################

def reverse_dictionary(dictionary):
    return {value: key for key, value in dictionary.items()}

def encode_sentence(sentence, dictionary):
    return [dictionary[word] for word in sentence.split()]

def decode_sentence(sentence, reverse):
    return [reverse[index] for index in sentence]

def pair_dimensions(q, a):
    return (len(q), len(a))

