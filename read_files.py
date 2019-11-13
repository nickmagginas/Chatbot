import json
import string

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

def create_dataset(filename):
    dialogues = read_file(filename)
    pairs = get_pairs(dialogues)
    filtered_pairs = filter_pairs(pairs)
    normalized_pairs = normalize_pairs(filtered_pairs)
    unique_words = get_unique_words(normalized_pairs)
    dictionary = {key: value + 2 for value, key in enumerate(unique_words)}
    return pairs, {**{'SOS': 1, 'EOS': 2}, **dictionary}



#################################################################################################

def reverse_dictionary(dictionary):
    return {value: key for key, value in dictionary.items()}

def encode_sentence(sentence, dictionary):
    return [dictionary[word] for word in sentence.split()]

def decode_sentence(sentence, reverse):
    return [reverse[index] for index in sentence]
