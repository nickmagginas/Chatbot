import json
import torch
import random

FILENAME = '../data/dialogues/AGREEMENT_BOT.txt'
HIDDEN_SIZE = 256 
LEARNING_RATE = 0.005

### Flatten List
flatten = lambda l: [s for e in l for s in e]

### JSON parse and prepend Greeting
def read_file():
	return [['Hi'] + json.loads(line)['turns'] for line in open(FILENAME, 'r')]

original_lines = read_file()
lines = flatten(original_lines)

### Set of unique words -- Does not retain order between module calls
unique_words = set(flatten([line.split() for line in lines]))

### Lookup Index -- Index to Word
dictionary = {**{0: 'SOS', 1: 'EOS'}, **{key + 2: value for key, value in enumerate(unique_words)}} 

### Lookup Words -- Words to Index
reverse_dictionary = {value: key for key, value in dictionary.items()} 

### Built Pairs out of dialogue
def construct_pair(line):
	return [(line[2*i], line[2*i + 1] + ' EOS') for i in range(len(line) // 2)]

def construct_pairs(lines):
	return [*map(construct_pair, lines)]

pairs = flatten(construct_pairs(original_lines))

### Encode Sentence need reverse dictionary -- Words to Index
def encode_sentence(sentence, dictionary):
	return [dictionary[word] for word in sentence.split()]

### Encodes whole pair
def encode_pair(pair, dictionary):
	return tuple(map(lambda s: encode_sentence(s, dictionary), pair))

encoded_pairs = [*map(lambda p: encode_pair(p, reverse_dictionary), pairs)]

def prepare_pairs(pairs):
	tensorize = lambda s: torch.tensor(s, dtype = torch.long).view(-1, 1)
	return [tuple(map(tensorize, pair)) for pair in pairs]

final_pairs = prepare_pairs(encoded_pairs)
input_size = dictionary.__len__()

class Encoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Encoder, self).__init__()
		self.embedding = torch.nn.Embedding(input_size, hidden_size)
		self.gru = torch.nn.GRU(hidden_size, hidden_size)

	def forward(self, x, s):
		embedded = self.embedding(x).view(1, 1, -1)
		return self.gru(embedded, s)

	@staticmethod
	def __init_hidden__(hidden_size): return torch.zeros(1, 1, hidden_size)

class Decoder(torch.nn.Module):
	def __init__(self, hidden_size, output_size):
		super(Decoder, self).__init__()
		self.embedding = torch.nn.Embedding(output_size, hidden_size)
		self.gru = torch.nn.GRU(hidden_size, hidden_size)
		self.fc = torch.nn.Linear(hidden_size, output_size)
		self.softmax = torch.nn.LogSoftmax(dim = 2)

	def forward(self, x, s):
		embedded = torch.nn.functional.relu(self.embedding(x))
		output, hidden = self.gru(embedded, s)
		fc = self.fc(output)
		return self.softmax(fc), hidden

def train(pairs, iterations):
	encoder = Encoder(input_size, HIDDEN_SIZE)
	encoder_state = encoder.__init_hidden__(HIDDEN_SIZE)

	decoder = Decoder(HIDDEN_SIZE, input_size)


	encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr = LEARNING_RATE)
	decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr = LEARNING_RATE)

	loss_function = torch.nn.NLLLoss()
	for i in range(iterations):
		pair = random.choice(pairs)
		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()
		loss = pair_loss(pair, encoder, decoder, encoder_state, loss_function)
		loss.backward()
		encoder_optimizer.step() 
		decoder_optimizer.step()
		print(f'Iteration: {i}, Loss: {loss.item()}')

	return encoder, decoder

	

def pair_loss(pair, encoder, decoder, encoder_state, loss_function):
	pair_loss = 0
	question, target = pair
	iterable_question = iter(question)
	encoder_state = propagate_encoder_state(encoder, iterable_question, encoder_state)
	decoder_input = torch.tensor([[0]])
	decoder_state = encoder_state
	for target_word in target:
		decoder_output, decoder_state = decoder(decoder_input, decoder_state)
		decoder_input = torch.argmax(decoder_output, dim = 2).detach()
		pair_loss += loss_function(decoder_output.view(1, -1), target_word)

	return pair_loss

def get_answer(encoder, decoder, question, encoder_state):
	iterable_question = iter((lambda s: torch.tensor(s, dtype = torch.long).view(-1, 1))(question))
	encoder_state = propagate_encoder_state(encoder, iterable_question, encoder_state)
	decoder_input = torch.tensor([[0]])
	decoder_state = encoder_state
	answer = []
	while True:
		decoder_output, decoder_state = decoder(decoder_input, decoder_state)
		decoder_input = torch.argmax(decoder_output, dim = 2).detach()
		answer += [decoder_input.item()]
		if decoder_input == 1:
			break
	return answer



def propagate_encoder_state(encoder, sentence, state):
	try: word = next(sentence)
	except StopIteration: return state
	_, next_state = encoder(word, state)
	return propagate_encoder_state(encoder, sentence, next_state)


encoder, decoder = train(final_pairs, 10000)

def eval_loop():
	while True:
		q = input('Q: ')
		encoded = encode_sentence(q, reverse_dictionary)
		print(get_answer(encoder, decoder, encoded, encoder.__init_hidden__(HIDDEN_SIZE)))

eval_loop()















