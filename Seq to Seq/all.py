import json
import torch
import string
import random

FILENAMES = ['../data/dialogues/AGREEMENT_BOT.txt', '../data/dialogues/ALARM_SET.txt', '../data/dialogues/BUS_SCHEDULE_BOT']
HIDDEN_SIZE = 512
LEARNING_RATE = 0.005
MAX_LENGTH = 15

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Flatten List
flatten = lambda l: [s for e in l for s in e]

def normalize_sentence(s):
	return ''.join([*map(lambda w: w.translate({ord(i): None for i in string.punctuation}).lower(), s)])

### JSON parse and prepend Greeting
def read_file(filename):
	return [['Hi'] + json.loads(line)['turns'] for line in open(filename, 'r')]

original_lines = read_file(FILENAMES[0]) + read_file(FILENAMES[1]) + read_file(FILENAMES[2])
lines = flatten(original_lines)
normalized_lines = [*map(normalize_sentence, lines)]

### Set of unique words -- Does not retain order between module calls
unique_words = set(flatten([line.split() for line in normalized_lines]))

### Lookup Index -- Index to Word
dictionary = {**{0: 'SOS', 1: 'EOS'}, **{key + 2: value for key, value in enumerate(unique_words)}} 

json.dump(dictionary, open('dictionary', 'w'))
### dictionary = json.load(open('dictionary', 'r'))

### Lookup Words -- Words to Index
reverse_dictionary = {value: key for key, value in dictionary.items()} 

def filter_pair(pair):
	question, answer = pair
	return len(question) < MAX_LENGTH and len(answer) < MAX_LENGTH

### Filter length
def filter_pairs(pairs):
	return [*filter(filter_pair, pairs)]

### Built Pairs out of dialogue
def construct_pair(line):
	return [(normalize_sentence(line[2*i]), normalize_sentence(line[2*i + 1]) + ' EOS') for i in range(len(line) // 2)]

def construct_pairs(lines):
	return [*map(construct_pair, lines)]

pairs = flatten(construct_pairs(original_lines))

def decode_sentence(sentence, dictionary):
	return [dictionary[index] for index in sentence]

### Encode Sentence need reverse dictionary -- Words to Index
def encode_sentence(sentence, dictionary):
	return [dictionary[word] for word in sentence.split()]

### Encodes whole pair
def encode_pair(pair, dictionary):
	return tuple(map(lambda s: encode_sentence(s, dictionary), pair))

encoded_pairs = [*map(lambda p: encode_pair(p, reverse_dictionary), pairs)]

### Move to Tensor Data and GPU if available
def prepare_pairs(pairs):
	tensorize = lambda s: torch.tensor(s, dtype = torch.long, device = DEVICE).view(-1, 1)
	return [tuple(map(tensorize, pair)) for pair in pairs]

final_pairs = filter_pairs(prepare_pairs(encoded_pairs))
input_size = dictionary.__len__()

### Vanilla Encoder with Embedding Layer
class Encoder(torch.nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Encoder, self).__init__()
		self.embedding = torch.nn.Embedding(input_size, hidden_size)
		self.gru = torch.nn.GRU(hidden_size, hidden_size)

	def forward(self, x, s):
		embedded = self.embedding(x).view(1, 1, -1)
		return self.gru(embedded, s)

	@staticmethod
	def __init_hidden__(hidden_size): return torch.zeros(1, 1, hidden_size, device = DEVICE)

### Vannila Decoder with Embedding Layer -- To Add Attention
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

### Main training routine
def train(pairs, iterations):
	encoder = Encoder(input_size, HIDDEN_SIZE).to(DEVICE)
	encoder_state = encoder.__init_hidden__(HIDDEN_SIZE)

	decoder = Decoder(HIDDEN_SIZE, input_size).to(DEVICE)


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
		print(f'Iteration: {i} , Loss: {loss.item()}')

	torch.save(encoder.state_dict(), 'encoder')
	torch.save(decoder.state_dict(), 'decoder')

	return encoder, decoder

### Calculate Loss for single QA Pair
def pair_loss(pair, encoder, decoder, encoder_state, loss_function):
	pair_loss = 0
	question, target = pair
	iterable_question = iter(question)
	encoder_state = propagate_encoder_state(encoder, iterable_question, encoder_state)
	decoder_input = torch.tensor([[0]], device = DEVICE)
	decoder_state = encoder_state
	for target_word in target:
		decoder_output, decoder_state = decoder(decoder_input, decoder_state)
		decoder_input = torch.argmax(decoder_output, dim = 2).detach()
		pair_loss += loss_function(decoder_output.view(1, -1), target_word)

		if decoder_input.item() == 2:
			break

	return pair_loss

### For Test Feed Question get answer
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

### Propagate the state to get final encoding
def propagate_encoder_state(encoder, sentence, state):
	try: word = next(sentence)
	except StopIteration: return state
	_, next_state = encoder(word, state)
	return propagate_encoder_state(encoder, sentence, next_state)


encoder, decoder = train(final_pairs, 100000)

encoder = Encoder(input_size, HIDDEN_SIZE)
decoder = Decoder(HIDDEN_SIZE, input_size)
encoder.load_state_dict(torch.load('encoder'))
decoder.load_state_dict(torch.load('decoder'))

### Simple Eval Loop for Conversing
def eval_loop():
	while True:
		q = normalize_sentence(input('Q: '))
		try: 
			encoded = encode_sentence(q, reverse_dictionary)
			answer = get_answer(encoder, decoder, encoded, encoder.__init_hidden__(HIDDEN_SIZE).to(torch.device('cpu')))
			print(decode_sentence(answer, dictionary))
		except KeyError: print('I have never seen something like this before')

	
eval_loop()















