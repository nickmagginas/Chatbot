import read_files as r
import torch
from encoder import Encoder
from decoder import Decoder

LEARNING_RATE = 0.01
HIDDEN_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transpose = lambda t: torch.transpose(t, 0, 1)

def propagate_encoder(encoder, question, state):
	try: encoder_input = next(question).view(1, 1)
	except StopIteration: return state
	_ , encoder_state = encoder(encoder_input, state)
	return propagate_encoder(encoder, question, encoder_state)

def propagate_decoder(decoder, answer, decoder_input, state, loss_function, loss = 0):
	try: target = next(answer).view(1)
	except StopIteration: return loss
	decoder_prediction, decoder_state = decoder(decoder_input, state)
	sample_loss = loss_function(decoder_prediction.squeeze(0), target)
	next_input = torch.argmax(decoder_prediction, dim = 2).detach()
	if next_input.item() == 1: return loss
	return propagate_decoder(decoder, answer, next_input, decoder_state, loss_function, loss + sample_loss)

def train():
	pairs, dictionary = r.create_dataset('data/dialogues/ALARM_SET.txt')
	encoder = Encoder(dictionary.__len__(), HIDDEN_SIZE).to(DEVICE)
	encoder_optimizer = torch.optim.SGD(encoder.parameters(), LEARNING_RATE)
	decoder = Decoder(dictionary.__len__(), HIDDEN_SIZE).to(DEVICE)
	decoder_optimizer = torch.optim.SGD(decoder.parameters(), LEARNING_RATE)
	loss_function = torch.nn.NLLLoss()
	for iteration in range(50):
		iteration_loss = 0
		for (question, answer) in pairs:
			question = iter(torch.tensor(question, dtype = torch.long, device = DEVICE))
			initial_hidden_state = encoder.__init_hidden__(HIDDEN_SIZE).repeat(1, 1, 1).to(DEVICE)
			encoder_state = propagate_encoder(encoder, question, initial_hidden_state)
			answer = iter(torch.tensor(answer, dtype = torch.long, device = DEVICE))
			initial_decoder_token = torch.tensor([[0]], device = DEVICE)
			loss = propagate_decoder(decoder, answer, initial_decoder_token, encoder_state, loss_function)
			if loss != 0:
				encoder_optimizer.zero_grad()
				decoder_optimizer.zero_grad()
				loss.backward() 
				encoder_optimizer.step()
				decoder_optimizer.step()
				iteration_loss += loss

		print(f'Iteration: {iteration}. Loss: {iteration_loss / len(pairs)}')
	torch.save(encoder.state_dict(), 'np encoder')
	torch.save(decoder.state_dict(), 'np decoder')


train()