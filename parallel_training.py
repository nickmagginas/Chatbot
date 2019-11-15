import read_files as r
import torch
from encoder import Encoder
from decoder import Decoder
from time import sleep

LEARNING_RATE = 0.01
HIDDEN_SIZE = 512
BATCH_SIZE = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



transpose = lambda t: torch.transpose(t, 0, 1)

def parallel_training(pairs):
    dimension_range = range(r.MIN_LENGTH, r.MAX_LENGTH + 1)
    for dimension in [(ql, al) for ql in dimension_range for al in dimension_range]:
        yield [*filter(lambda p: r.pair_dimensions(*p) == dimension , pairs)]

def propagate_encoder(encoder, questions, sequence_size):
    questions = transpose(torch.tensor(questions, dtype = torch.long, device = DEVICE))
    return encoder(questions, encoder.__init_hidden__(HIDDEN_SIZE).repeat(1, sequence_size, 1).to(DEVICE))

loss_function = torch.nn.NLLLoss()

def propagate_decoder(decoder, answers, output, decoder_state, loss_function, accumulator = 0):
    try: target = next(answers)
    except StopIteration: return accumulator
    loss = loss_function(output.squeeze(0), target)
    forward_inputs = torch.argmax(output, dim = 2).detach()
    forward_outputs, update_state = decoder(forward_inputs, decoder_state)
    return propagate_decoder(decoder, answers, forward_outputs, update_state, loss_function, accumulator + loss)

def training_update(encoder_optimizer, decoder_optimizer, loss):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

def training_iteration(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, pairs):
    iteration_loss = 0
    for parallel_pairs in parallel_training(pairs):
        if parallel_pairs == []: continue
        batch_number  = len(parallel_pairs) // BATCH_SIZE 
        if len(parallel_pairs) < BATCH_SIZE:
            sequence_size = len(parallel_pairs)
            questions, answers = tuple(map(list, zip(*parallel_pairs)))
            encoder_outputs, encoder_state = propagate_encoder(encoder, questions, sequence_size)
            output_length = len(answers[0])
            answers = iter(transpose(torch.tensor(answers, dtype = torch.long, device = DEVICE)))
            decoder_start_tokens = torch.tensor([0], dtype = torch.long, device = DEVICE).repeat(1, sequence_size)
            output, decoder_state = decoder(decoder_start_tokens, encoder_state)
            loss = propagate_decoder(decoder, answers, output, decoder_state, loss_function)
            training_update(encoder_optimizer, decoder_optimizer, loss)
            iteration_loss += loss
        else:
            for bi in range(batch_number):
                batch_pairs = parallel_pairs[bi * BATCH_SIZE: (bi + 1) * BATCH_SIZE]
                sequence_size = len(batch_pairs)
                questions, answers = tuple(map(list, zip(*batch_pairs)))
                encoder_outputs, encoder_state = propagate_encoder(encoder, questions, sequence_size)
                output_length = len(answers[0])
                answers = iter(transpose(torch.tensor(answers, dtype = torch.long, device = DEVICE)))
                decoder_start_tokens = torch.tensor([0], dtype = torch.long, device = DEVICE).repeat(1, sequence_size)
                output, decoder_state = decoder(decoder_start_tokens, encoder_state)
                loss = propagate_decoder(decoder, answers, output, decoder_state, loss_function)
                training_update(encoder_optimizer, decoder_optimizer, loss)
                iteration_loss += loss

    return iteration_loss

def train():
    pairs, dictionary = r.create_whole_corpus()
    print(f'Number of pairs: {len(pairs)}')
    encoder = Encoder(dictionary.__len__(), HIDDEN_SIZE).to(DEVICE)
    decoder = Decoder(dictionary.__len__(), HIDDEN_SIZE).to(DEVICE)
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), LEARNING_RATE)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), LEARNING_RATE)
    loss_function = torch.nn.NLLLoss()
    encoder.load_state_dict(torch.load('encoder'))
    decoder.load_state_dict(torch.load('decoder'))
    for i in range(1000):
        loss = training_iteration(encoder, decoder, encoder_optimizer, decoder_optimizer, loss_function, pairs)
        print(f'Iteration: {i}, Loss: {loss}')
        if i % 20 == 0:
            print('Saving Models')
            torch.save(encoder.state_dict(), 'encoder')
            torch.save(decoder.state_dict(), 'decoder')
    
    torch.save(encoder.state_dict(), 'encoder')
    torch.save(decoder.state_dict(), 'decoder')
    

train()

'''
    for output_index in range(output_length):
        output, decoder_state = decoder(decoder_start_tokens, decoder_state)
        target = answers[output_index]
        loss(output.squeeze(0), target)
        predictions = torch.argmax(output, dim = 2)
        break
    break
'''
