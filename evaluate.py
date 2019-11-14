import read_files as r
import torch
from encoder import Encoder
from decoder import Decoder

def evaluate_loop(encoder, decoder, dictionary):
    reverse_dictionary = r.reverse_dictionary(dictionary)
    while True:
        question = r.normalize_sentence(input('Q:'))
        question_length = len(question.split())
        encoder_state = encoder.__init_hidden__(64).repeat(1, 1, 1)
        encoded_sentence = r.encode_sentence(question, dictionary)
        query = torch.tensor(encoded_sentence, dtype = torch.long).view(-1, 1)
        _ , decoder_state = encoder(query, encoder_state)
        decoder_input = torch.tensor([[1]], dtype = torch.long)
        final_output = []
        while True:
            decoder_output, decoder_state = decoder(decoder_input, decoder_state)
            decoder_input = torch.argmax(decoder_output, dim = 2)
            word_index = decoder_input.item()
            final_output = final_output + [word_index]
            if word_index == 1: break
        print(r.decode_sentence(final_output, reverse_dictionary))

pairs, dictionary = r.create_dataset('data/dialogues/AGREEMENT_BOT.txt')
encoder = Encoder(dictionary.__len__(), 64)
decoder = Decoder(dictionary.__len__(), 64)
evaluate_loop(encoder, decoder, dictionary)

