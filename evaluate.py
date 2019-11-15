import read_files as r
import torch
from encoder import Encoder
from decoder import Decoder
from time import sleep

def evaluate_loop(encoder, decoder, dictionary):
    reverse_dictionary = r.reverse_dictionary(dictionary)
    while True:
        question = r.normalize_sentence(input('Q:'))
        question_length = len(question.split())
        encoder_state = encoder.__init_hidden__(512).repeat(1, 1, 1)
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
            print(r.decode_sentence(final_output, reverse_dictionary))
            sleep(1)
            if word_index == 1: break
        print(r.decode_sentence(final_output, reverse_dictionary))

pairs, dictionary = r.create_whole_corpus()
print(len(dictionary))
encoder = Encoder(dictionary.__len__(), 512)
decoder = Decoder(dictionary.__len__(), 512)
encoder.load_state_dict(torch.load('encoder'))
decoder.load_state_dict(torch.load('decoder'))
evaluate_loop(encoder, decoder, dictionary)

