# Chatbot
### Sequence to Sequence models for Dialogue Learning
* Notebook is a good summary of the work but is more for intuition than performance.
* Notebook is trained with attention but on a subset of the data. For the whole corpus no attention is used. I don't have such an expensive GPU.
* Main training file is _parallel_training.py_. You can change encoders and decoders by editing the appropriate files
* Models are autosaved every 20 iterations of the whole corpus and are created in the main directory of the repository with names __encoder__ and __decoder__.
* If you wish to continue training from the checkpoints _parallel_training.py_ has a recover flag in its main function
* _training.py_ is lighter for hardware but much slower. 

#### Codebase was tested on Ubuntu 18.04, Python 3 and PyTroch 1.0.1.post2. No Windows Compatability is expected
#### Sources:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

https://pytorch.org/docs/stable/nn.html

## Sample Conversations:
#### Initial Training
Q: Hi 

A: Hello how may i help you

Q: I am god

A: Yes you are the the

Q: Thank you

A: Have a a

Q: You are bad

A: I agree

Q: I am the programmer

A: Yes you are the person

#### Final Training
Q: Hi Mr Robot

A: How may I help you

Q: I am god

A: You are

Q: I can have anything I want

A: You can

Q: Are you always this agreeable?

A: I am aggreable yes
