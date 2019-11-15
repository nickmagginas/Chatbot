# Chatbot
### Sequence to Sequence models for Dialogue Learning
* Notebook is a good summary of the work but is more for intuition than performance.
* Main training file is _parallel_training.py_. You can change encoders and decoders by editing the appropriate files
* Models are autosaved every 20 iterations of the whole corpus and are created in the main directory of the repository with names __encoder__ and __decoder__.
* If you wish to continue training from the checkpoints _parallel_training.py_ has a recover flag in its main function
* _training.py_ is lighter for hardware but much slower. 

#### Codebase was tested on Ubuntu 18.04, Python 3 and PyTroch 1.0.1.post2. No Windows Compatability is expected
