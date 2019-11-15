# Chatbot
### Sequence to Sequence models for Dialogue Learning
* Notebook is more for intuition than performance. 
* Main training file is __parallel_training.py__. You can change encoders and decoders by editing the appropriate files
* Models are autosaved every 20 iterations of the whole corpus and are created in the main directory of the repository with names __encoder__ and __decoder__.
* If you wish to continue training from the checkpoints __parallel_training.py__ has a recover flag in its main function
* __training.py__ is lighter for hardware but much slower. 

#### Codebase was tested on Ubuntu 18.04, Python 3 and PyTroch 1.0.1.post2. No Windows Compatability is expected
