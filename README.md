# CNN-Encoder-Decoder



Files descriptions:
* `run_training.py` -- the main script to assign the parameters of the model and to run the model. This script calls the initiaion of the data sets and the model (from the file `utils.py`) and training of the moodel (from the file `nets.py`);
* `nets.py` -- the class for the autoencoder model;
* `train_and_test.py` -- functions for model training, validation and testing;
* `utils.py` -- auxiliary functions for model assembly, fixing the random seed, data loader, etc.
* `requirements.txt` files with required libraries for the scripts to run.

To run the script, type in the terminal:

`>> conda create --name cnn-training`

`>> conda activate cnn-training`
 
`>> conda install pip` 

`>> pip install -r requirements.txt `

`>> python run_training.py`

Link for the data: 

The folder _data/_ from the archive is to replace the spaceholder folder _data/_ in the current repository, i.e. it must be placed a the same level as the _.py_ files.
